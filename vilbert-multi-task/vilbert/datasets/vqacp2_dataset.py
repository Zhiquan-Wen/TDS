"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from mio import MioWriter, MIO
import struct
COUNTING_ONLY = False
from vilbert.datasets import utils

# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
            ('number of' in q.lower() and 'number of the' not in q.lower()) or \
                    'amount of' in q.lower() or \
                    'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False

def _create_entry(img, question, answer):
    if None != answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


def _load_dataset(dataroot, name, label2ans,ratio=1.0):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'test'
    """
    question_path = os.path.join(dataroot, 'vqacp_v2_%s_questions.json' % (name))
    questions = sorted(json.load(open(question_path)), key=lambda x: x['question_id'])

    # train, val
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])[0:len(questions)]

    utils.assert_eq(len(questions), len(answers))

    if ratio < 1.0:
        # sampling traing instance to construct smaller training set.
        index = random.sample(range(0,len(questions)), int(len(questions)*ratio))
        questions_new = [questions[i] for i in index]
        answers_new = [answers[i] for i in index]
    else:
        questions_new = questions
        answers_new = answers

    entries = []
    for question, answer in zip(questions_new, answers_new):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
            entries.append(_create_entry(img_id, question, answer))

    return entries


class VQAClassificationDataset(Dataset):
    def __init__(self, name, dataroot, img_root, tokenizer, ratio, padding_idx, max_num_obj, max_seq_length):
        super(VQAClassificationDataset, self).__init__()
        assert name in ['train', 'test']

        ans2label_path = os.path.join(dataroot, 'cache', 'train_test_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'train_test_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.img_info = np.load(os.path.join(dataroot, 'cache', 'imgid_to_height_and_width.npy'), allow_pickle=True)[0]

        self._tokenizer = tokenizer
        self._padding_index = padding_idx
        self.max_obj_num = max_num_obj
        self.max_seq_length = max_seq_length


        print('loading image features in MIO')
        # Load image features and bounding boxes
        self.m = MIO(img_root)
        print('loading image features in MIO done!')

        self.ids = {}
        for i in range(self.m.size):
            id_= struct.unpack("<I", self.m.get_collection_metadata(i))[0]
            self.ids[id_] = i

        self.entries = _load_dataset(dataroot, name, self.label2ans, ratio)
        self.tokenize()
        self.tensorize()

        self.v_dim = 2048

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["question"])
            tokens = tokens[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        img_id = entry['image_id']
        true_feature_id = self.ids[img_id]
        feature = self.m.fetchone(colletion_id=true_feature_id, object_id=1)
        features = torch.from_numpy(np.frombuffer(feature, dtype=np.float32).reshape(2048, 36).copy()).permute(1, 0)
        box = self.m.fetchone(colletion_id=true_feature_id, object_id=0)
        boxes = torch.from_numpy(np.frombuffer(box, dtype=np.float32).reshape(4, 36).copy()).permute(1, 0)
        normalize_boxes = torch.from_numpy(self.compute_lfeats(img_id, boxes))

        image_mask = [1] * (int(self.max_obj_num))
        image_mask = torch.tensor(image_mask).long()

        question = entry['q_token']
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]
        co_attention_mask = torch.zeros((self.max_obj_num, self.max_seq_length))
        question_id = entry['question_id']
        answer = entry['answer']
        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, normalize_boxes, question, target, image_mask, input_mask, segment_ids, co_attention_mask, question_id
        else:
            print("No label")
            return features, normalize_boxes, question, image_mask, input_mask, segment_ids, co_attention_mask, question_id

    def __len__(self):
        return len(self.entries)

    def compute_lfeats(self, imgid, boxes):
        lfeats = np.zeros((self.max_obj_num, 5))
        img_info = self.img_info[imgid]
        for ix, box in enumerate(boxes):
            lfeats[ix] = np.array([box[0] / img_info['w'], box[1] / img_info['h'], box[2] / img_info['w'],
                                   box[3] / img_info['h'], ((box[2]-box[0]+1)*(box[3]-box[1]+1))/ (img_info['w']*img_info['h'])])
        return lfeats
