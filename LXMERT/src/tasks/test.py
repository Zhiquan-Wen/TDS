"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataset_vqacp_lxmert import Dictionary, VQAFeatureDataset
from tasks.vqa_model import VQAModel
import utils_1
from src.param import args as opt

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


@torch.no_grad()
def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    K = 36
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    entropy = []
    bar = progressbar.ProgressBar(maxval=N or None).start()
    for v, b, q, a, i in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        v = v.cuda()
        b = b.cuda()
        logits = model(v, b, list(q))
        entropy += softmax_entropy(logits).data.cpu().numpy().tolist()
        pred[idx:idx+batch_size,:].copy_(logits.data)
        qIds[idx:idx+batch_size].copy_(i)
        idx += batch_size

    bar.update(idx)
    return pred, qIds, entropy


def make_json(logits, qIds, dataloader):
    utils_1.assert_eq(logits.size(0), len(qIds))
 
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

if __name__ == '__main__':
    # opt = opts.parse_opt()

    torch.backends.cudnn.benchmark = True

    opt.dataroot = '/mnt/cephfs/dataset/wenzhiquan/VQACP2/vqacp2/'

    dictionary = Dictionary.load_from_file('/mnt/cephfs/dataset/wenzhiquan/VQACP2/vqacp2/dictionary.pkl')
    opt.ntokens = dictionary.ntoken

    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root, 1.0, adaptive=False)

    n_device = torch.cuda.device_count()
    batch_size = opt.batch_size * n_device


    model = VQAModel(2274)
    model = model.cuda()

    # eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils_1.trim_collate)
    eval_loader = DataLoader(eval_dset, 1000, shuffle=False, num_workers=1, collate_fn=utils_1.trim_collate)

    def process(args, model, eval_loader):

        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)
        model_data = {key.replace('module.',''): value for key, value in model_data.items()}
        model.load_state_dict(model_data)

        model = nn.DataParallel(model).cuda()
        # opt.s_epoch = model_data['epoch'] + 1

        model.train(False)

        logits, qIds, entropy = get_logits(model, eval_loader)
        results = make_json(logits, qIds, eval_loader)
        model_label = 'best'  # opt.label 
        
        utils_1.create_dir(opt.output)

        with open(opt.output+'/test_%s.json' \
            % (model_label), 'w') as f:
            json.dump(results, f)
        
        np.save('./entropy.npy', entropy)

    process(opt, model, eval_loader)
