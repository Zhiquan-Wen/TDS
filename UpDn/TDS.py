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
from dataset_mio_vqacp import Dictionary, VQAFeatureDataset
import base_model
import utils
import os
from tqdm import tqdm
import random
import torch.nn.functional as F

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
 
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp1_num_hid_512')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--dataroot', type=str, default='/mnt/cephfs/dataset/wenzhiquan/VQACP2/vqacp2')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--img_root', type=str, default=' ')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--rate', type=float, default=0.2)

    args = parser.parse_args()
    return args

# if __name__ == '__main__':
def main(opt):

    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True  # cudnn

    dictionary = Dictionary.load_from_file(os.path.join(opt.dataroot, 'dictionary.pkl'))
    opt.ntokens = dictionary.ntoken
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root, 1.0, adaptive=False)

    batch_size = opt.batch_size

    constructor = 'build_%s' % opt.model
    model = getattr(base_model, constructor)(eval_dset, opt.num_hid)

    Entropy = 7.7293 * opt.rate # max entropy 7.7293, select the threshold *rate

    print(model)

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)

    def process(opt, model, eval_loader, self_sup=True):

        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))
        optim = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)

        N = len(eval_loader.dataset)
        M = eval_loader.dataset.num_ans_candidates
        K = 36
        pred = torch.FloatTensor(N, M).zero_()
        qIds = torch.IntTensor(N).zero_()
        idx = 0

        for it, (v, b, q, a, mask, i) in enumerate(eval_loader):
            batch_size = v.size(0)
            v = v.cuda()
            q = q.cuda()
            b = b.cuda()
            a = a.cuda()
            # predict first
            model.train(False)
            logits = model(v, b, q, mask, a)
            pred[idx:idx+batch_size,:].copy_(logits.data)
            qIds[idx:idx+batch_size].copy_(i)

            # then train the model
            model.train(True)
            logits_ = model(v, b, q, mask, a)
            if self_sup:
                index_v = random.sample(range(0, batch_size), batch_size)
                gv_neg = v[index_v]
                logits_neg_v = model(gv_neg, b, q, mask, a)

            optim.zero_grad()
            loss = softmax_entropy(logits_)

            # filter samples with high entropy
            loss_mask = (loss <= Entropy).cuda()

            gt_predict = torch.argmax(logits_, dim=1)
            neg_v_predict = torch.argmax(logits_neg_v, dim=1)

            # filter biased samples
            mask_v = (gt_predict == neg_v_predict)

            # recognize true biased samples
            bias_sample = mask_v & loss_mask
            loss_min_logits, _ = torch.max(F.softmax(logits_neg_v, dim=1), dim=1)
            loss_min_logits_, _ = torch.max(F.softmax(logits_, dim=1), dim=1)
            loss_logits = (bias_sample.float() * loss_min_logits).sum() / sum(bias_sample.float())
            loss_logits_ = (bias_sample.float() * loss_min_logits_).sum() / sum(bias_sample.float())

            # fitler the gradient of both biased and high entropy samples 
            loss_mask = loss_mask & (1 - mask_v)

            loss = (loss * loss_mask.float()).sum() / sum(loss_mask.float() + 1e-5) 

            total_loss = loss + loss_logits + loss_logits_

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            idx += batch_size

            print("iter: {}, loss: {}".format(it, total_loss.item()))

        results = make_json(pred, qIds, eval_loader)
        
        save_path = opt.output
        utils.create_dir(save_path)

        with open(os.path.join(save_path,'test.json'), 'w') as f:
            json.dump(results, f)

    process(opt, model, eval_loader)

if __name__ == '__main__':
    opt = parse_args()

    main(opt)