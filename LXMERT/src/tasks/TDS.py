"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import json
# import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataset_vqacp_lxmert import Dictionary, VQAFeatureDataset
from tasks.vqa_model import VQAModel
import utils_1
from src.param import args as opt
import os
from tqdm import tqdm
import random
import torch.nn.functional as F

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]

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
    random.seed(777)

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(os.path.join(opt.dataroot, 'dictionary.pkl'))
    opt.ntokens = dictionary.ntoken

    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root, 1.0, adaptive=False)

    model = VQAModel(2274)
    model = model.cuda()

    Entropy = 7.7293 * opt.rate # max 7.7293, select the threshold *0.2

    print(model)

    eval_loader = DataLoader(eval_dset, opt.batch_size, shuffle=True, num_workers=1, collate_fn=utils_1.trim_collate)

    def process(opt, model, eval_loader, self_sup=True):

        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)
        model_data = {key.replace('module.',''): value for key, value in model_data.items()}
        model.load_state_dict(model_data)

        model = nn.DataParallel(model).cuda()
        optim = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)

        N = len(eval_loader.dataset)
        M = eval_loader.dataset.num_ans_candidates
        K = 36
        pred = torch.FloatTensor(N, M).zero_()
        qIds = torch.IntTensor(N).zero_()
        idx = 0

        for it, (v, b, q, a, i) in enumerate(eval_loader):
            batch_size = v.size(0)
            v = v.cuda()
            b = b.cuda()
            model.train(False)
            logits = model(v, b, list(q))
            pred[idx:idx+batch_size,:].copy_(logits.data)
            qIds[idx:idx+batch_size].copy_(i)

            model.train(True)
            logits_ = model(v, b, list(q))
            if self_sup:
                index_v = random.sample(range(0, batch_size), batch_size)
                gv_neg = v[index_v]
                logits_neg_v = model(gv_neg, b, list(q))

            optim.zero_grad()
            loss = softmax_entropy(logits_)

            loss_mask = (loss <= Entropy).cuda()

            gt_predict = torch.argmax(logits_, dim=1)
            neg_v_predict = torch.argmax(logits_neg_v, dim=1)

            mask_v = (gt_predict == neg_v_predict)

            bias_sample = mask_v & loss_mask
            loss_min_logits, _ = torch.max(F.softmax(logits_neg_v, dim=1), dim=1)
            loss_min_logits_, _ = torch.max(F.softmax(logits_, dim=1), dim=1)
            loss_logits = (bias_sample.float() * loss_min_logits).sum() / sum(bias_sample.float())
            loss_logits_ = (bias_sample.float() * loss_min_logits_).sum() / sum(bias_sample.float())

            loss_mask = loss_mask & (1 - mask_v)

            loss = (loss * loss_mask.float()).sum() / sum(loss_mask.float())

            total_loss = loss + loss_logits + loss_logits_
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            idx += batch_size

            print("iter: {}, loss: {}".format(it, total_loss.item()))

        results = make_json(pred, qIds, eval_loader)

        utils_1.create_dir(opt.output)

        with open(os.path.join(opt.output,'test.json'), 'w') as f:
            json.dump(results, f)

    process(opt, model, eval_loader)