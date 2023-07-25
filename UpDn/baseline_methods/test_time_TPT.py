"""
batch_size 512 learning_rate 0.001
"""
import argparse
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from TDS.UpDn.dataset_mio_vqacp import Dictionary, VQAFeatureDataset
import TDS.base_model as base_model
import TDS.UpDn.utils as utils
import os
from tqdm import tqdm

def select_confident_samples(logits, top): # logits: batch, num_aug, num_cls
    batch_entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1) # batch, num_aug
    idx = torch.argsort(batch_entropy, descending=False)[:, :int(batch_entropy.size()[0] * top)]
    logits_select = torch.gather(logits, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, logits.size()[-1]))
    return logits_select

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [b, N, 1000]
    avg_logits = logits.logsumexp(dim=1) - np.log(logits.shape[1]) # avg_logits = logits.mean(0) [b, 1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

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
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--img_root', type=str, default=' ')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)

    args = parser.parse_args()
    return args

# augment 8 tims，choose 6 times update
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

    print(model)

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)

    def process(opt, model, eval_loader):

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
            model.train(False)
            logits = model(v, b, q, mask, a)
            pred[idx:idx+batch_size,:].copy_(logits.data)
            qIds[idx:idx+batch_size].copy_(i)
            aug_pred = []

            model.train(True)
            aug_mask = mask.clone()
            aug_mask = aug_mask.unsqueeze(0).repeat(9, 1, 1)
            aug_mask = aug_mask.cuda()

            logits_ = model(v, b, q, mask, a)
            aug_pred.append(logits_)
            aug_mask_idx = [[random.sample(range(0, mask.size()[1]), 5)] for _ in range(9)]
            for aug_idx, aug_mask_index in  enumerate(aug_mask_idx):
                aug_mask_index_ = torch.from_numpy(np.array(aug_mask_index)).cuda()
                aug_mask[aug_idx].index_fill_(1, aug_mask_index_.squeeze(), 0)
                logits_aug = model(v, b, q, aug_mask[aug_idx], a)
                aug_pred.append(logits_aug)

            selected_logits = select_confident_samples(torch.stack(aug_pred, dim=1), 0.60)
            loss = avg_entropy(selected_logits).mean()

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            idx += batch_size

            print("iter: {}, loss: {}".format(it, loss.item()))

        results = make_json(pred, qIds, eval_loader)
        save_path = opt.output + "_seed_{}".format(seed)
        utils.create_dir(save_path)

        with open(os.path.join(save_path,'test.json'), 'w') as f:
            json.dump(results, f)

    process(opt, model, eval_loader)

if __name__ == '__main__':
    opt = parse_args()

    main(opt)