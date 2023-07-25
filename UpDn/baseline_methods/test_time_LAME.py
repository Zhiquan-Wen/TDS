"""
python test_time_LAME.py --batch_size 512 --learning_rate 0.001 
"""
import argparse
import json
import random
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from TDS.UpDn.dataset_mio_vqacp import Dictionary, VQAFeatureDataset
import base_model_LAME
import TDS.UpDn.utils as utils
import os
from tqdm import tqdm
from lame import LAME


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()


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

    n_device = torch.cuda.device_count()
    batch_size = opt.batch_size

    constructor = 'build_%s' % opt.model
    model = getattr(base_model_LAME, constructor)(eval_dset, opt.num_hid)

    print(model)

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)

    def process(opt, model, eval_loader):

        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))

        N = len(eval_loader.dataset)
        M = eval_loader.dataset.num_ans_candidates
        K = 36
        pred = torch.FloatTensor(N, M).zero_()
        qIds = torch.IntTensor(N).zero_()
        idx = 0
        model.train(False)
        lame = LAME(model, 5, 1.0)

        # opt.s_epoch = model_data['epoch'] + 1

        for it, (v, b, q, a, mask, i) in enumerate(eval_loader):
            batch_size = v.size(0)
            v = v.cuda()
            q = q.cuda()
            b = b.cuda()
            a = a.cuda()
            logits = lame.infer(v, b, q, a, mask)
            pred[idx:idx+batch_size,:].copy_(logits.data)
            qIds[idx:idx+batch_size].copy_(i)

            idx += batch_size

            print("iter: {}, loss: {}".format(it, 0))

        results = make_json(pred, qIds, eval_loader)
        save_path = opt.output
        utils.create_dir(save_path)

        with open(os.path.join(save_path,'test.json'), 'w') as f:
            json.dump(results, f)

    process(opt, model, eval_loader)

if __name__ == '__main__':
    opt = parse_args()

    main(opt)