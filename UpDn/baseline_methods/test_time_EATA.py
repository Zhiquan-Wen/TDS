"""
python test_time_EATA.py --learning_rate 0.001 --batch_size 512 --threshold 0.2 
--checkpoint_path --dataroot  --img_root
"""
import argparse
import json
from multiprocessing import current_process
import random
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from TDS.UpDn.dataset_mio_vqacp import Dictionary, VQAFeatureDataset
import TDS.base_model as base_model
import TDS.UpDn.utils as utils
import os
import torch.nn.functional as F
from tqdm import tqdm

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

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

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
    parser.add_argument('--rate', type=float, default=0.4)

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

    n_device = torch.cuda.device_count()
    batch_size = opt.batch_size

    constructor = 'build_%s' % opt.model
    model = getattr(base_model, constructor)(eval_dset, opt.num_hid)

    Entropy = 7.7293 * opt.rate # max 7.7293, select the threshold *0.4

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
        d_margin = 0.5

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

            model.train(True)
            logits_ = model(v, b, q, mask, a)
            optim.zero_grad()
            loss = softmax_entropy(logits_)
            filter_ids_1 = loss <= Entropy
            ids1 = filter_ids_1
            entropys = loss[filter_ids_1] 

            if it == 0:
                current_model_probs = None

            if current_model_probs is not None: 
                cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), logits_[filter_ids_1].softmax(1), dim=1)
                filter_ids_2 = torch.abs(cosine_similarities) < d_margin
                entropys = entropys[filter_ids_2]
                ids2 = filter_ids_2
                updated_probs = update_model_probs(current_model_probs, logits_[filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = update_model_probs(current_model_probs, logits_[filter_ids_1].softmax(1))
            
            current_model_probs = updated_probs
            
            coeff = 1 / (torch.exp(entropys.clone().detach() - Entropy))

            entropys = entropys.mul(coeff)

            loss = entropys.mean()

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