# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel

import torch.nn.functional as F

# multi-label soft loss
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

@torch.no_grad()
def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    entropy = 0
    for i, (v, b, q, a, q_id) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        q = list(q)
        q_id = q_id.cuda()
        pred = model(v, b, q)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score.item()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    return score, upper_bound


def get_our_data():
    from dataset_vqacp1_lxmert import Dictionary, VQAFeatureDataset
    import utils_1
    from src.param import args as opt

    dictionary = Dictionary.load_from_file(os.path.join(opt.dataroot, 'dictionary.pkl'))
    opt.ntokens = dictionary.ntoken

    train_dset = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio, adaptive=False)  # load labeld data
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root,ratio=1.0, adaptive=False)

    train_loader = DataLoader(train_dset, opt.batch_size, shuffle=True, num_workers=1, collate_fn=utils_1.trim_collate)
    opt.use_all = 1
    val_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=1, collate_fn=utils_1.trim_collate)
    return train_loader, val_loader

class VQA:
    def __init__(self):
        # Datasets
        self.train_loader, self.val_loader = get_our_data()
        self.model = VQAModel(1691)

        # Load pre-trained weights
        # if args.load_lxmert is not None:
        #     self.model.lxrt_encoder.load(args.load_lxmert)
        # if args.load_lxmert_qa is not None:
        #     load_lxmert_qa(args.load_lxmert_qa, self.model,
        #                    label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_loader, val_loader):
        best_valid = 0.

        # train
        total_num = len(train_loader.dataset)
        for epoch in range(args.epochs):
            total_loss = 0
            train_score = 0
            
            for i, (feats, boxes, sent, target, ques_id) in tqdm(enumerate(train_loader)):

                self.model.train()
                self.optim.zero_grad()
                batch_size = feats.size(0)

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logits = self.model(feats, boxes, list(sent))

                bce_loss = instance_bce_with_logits(
                    logits, target, reduction='mean')

                total_loss += bce_loss.item() * batch_size

                bce_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score = compute_score_with_logits(logits, target.data).sum()
                train_score += score.item()

                if i and i%100==0:
                    log_str = 'traing: %d/%d, bce_loss: %.6f, train_acc: %.6f' %(i, len(train_loader), total_loss / total_num,
                     100 * train_score / total_num)          
                    print(log_str)

            self.save("LAST")

            # if self.valid_tuple is not None:  # Do Validation
            self.model.train(False)
            valid_score, upper_bound = evaluate(self.model, val_loader)
            self.model.train(True)
            if valid_score > best_valid:
                best_valid = valid_score
                self.save("BEST")

            log_str = "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                        "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str)

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()
            
            # if epoch == 9:
            #     break

        return best_valid

    def save(self, name):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    vqa = VQA()
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    # if args.load is not None:
    #     vqa.load(args.load)
    vqa.train(vqa.train_loader, vqa.val_loader)



