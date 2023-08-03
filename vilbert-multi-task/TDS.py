# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
from pickletools import optimize
import random
from io import open
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict
import sys
import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD

from vilbert.task_utils_vqacp2 import (
    LoadDatasetEval,
    LoadLosses,
    ForwardModelsTrain,
    ForwardModelsVal,
    # EvaluatingModel,
)

import vilbert.utils as utils
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--save_name", default="", type=str, help="save name for training."
    )
    parser.add_argument(
        "--use_chunk",
        default=0,
        type=float,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--batch_size", default=256, type=int, help="what is the batch size?"
    )
    parser.add_argument(
        "--tasks", default="20", type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--in_memory",
        default=False,
        type=bool,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument("--split", default="", type=str, help="which split to use.")
    parser.add_argument(
        "--dynamic_attention",
        action="store_true",
        help="whether use dynamic attention.",
    )
    parser.add_argument(
        "--clean_train_sets",
        default=True,
        type=bool,
        help="whether clean train sets for multitask data.",
    )
    parser.add_argument(
        "--visual_target",
        default=0,
        type=int,
        help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.",
    )
    parser.add_argument(
        "--task_specific_tokens",
        action="store_true",
        help="whether to use task specific tokens for the multi-task learning.",
    )

    args = parser.parse_args()
    with open("vilbert_tasks.yml", "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.baseline:
        from pytorch_transformers.modeling_bert import BertConfig
        from vilbert.basebert import BaseBertForVLTasks
    else:
        from vilbert.vilbert_for_vqacp2 import BertConfig
        from vilbert.vilbert_for_vqacp2 import VILBertForVLTasks

    task_names = []
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        name = task_cfg[task]["name"]
        task_names.append(name)

    if args.task_specific_tokens:
        config.task_specific_tokens = True

    # timeStamp = '-'.join(task_names) + '_' + args.config_file.split('/')[1].split('.')[0]
    timeStamp = args.from_pretrained.split("/")[-1] + "-" + args.save_name
    savePath = os.path.join(args.output_dir, timeStamp)
    config = BertConfig.from_json_file(args.config_file)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    task_batch_size, task_num_iters, task_ids, task_datasets_val, task_dataloader_val = LoadDatasetEval(
        args, task_cfg, args.tasks.split("-")
    )

    tbLogger = utils.tbLogger(
        timeStamp,
        savePath,
        task_names,
        task_ids,
        task_num_iters,
        1,
        save_logger=False,
        txt_name="eval.txt",
    )

    if args.visual_target == 0:
        config.v_target_size = 1601
        config.visual_target = args.visual_target
    else:
        config.v_target_size = 2048
        config.visual_target = args.visual_target

    if args.task_specific_tokens:
        config.task_specific_tokens = True

    num_labels = None
    if args.baseline:
        model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained,
            config=config,
            num_labels=num_labels,
            default_gpu=default_gpu,
        )
    else:
        model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained,
            config=config,
            num_labels=num_labels,
            default_gpu=default_gpu,
        )

    task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)

    elif n_gpu > 1:
        model = nn.DataParallel(model)

    print("***** Running evaluation *****")
    print("  Num Iters: ", task_num_iters)
    print("  Batch size: ", task_batch_size)

    # model.eval()
    # when run evaluate, we run each task sequentially.
    optimizer = SGD(model.parameters(), lr=task_cfg[task]["lr"], momentum=0.9, weight_decay=task_cfg[task]["weight_decay"])
    for task_id in task_ids:
        results = []
        others = []
    
        loss, score, batch_size, results, others = EvaluatingModel(
            args,
            task_cfg,
            device,
            task_id,
            model,
            task_dataloader_val,
            task_losses,
            results,
            others,
            optimizer
        )

        tbLogger.step_val(0, float(loss), float(score), task_id, batch_size, "val")

        sys.stdout.write("%d/%d\r" % (i, len(task_dataloader_val[task_id])))
        sys.stdout.flush()
        # save the result or evaluate the result.
        ave_score = tbLogger.showLossVal(task_id)

        if args.split:
            json_path = os.path.join(savePath, args.split)
        else:
            json_path = os.path.join(savePath, task_cfg[task_id]["val_split"])

        json.dump(results, open(json_path + "_result.json", "w"))
        # json.dump(others, open(json_path + "_others.json", "w"))

def EvaluatingModel(
    args,
    task_cfg,
    device,
    task_id,
    model,
    task_dataloader,
    task_losses,
    results,
    others,
    optimizer
):
    all_loss = 0
    # idx = 0
    for i, batch in enumerate(task_dataloader[task_id]):
        features, spatials, question, target, image_mask, input_mask, segment_ids, co_attention_mask, question_id = batch
        features = features.cuda()
        spatials = spatials.float().cuda()
        question = question.cuda()
        target = target.cuda()
        image_mask = image_mask.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        co_attention_mask = co_attention_mask.cuda()

        # features, spatials, question, target, image_mask, input_mask, segment_ids, co_attention_mask, question_id = batch

        batch_size = features.size()[0]

        task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

        Entropy = 7.7293 * task_cfg[task_id]['rate']

        # with torch.no_grad():
        model.train(False)
        vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
            task_tokens,
        )
        logits = torch.max(vil_prediction, 1)[1].data  # argmax
        loss = 0
        batch_score = 0
        for j in range(logits.size(0)):
            results.append(
                {
                    "question_id": question_id[j].item(),
                    "answer": task_dataloader[task_id].dataset.label2ans[
                        logits[j].item()
                    ],
                }
            )
        
        model.train(True)
        vil_prediction_, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
            task_tokens,
        )

        index_v = random.sample(range(0, batch_size), batch_size)
        gv_neg = features[index_v]
        spatials_neg = spatials[index_v]
        # gv_neg = torch.randn(v.size()).cuda()
        vil_prediction_neg_v, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
            question,
            gv_neg,
            spatials_neg,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
            task_tokens,
        )

        optimizer.zero_grad()
        loss = softmax_entropy(vil_prediction_)

        loss_mask = (loss <= Entropy).cuda()

        gt_predict = torch.argmax(vil_prediction_, dim=1)
        neg_v_predict = torch.argmax(vil_prediction_neg_v, dim=1)
        mask_v = (gt_predict == neg_v_predict)

        bias_sample = mask_v & loss_mask
        loss_min_logits, _ = torch.max(F.softmax(vil_prediction_neg_v, dim=1), dim=1)
        loss_min_logits_, _ = torch.max(F.softmax(vil_prediction_, dim=1), dim=1)
        loss_logits = (bias_sample.float() * loss_min_logits).sum() / sum(bias_sample.float())
        loss_logits_ = (bias_sample.float() * loss_min_logits_).sum() / sum(bias_sample.float())

        loss_mask = loss_mask & (~mask_v)

        loss = (loss * loss_mask.float()).sum() / sum(loss_mask.float()) 

        total_loss = loss + loss_logits + loss_logits_
        
        all_loss += total_loss.item() * batch_size

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        # idx += batch_size

        print("iter: {}, loss: {}".format(i, total_loss.item()))

    return float(all_loss/len(task_dataloader[task_id].dataset)), float(batch_score), batch_size, results, others

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

if __name__ == "__main__":

    main()
