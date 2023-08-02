# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from email.mime import image
from io import open
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
# from vilbert.datasets import DatasetMapTrain, DatasetMapEval
# from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
import pdb

logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
}


def ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses):
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

    batch_score = compute_score_with_logits(vil_prediction, target).sum()
    loss = task_losses[task_id](vil_prediction, target).mean() * target.size(1)

    return float(loss), float(batch_score), batch_size


def ForwardModelsTrain(
    args,
    task_cfg,
    device,
    task_id,
    task_count,
    task_iter_train,
    task_dataloader_train,
    model,
    task_losses,
):
    # given the current task, decided whether to forward the model and forward with specific loss.

    # reset the task iteration when needed.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    task_count[task_id] += 1
    # get the batch
    batch = task_iter_train[task_id].next()
    features, spatials, question, target, image_mask, input_mask, segment_ids, co_attention_mask, question_id = batch
    features = features.cuda()
    spatials = spatials.float().cuda()
    question = question.cuda()
    target = target.cuda()
    image_mask = image_mask.cuda()
    input_mask = input_mask.cuda()
    segment_ids = segment_ids.cuda()
    co_attention_mask = co_attention_mask.cuda()

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
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

    loss = task_losses[task_id](vil_prediction, target)
    loss = loss.mean() * target.size(1)
    batch_score = compute_score_with_logits(vil_prediction, target).sum() / vil_prediction.size()[0]

    return loss, batch_score


def LoadLosses(args, task_cfg, task_ids):

    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = "TASK" + task_id
        model_type = task_cfg[task]["type"]
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg[task]["loss"]]

    return losses


def LoadDatasets(args, task_cfg, ids, split="traintest"):

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )
    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}
    from vilbert.datasets.vqacp2_dataset import VQAClassificationDataset

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_name = task_cfg[task]["name"]
        task_ids.append(task)
        batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps
        num_workers = args.num_workers
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())
            num_workers = int(num_workers / dist.get_world_size())

        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        task_datasets_train[task] = None
        if "train" in split:
            task_datasets_train[task] = VQAClassificationDataset(
                'train',
                dataroot=task_cfg[task]["dataroot"],
                tokenizer=tokenizer,
                ratio=1.0,
                padding_idx=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_num_obj=task_cfg[task]["max_region_num"]
            )

        task_datasets_val[task] = None
        if "test" in split:
            task_datasets_val[task] = VQAClassificationDataset(
                'test',
                dataroot=task_cfg[task]["dataroot"],
                tokenizer=tokenizer,
                ratio=1.0,
                padding_idx=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_num_obj=task_cfg[task]["max_region_num"]
            )

        task_num_iters[task] = 0
        task_batch_size[task] = 0
        from vilbert.datasets import utils
        if "train" in split:
            if args.local_rank == -1:
                train_sampler = RandomSampler(task_datasets_train[task])
            else:
                # TODO: check if this works with current data generator from disk that relies on next(file)
                # (it doesn't return item back by index)
                train_sampler = DistributedSampler(task_datasets_train[task])

            task_dataloader_train[task] = DataLoader(
                task_datasets_train[task],
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=utils.trim_collate
            )

            task_num_iters[task] = len(task_dataloader_train[task])
            task_batch_size[task] = batch_size

        if "test" in split:
            task_dataloader_val[task] = DataLoader(
                task_datasets_val[task],
                shuffle=False,
                batch_size=batch_size,
                num_workers=2,
                pin_memory=True,
                collate_fn=utils.trim_collate
            )

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )

def LoadDatasetEval(args, task_cfg, ids):

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    task_datasets_val = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}
    from vilbert.datasets.vqacp2_dataset import VQAClassificationDataset
    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_ids.append(task)
        task_name = task_cfg[task]["name"]
        batch_size = args.batch_size
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())

        # num_workers = int(args.num_workers / len(ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        if args.split:
            eval_split = args.split
        else:
            eval_split = task_cfg[task]["val_split"]

        task_datasets_val[task] = VQAClassificationDataset(
                'test',
                dataroot=task_cfg[task]["dataroot"],
                img_root=task_cfg[task]['img_root'],
                tokenizer=tokenizer,
                ratio=1.0,
                padding_idx=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_num_obj=task_cfg[task]["max_region_num"]
            )
            
        from vilbert.datasets import utils
        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=utils.trim_collate
        )

        task_num_iters[task] = len(task_dataloader_val[task])
        task_batch_size[task] = batch_size

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_val,
        task_dataloader_val,
    )

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores

def EvaluatingModel(
    args,
    task_cfg,
    device,
    task_id,
    batch,
    model,
    task_dataloader,
    task_losses,
    results,
    others,
):
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

    with torch.no_grad():
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
    for i in range(logits.size(0)):
        results.append(
            {
                "question_id": question_id[i].item(),
                "answer": task_dataloader[task_id].dataset.label2ans[
                    logits[i].item()
                ],
            }
        )

    return float(loss), float(batch_score), batch_size, results, others