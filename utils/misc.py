# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

import errno
import os
import os.path as op
import yaml
import random
import torch
import numpy as np
import torch.distributed as dist

from PIL import Image

def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname)
            )


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def compute_metrics_from_logits(logits, targets):
    """
        recall@k for N candidates

            logits: (batch_size, num_candidates)
            targets: (batch_size, )
    """
    batch_size, num_candidates = logits.shape

    sorted_indices = logits.sort(descending=True)[1]
    targets = targets.tolist()

    recall_k = dict()
    if num_candidates <= 10:
        ks = [1, max(1, round(num_candidates*0.2)), max(1, round(num_candidates*0.5))]
    elif num_candidates <= 100:
        ks = [1, max(1, round(num_candidates*0.1)), max(1, round(num_candidates*0.5))]
    else:
        raise ValueError("num_candidates: {0} is not proper".format(num_candidates))
    for k in ks:
        # sorted_indices[:,:k]: (batch_size, k)
        num_ok = 0
        for tgt, topk in zip(targets, sorted_indices[:,:k].tolist()):
            if tgt in topk:
                num_ok += 1
        recall_k[f'recall@{k}'] = (num_ok/batch_size)

    # MRR
    MRR = 0
    for tgt, topk in zip(targets, sorted_indices.tolist()):
        rank = topk.index(tgt)+1
        MRR += 1/rank
    MRR = MRR/batch_size
    return recall_k, MRR

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
