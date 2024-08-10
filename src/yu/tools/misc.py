"""
@author: longlong.yu
@email: yulonglong.hz@qq.com
@date: 2023-03-30
@description: misc function tools
"""
import json
import os
from datetime import datetime
from typing import List

from scipy.stats import qmc

from yu.tools.time_util import TimeUtil
import numpy as np

def get_inputs_targets_outputs(model, data_loader, DCT: bool=False, thresh: float=None):
    '''
    return inputs, targets, outputs
    '''
    model.eval()

    lines = []
    inputs = []
    outputs = []
    targets = []
    for (input, target, _indices) in data_loader:
        output = model(input)
        outputs.extend(output.detach().cpu().tolist())
        inputs.extend(input.detach().cpu().tolist())
        targets.extend(target.detach().cpu().tolist())

    if thresh:
        outputs = np.array(outputs)
        outputs[outputs < thresh] = 0.0
        outputs = outputs.tolist()
    return inputs, targets, outputs

def get_OSCI_ratio(data_path: str, param: int):
    NO = 0
    O = 0
    eps = 1e-6
    with open(data_path, 'r') as f:
        for line in f.readlines():
            fields = [float(_) for _ in line[:-1].split(',')]
            if fields[param] < eps:
                NO += 1
            else:
                O += 1
    return O / (NO + O), NO + O


def to_json(o, exclude_fields: List[str] = None, **kwargs) -> str:
    """ trans to json """
    tmp = {}
    for key, value in o.__dict__.items():
        if not exclude_fields or key not in exclude_fields:
            if isinstance(value, datetime):
                value = TimeUtil.strftime(value, '%Y-%m-%d %H:%M:%S')
            tmp[key] = value
    return json.dumps(tmp, **kwargs)

def makedir(dir_path: str):
    """ makedirs """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def sample_lhs(lb, ub, n, d):
    """
    :param lb: lower bound
    :param ub: upper bound
    :param n: sample size
    :param d: dimension
    :return: samples
    """
    sampler = qmc.LatinHypercube(d=d)
    samples = sampler.random(n=n)
    return qmc.scale(samples, [lb], [ub])
