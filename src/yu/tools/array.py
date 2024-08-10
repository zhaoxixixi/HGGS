"""
@author: longlong.yu
@email: yulonglong.hz@qq.com
@date: 2023-04-01 14:40
@description: tools for array
"""
import numpy as np


def has_nan(x) -> bool:
    """ check if x contains nan """
    return np.isnan(x).sum() > 0


def is_exceeded(x, abs_max=10 ** 9):
    """ check if the absolute values of x exceed the max value """
    return abs(x).any() > abs_max
