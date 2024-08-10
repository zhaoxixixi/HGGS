"""
@author: longlong.yu
@email: yulonglong.hz@qq.com
@date: 2023-03-29
@description:
"""
import numpy as np
import scipy.fft as sp_fft
from statsmodels.tsa.stattools import acf


def to_fft(
        y, f, n,
        normalized: bool = False,
        only_positive: bool = True
):
    """
     y  Fourier

    f: sample frequency
    n: sample size
    normalized: if True, y will be divided by period t (n / f)
    only_positive: if True, function will only return half of fft(y) where frequency is not negative
    """
    a_axis = np.abs(sp_fft.fft(y))
    if normalized:
        a_axis = a_axis * f / n

    a_axis = sp_fft.fftshift(a_axis)
    f_axis = sp_fft.fftfreq(a_axis.size, d=1 / f)
    f_axis = sp_fft.fftshift(f_axis)
    if only_positive:
        half = a_axis.size // 2
        return a_axis[half:], f_axis[half:]
    else:
        return a_axis, f_axis


def search_by_acf(data, period: int, ode_model: str='HSECC'):
    """
    find point around period whose acf is max
    :return: point, score
    """

    current, last_score = period, np.abs(acf(data, nlags=period)[-1])
    step = max(period // 10, 1)
    while step > 0:
        score = np.abs(acf(data, nlags=current + step)[-1])
        if score > last_score:
            current, last_score = current + step, score
            continue

        if current - step > period // 2:
            score = np.abs(acf(data, nlags=current - step)[-1])
            if score > last_score:
                current, last_score = current - step, score
                continue

        step //= 2
    return current, last_score
