# http://dx.doi.org/10.1063/1.3677190

import numpy as np
from random import random
from yu.tasks.ode_model import Normal_Equation, Normal_Parameters
from typing import List, Any
from yu.tools.array import has_nan, is_exceeded
from yu.const.normal import PDEType
from yu.exception.normal import PDEException

from yu.tasks.pde_models import Truth
# Cell Cycle System
class Parameters(Normal_Parameters):
    """ Define parameters in ODEs with default values """
    ode_model_name = 'HSECC'
    mu = 0.006
    k_dx = 0.04  # 2
    k_dy = 0.02  # 4 6
    k_hyz = 7.5
    k_pyx = 1.88
    k_dz = 0.1   # 6
    k_dmx = 3.5
    k_smy = 7.0
    k_dmy = 3.5
    k_smz = 0.001
    k_smzx = 10.0
    k_dmz = 0.15

    k_sx = 1.53  # 1
    k_sy = 1.35  # 3 7
    k_hy = 29.7  #   5
    k_sz = 1.35  # 5
    k_smx = 1.04
    k_dxy = 0.00741  # 7
    J_hy = 5.4
    J_pyx = 5.4
    J_smzx = 756
    '''
    self.k_sx = params[0]  1.53
    self.k_dx = params[1]  0.00741
    self.k_sy = params[2]  1.35
    self.k_dy = params[3]  0.02

    self.k_sz = params[4]  1.35
    self.k_dz = params[5]  0.1
    self.k_dxy = params[6]  0.00741
    '''

    @staticmethod
    def _random(k1, k2):
        return (k2 - k1) * random() + k1

    def random(self, rate: List):
        """ random in rate """
        r_rate = []
        for item in rate:
            if item[0] != item[1]:
                r_rate.append(self._random(*item))
            else:
                r_rate.append(item[0])
        return self._multiple(r_rate)

    @classmethod
    def of_coefficients(cls, coefficients):
        """ init """
        return cls()._multiple(coefficients)

    def _multiple(self, coefficients):
        self.k_sx *= coefficients[0]
        self.k_dx *= coefficients[1]
        self.k_sy *= coefficients[2]
        self.k_dy *= coefficients[3]

        self.k_sz *= coefficients[4]
        self.k_dz *= coefficients[5]
        self.k_dxy *= coefficients[6]
        return self

    def assign(self, params):
        self.k_sx = params[0]
        self.k_dx = params[1]
        self.k_sy = params[2]
        self.k_dy = params[3]

        self.k_sz = params[4]
        self.k_dz = params[5]
        self.k_dxy = params[6]

    def build_csv(self, truth: Truth, params: Normal_Parameters, name: str):
        rets = [name]
        i = 0
        for item in truth.periods:
            y_ratio = '>10' if item[4] > 10 else f'{item[4]:.2f}'
            rets += [
                truth.curve_pde_types[i] if truth.success else f'{truth.pde_type}?',
                # f'{item[1]:.4f}',
                f'{item[1]}',
                y_ratio,
                # f'{item[0]:.4f}',
                f'{item[0]}',
            ]
            i += 1

        rets += [
            params.k_sx,
            params.k_dx,
            params.k_sy,
            params.k_dy,
            params.k_sz,
            params.k_dz,
            params.k_dxy,
        ]
        return ','.join([str(item) for item in rets])


class Equation(Normal_Equation):
    """ Define ODEs"""
    @classmethod
    def instantiate(cls, y, t, params: Parameters):
        """
        instantiate ODEs according to params
        y array is：V X YT Y Z
        """
        if has_nan(y):
            raise PDEException(PDEType.NAN)
        if is_exceeded(y):
            raise PDEException(PDEType.INF)

        return np.asarray([
            # params.mu * y[0],
            0,
            params.k_sx * (params.k_smx * y[0] / params.k_dmx) * y[0] - params.k_dx * y[1] - params.k_dxy * y[1] * y[3] / y[0],
            params.k_sy * (params.k_smy / params.k_dmy) * y[0] - params.k_dy * y[2],
            params.k_sy * (params.k_smy / params.k_dmy) * y[0] - params.k_dy * y[3]
                + (params.k_hy * y[0] + params.k_hyz * y[4]) * (y[2] - y[3]) / (params.J_hy * y[0] + y[2] - y[3])
                - params.k_pyx * y[1] * y[3] / (params.J_pyx * y[0] + y[3]),
            params.k_sz * (params.k_smz + params.k_smzx * (y[1] ** 2) / ((params.J_smzx * y[0]) ** 2 + y[1] ** 2))
                 / params.k_dmz * y[0] - params.k_dz * y[4]
        ])

