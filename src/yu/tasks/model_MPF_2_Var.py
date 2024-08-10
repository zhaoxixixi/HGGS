# MPF_2_Var System
# https://github.com/EnzeXu/Fourier_PINN/blob/main/model_MPF_2_Var.ipynb

import numpy as np
from random import random
from yu.tasks.ode_model import Normal_Equation, Normal_Parameters
from typing import List, Any
from yu.tools.array import has_nan, is_exceeded
from yu.const.normal import PDEType
from yu.exception.normal import PDEException

from yu.tasks.pde_models import Truth

class Parameters(Normal_Parameters):
    """ Define parameters in ODEs with default values """
    k_1_ = 0.01 # \lambda_1
    k_2_ = 0.01 # \lambda_2
    k_2__ = 10
    k_25_ = 0.04    # \lambda_3
    k_25__ = 100
    k_Wee = 1.5 # \lambda_4
    k_INH = 0.1 # \lambda_5
    k_CAK = 1 # \lambda_6
    G = 1 + k_INH/k_CAK 
    ode_model_name = 'MPF_2_Var'

    @staticmethod
    def _random(k1, k2):
        return (k2 - k1) * random() + k1

    def random(self, rate: List):
        r_rate = []
        for item in rate:
            if item[0] != item[1]:
                r_rate.append(self._random(*item))
            else:
                r_rate.append(item[0])
        return self._multiple(r_rate)

    @classmethod
    def of_coefficients(cls, coefficients):
        return cls()._multiple(coefficients)

    def _multiple(self, coefficients):
        self.k_1_ *= coefficients[0]
        self.k_2_ *= coefficients[1]
        self.k_2__ *= coefficients[2]
        self.k_25_ *= coefficients[3]
        self.k_25__ *= coefficients[4]
        self.k_Wee *= coefficients[5]
        self.k_INH *= coefficients[6]
        self.k_CAK *= coefficients[7]
        self.G = 1 + self.k_INH / self.k_CAK 
        return self

    def assign(self, params):
        self.k_1_ = params[0]
        self.k_2_ = params[1]
        self.k_2__ = params[2]
        self.k_25_ = params[3]
        self.k_25__ = params[4]
        self.k_Wee = params[5]
        self.k_INH = params[6]
        self.k_CAK = params[7]

        if self.k_CAK == 0:
            self.G = 1
        else:
            self.G = 1 + self.k_INH / self.k_CAK 

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
            params.k_1_,
            params.k_2_,
            params.k_2__,
            params.k_25_,
            params.k_25__,
            params.k_Wee,
            params.k_INH,
            params.k_CAK,
        ]
        return ','.join([str(item) for item in rets])


class Equation(Normal_Equation):
    """ Define ODEs"""
    @classmethod
    def instantiate(cls, y, t, params: Parameters):
        """
        instantiate ODEs according to params
        y array is: dx/dt dy/dt
        """
        if has_nan(y):
            raise PDEException(PDEType.NAN)
        if is_exceeded(y):
            raise PDEException(PDEType.INF)

        # self.params.k_1_ / self.params.G - ( self.params.k_2_ + self.params.k_2__ * y[0] ** 2 + self.params.k_Wee ) * y[0] + ( self.params.k_25_ + self.params.k_25__ * y[0]**2 ) * ( y[1]/self.params.G - y[0] ),
        # self.params.k_1_ - ( self.params.k_2_ + self.params.k_2__ * y[0] ** 2 ) * y[1]

        return np.asarray([
            params.k_1_ / params.G - ( params.k_2_ + params.k_2__ * y[0] ** 2 + params.k_Wee ) * y[0] + ( params.k_25_ + params.k_25__ * y[0]**2 ) * ( y[1]/params.G - y[0] ),
            params.k_1_ - ( params.k_2_ + params.k_2__ * y[0] ** 2 ) * y[1]
        ])