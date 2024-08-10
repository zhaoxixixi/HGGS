# Problem set 2, Number 1
# Activator inhibitor system

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
    c = 2.8
    y0 = 0.1
    eps = 0.1
    a = 1
    b = 5
    r = 1
    ode_model_name = 'PS2_01'

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
        self.c *= coefficients[0]
        self.y0 *= coefficients[1]
        self.eps *= coefficients[2]
        self.a *= coefficients[3]
        self.b *= coefficients[4]
        self.r *= coefficients[5]
        return self

    def assign(self, params):
        self.c = params[0]
        self.y0 = params[1]
        self.eps = params[2]
        self.a = params[3]
        self.b = params[4]
        self.r = params[5]

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
            params.c,
            params.y0,
            params.eps,
            params.a,
            params.b,
            params.r,
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

        # dx/dt=(a+b*x^4)/(1+x^4+r*y) - x
        # dy/dt=eps*(c*x +y0-y)

        # dx/dt=(a+b*x^2)/(1+x^2+r*y) - x
        # dy/dt=eps*(c*x +y0-y)
        return np.asarray([
            (params.a + params.b * y[0] ** 2) / (1 + y[0] ** 2 + params.r * y[1]) - y[0],
            params.eps * (params.c * y[0] + params.y0 - y[1])
        ])