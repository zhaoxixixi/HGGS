"""
@author: longlong.yu
@email: yu.hz@qq.com
@date: 2023-04-01
@description:
"""
from yu.const.normal import PDEType
from yu.exception.base import YuBaseException


class PDEException(YuBaseException):
    pde_type: PDEType

    def __init__(self, pde_type: PDEType, *args):
        super().__init__(*args)
        self.pde_type = pde_type

    def __str__(self):
        return f'[{self.pde_type}]{super().__str__()}'
