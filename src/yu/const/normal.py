from enum import StrEnum


class PDEType(StrEnum):
    NAN = 'NAN'
    INF = 'INF'
    OSCILLATION = 'OSCI'
    CONVERGENCE = 'CONV'


class RandomType(StrEnum):
    """ 随机类型 """
    NO = 'NO'
    RANDOM = 'RANDOM'
    LHS = 'LHS'


class DisplayType(StrEnum):
    """ 展示类型 """
    NONE = 'NONE'
    SAVE = 'SAVE'
    SHOW = 'SHOW'


class FigureType(StrEnum):
    """ 图表类型 """
    COLOR_MESH = 'COLOR_MESH'
    SCATTER = 'SCATTER'
    SCATTER3D = 'SCATTER3D'
    TSNE = 'TSNE'

class ConstData():
    eps = 1e-6
