from enum import StrEnum


class PDEType(StrEnum):
    NAN = 'NAN'
    INF = 'INF'
    OSCILLATION = 'OSCI'
    CONVERGENCE = 'CONV'


class RandomType(StrEnum):
    """ random """
    NO = 'NO'
    RANDOM = 'RANDOM'
    LHS = 'LHS'


class DisplayType(StrEnum):
    """ display """
    NONE = 'NONE'
    SAVE = 'SAVE'
    SHOW = 'SHOW'


class FigureType(StrEnum):
    """ figure """
    COLOR_MESH = 'COLOR_MESH'
    SCATTER = 'SCATTER'
    SCATTER3D = 'SCATTER3D'
    TSNE = 'TSNE'

class ConstData():
    eps = 1e-6
