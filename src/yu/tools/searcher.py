import abc
import json
from typing import List

import numpy as np

from yu.tools.misc import to_json
from yu.tools.queue import SqliteQueue


class Position:
    """ keep grid position information. (0, 0) is in the place of top-left """
    coords: List[int]
    score: float = 0

    def __init__(self, coords: List[int]):
        self.coords = coords

    def __str__(self):
        return to_json(self)

    @classmethod
    def from_str(cls, s: str):
        """ load from str """
        tmp = json.loads(s)
        ret = cls(coords=tmp.get('coords', []))
        if 'score' in tmp:
            ret.score = tmp['score']
        return ret


class Target(metaclass=abc.ABCMeta):
    """  """
    positions: List[Position]
    max_split_times: int
    per_split_n: List[int]
    init_split_n: int
    dimensions: int
    threshold: List[float]

    def dump_positions(self, type = False):
        """  """
        if type:
            if self.positions:
                return str(self.positions[-1])
            return []
        return '|'.join([str(item) for item in self.positions or []])

    def load_positions(self, s: str):
        """  """
        self.positions = []
        if s:
            for tmp in s.split('|'):
                self.positions.append(Position.from_str(tmp))

    def position(self, type = False):
        if type:
            # type=1 => File Data
            return np.array(self.positions[-1].coords), self.per_split_n[0], 1
        """ Calculate the coordinate values and return the coordinate vector along with the total edge length. """
        depth = len(self.positions)
        coords = np.zeros(self.dimensions)
        edge = 1

        # The length of depth indicates the number of times the original Grid is subdivided.
        for i in range(depth - 1, -1, -1):
            for j in range(self.dimensions):
                coords[j] += self.positions[i].coords[j] * edge
            edge *= self.per_split_n[i]

        return coords, edge, depth

    def can_split(self):
        """ split """
        if not self.positions or len(self.positions) >= self.max_split_times:
            return False

        if len(self.positions) < self.init_split_n:
            return True

        if self.threshold is None:
            return True
        elif self.threshold[0] is not None and self.positions[-1].score <= self.threshold[0]:
            return False
        elif self.threshold[1] is not None and self.positions[-1].score >= self.threshold[1]:
            return False

        return True


class Searcher(metaclass=abc.ABCMeta):
    """  """
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def init(self,  target: Target, **kwargs):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def next(self, target: Target):
        pass


class BFS(Searcher):

    queue: SqliteQueue

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.queue = SqliteQueue(db_path=kwargs['db_path'], table=kwargs['table'])

    def init(self, target: Target, **kwargs):
        self.queue.init()
        target.positions = []
        if self.queue.empty():
            self._put_split(target)
        self.next(target)

    def close(self):
        self.queue.close()

    def next(self, target: Target):
        if target.positions and target.can_split():
            self._put_split(target)

        target.load_positions(self.queue.get())

        while target.positions and len(target.positions) < target.init_split_n:
            # 
            if target.can_split():
                self._put_split(target)

            target.load_positions(self.queue.get())

        return self

    def _put_split(self, target: Target):
        """  """
        per_split_n = target.per_split_n[len(target.positions)]
        total = per_split_n ** target.dimensions
        i = 0
        while i < total:
            coords = []
            tmp = i
            for j in range(target.dimensions):
                coords.append(tmp % per_split_n)
                tmp //= per_split_n
            i += 1

            child = Position(
                coords=coords[::-1],
            )
            target.positions.append(child)
            self.queue.put([target.dump_positions()])
            target.positions.pop()


class DFS(Searcher):

    def init(self, target: Target, **kwargs):
        init_positions = kwargs.get('init_positions')
        if init_positions:
            target.load_positions(init_positions)
            self.next(target)
        else:
            target.positions = []
            target.positions.append(Position(
                coords=[0] * target.dimensions
            ))
        self._validate_init_split(target)

    def close(self):
        pass

    def next(self, target: Target):
        if not target.positions:
            return self

        current = target.positions[-1]
        # can split
        if target.can_split():
            child = Position(
                coords=[0] * target.dimensions,
            )
            target.positions.append(child)
        # next
        else:
            while current:
                i = target.dimensions - 1
                while i >= 0 and current.coords[i] == target.per_split_n[len(target.positions) - 1] - 1:
                    current.coords[i] = 0
                    i -= 1
                    continue
                # not end
                if i >= 0:
                    current.coords[i] += 1
                    break
                else:
                    target.positions.pop()
                    current = target.positions[-1] if target.positions else None

        # validate
        self._validate_init_split(target)

    @staticmethod
    def _validate_init_split(target: Target):
        """ check """
        while target.positions and len(target.positions) < target.init_split_n:
            target.positions.append(Position(
                coords=[0] * target.dimensions,
            ))

def get_last_grid(path):
    with open(path, 'rb') as f:
        return eval(f.readlines()[-1])["coords"]
        delta = len(f.readline())
        f.seek(-delta * 2, 2)
        line = f.readlines()[-1].decode(encoding='UTF-8', errors='strict')
        return eval(line)['coords']
        # print(eval(line)['coords'])
        # print(eval(f.readlines()[-1])['coords'])

class File_Data(Searcher):

    def init(self, target: Target, **kwargs):
        data_path = kwargs['data_path']
        reload_path = kwargs['reload_path']
        init_positions = kwargs.get('init_positions')
        if init_positions:
            target.load_positions(init_positions)
            self.next(target)
        else:
            target.positions = []
            if reload_path is not None:
                data_path = reload_path

            with open(data_path, 'r') as f:
                for line in f.readlines():
                    if line[-1] == '\n':
                        line = line[:-1]
                    line = eval(line)
                    target.positions.append(Position(
                        coords=line
                    ))
        self._validate_init_split(target)

    def close(self):
        pass

    def next(self, target: Target):
        if not target.positions:
            return self

        target.positions.pop()
        # check
        self._validate_init_split(target)

    @staticmethod
    def _validate_init_split(target: Target):
        """ check """
        while target.positions and len(target.positions) < target.init_split_n:
            target.positions.append(Position(
                coords=[0] * target.dimensions,
            ))
