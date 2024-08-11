import json
import os.path
from typing import List

from yu.const.grid import SearchType
from yu.tools.misc import to_json
from yu.tools.searcher import Target, Searcher, BFS, DFS

from yu.tools.searcher import File_Data


class Grid(Target):
    """ split grid """
    # auto save?
    auto_save: bool
    # save dir
    save_dir: str
    # is init?
    inited: bool = False
    # data file
    data_f = None
    # search type
    search_type: SearchType
    searcher: Searcher
    kwargs: dict
    exclude_fields = ['exclude_fields', 'inited', 'positions', 'data_f', 'searcher']

    def __init__(
            self,
            max_split_times: int,
            per_split_n: List[int],
            init_split_n: int,
            dimensions: int,
            threshold: List[float],
            save_dir: str,
            auto_save: bool = True,
            search_type: SearchType = SearchType.BFS,
            **kwargs,
    ):
        super().__init__()
        self.max_split_times = max_split_times
        self.per_split_n = per_split_n
        self.init_split_n = init_split_n
        self.dimensions = dimensions
        self.threshold = threshold
        self.save_dir = save_dir
        self.auto_save = auto_save
        self.search_type = search_type
        self.kwargs = kwargs

    @property
    def config_file(self):
        """ config file path """
        return os.path.join(self.save_dir, 'grid_config.json') if self.save_dir else None

    @property
    def data_file(self):
        """ data file path """
        return os.path.join(self.save_dir, 'data') if self.save_dir else None

    @classmethod
    def reload(cls, save_dir: str, grid=None):
        """ reload from """
        if not grid:
            self = cls(
                max_split_times=0,
                per_split_n=[],
                init_split_n=0,
                dimensions=0,
                threshold=None, # noqa
                save_dir=save_dir
            )
        else:
            self = grid

        self.load_config(self.config_file)
        self.save_dir = save_dir
        self.save_config()

        self.data_f = open(self.data_file, 'a+')
        if self.search_type == SearchType.BFS:
            self.searcher = BFS(**self.kwargs)
            self.searcher.init(self)
        elif self.search_type == SearchType.DFS:
            self.searcher = DFS(**self.kwargs)
            f_size = os.path.getsize(self.data_file)

            if f_size > 1024:
                self.data_f.seek(f_size - 1024, 0)
            line = last_line = self.data_f.readline()
            while line:
                last_line = line
                line = self.data_f.readline()

            self.searcher.init(target=self, init_positions=last_line)

        self.inited = True
        return self

    def _validate_config(self):
        """ val? """
        msgs = []
        if self.dimensions < 1:
            msgs.append(f'dimension < 1({self.dimensions})')
        # for item in self.per_split_n:
        #     if item < 2:
        #         msgs.append(f'item < 2({self.per_split_n})')
        if self.init_split_n > self.max_split_times:
            msgs.append(f'init deep({self.init_split_n}) < max_deep({self.max_split_times})')
        if self.init_split_n < 1:
            msgs.append(f'init_split_n < 1({self.init_split_n})')
        if self.max_split_times < 1:
            msgs.append(f'max_split_times < 1({self.max_split_times})')
        if msgs:
            raise Exception(','.join(msgs))

    def init(self, **kwargs):
        """  """
        if not self.inited:
            # save config
            self.save_config()

            # init
            if not os.path.isdir(os.path.dirname(self.data_file)):
                os.makedirs(os.path.dirname(self.data_file))
            self.data_f = open(self.data_file, 'w')

            # init grid
            self.positions = []
            if self.search_type == SearchType.BFS:
                self.searcher = BFS(**self.kwargs)
            elif self.search_type == SearchType.DFS:
                self.searcher = DFS(**self.kwargs)
            elif self.search_type == SearchType.File_Data:
                self.searcher = File_Data(**self.kwargs)
            self.searcher.init(self, **kwargs)

            self.inited = True
        # check
        self._validate_config()
        return self

    def save_config(self):
        """ save config file """
        if self.config_file:
            if not os.path.isdir(os.path.dirname(self.config_file)):
                os.makedirs(os.path.dirname(self.config_file))
            with open(self.config_file, 'w') as f:
                f.write(to_json(self, self.exclude_fields) + '\n')

    def load_config(self, config_file: str):
        """ load config file """
        with open(config_file, 'r') as f:
            config = json.load(f)
        for key in self.__dict__.keys():
            if key not in self.exclude_fields and key in config:
                setattr(self, key, config[key])

    def save(self, type = False):
        """
        file format:
        ([coord0, coord1, ...], probability)
        :return:
        """
        if self.inited and self.data_f:
            self.data_f.write(f'{self.dump_positions(type)}\n')

    def execute(self, score_calculator, **kwargs):
        """ cal """
        self.init(**kwargs)

        type = self.search_type == SearchType.File_Data
        while self.positions:
            self.positions[-1].score = score_calculator(
                *self.position(type), **kwargs)
            # save
            if self.auto_save:
                self.save(type)

            # next grid
            self.next()

        self.close()
        return self

    def next(self):
        """  """
        self.searcher.next(self)
        return self

    def close(self):
        if self.inited and self.data_f:
            self.data_f.close()
            self.data_f = None
        if self.searcher:
            self.searcher.close()
