"""
@author: longlong.yu
@email: yulonglong.hz@qq.com
@date: 2023-03-30
@description: Grid：用于多维空间等度切分
"""
import json
import os.path
from typing import List

from yu.const.grid import SearchType
from yu.tools.misc import to_json
from yu.tools.searcher import Target, Searcher, BFS, DFS

from yu.tools.searcher import File_Data


class Grid(Target):
    """ 切分格子, 并根据格子的得分决定是否大于阈值来决定是否继续切分 """
    # 是否自动保存
    auto_save: bool
    # 保存目录
    save_dir: str
    # 是否已初始化
    inited: bool = False
    # data file
    data_f = None
    # search type
    search_type: SearchType
    searcher: Searcher
    # 其他字段
    kwargs: dict
    # 非序列化的字段
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
        """ 重新加载上次记录 """
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

        # 加载配置，并重新保存（相对路径可能变化，需要重新赋一下值）
        self.load_config(self.config_file)
        self.save_dir = save_dir
        self.save_config()

        # 初始化切分格子
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
        """ 检查配置是否合法 """
        msgs = []
        if self.dimensions < 1:
            msgs.append(f'维度不能小于1({self.dimensions})')
        # for item in self.per_split_n:
        #     if item < 2:
        #         msgs.append(f'每次切分不能小于2({self.per_split_n})')
        if self.init_split_n > self.max_split_times:
            msgs.append(f'初始化切分深度({self.init_split_n})不能大于最大切分深度({self.max_split_times})')
        if self.init_split_n < 1:
            msgs.append(f'初始化切分深度不能小于1({self.init_split_n})')
        if self.max_split_times < 1:
            msgs.append(f'最大切分深度不能小于1({self.max_split_times})')
        if msgs:
            raise Exception(','.join(msgs))

    def init(self, **kwargs):
        """ 初始化 """
        if not self.inited:
            # 保存 config
            self.save_config()

            # 重置数据文件
            if not os.path.isdir(os.path.dirname(self.data_file)):
                os.makedirs(os.path.dirname(self.data_file))
            self.data_f = open(self.data_file, 'w')

            # 初始化切分格子
            self.positions = []
            if self.search_type == SearchType.BFS:
                self.searcher = BFS(**self.kwargs)
            elif self.search_type == SearchType.DFS:
                self.searcher = DFS(**self.kwargs)
            elif self.search_type == SearchType.File_Data:
                self.searcher = File_Data(**self.kwargs)
            self.searcher.init(self, **kwargs)

            self.inited = True
        # 检查配置
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
        """ 开始计算当前格子的概率 """
        self.init(**kwargs)

        type = self.search_type == SearchType.File_Data
        while self.positions:
            # 计算概率
            self.positions[-1].score = score_calculator(
                *self.position(type), **kwargs)
            # 保存结果
            if self.auto_save:
                self.save(type)

            # 寻找下一个格子
            self.next()

        self.close()
        return self

    def next(self):
        """ （深度优先）寻找下一个格子 """
        self.searcher.next(self)
        return self

    def close(self):
        # 关闭 data file
        if self.inited and self.data_f:
            self.data_f.close()
            self.data_f = None
        if self.searcher:
            self.searcher.close()
