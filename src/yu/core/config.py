import os.path
import sys
from datetime import datetime
from typing import List, TypeVar, Generic

from pydantic import BaseSettings, validator, Extra
sys.path.extend(['../../../src'])  # noqa
from yu.tools.misc import to_json, makedir
from yu.tools.time_util import TimeUtil


class BaseConfig(BaseSettings):
    now: datetime = TimeUtil. now()
    name: str = ''
    save_dir: str = None
    config_path: str = 'config.json'

    exclude_fields: List[str] = [
        'exclude_fields'
    ]

    class Config:
        case_sensitive = True
        env_file_encoding = 'utf-8'
        extra = Extra.allow

    def __str__(self):
        ret = to_json(self, exclude_fields=self.exclude_fields, indent=2)
        return ''.join('\t' + line for line in ret.splitlines(True))

    def save(self):
        if self.save_dir:
            makedir(self.save_dir)
            with open(os.path.join(self.save_dir, self.config_path), 'w') as f:
                f.write(f'{self}')
    def save_by_self_dir(self, dir):
        with open(os.path.join(dir + '/config.json'), 'w') as f:
            f.write(f'{self}')
    @validator('save_dir', pre=True)
    def set_save_dir(cls, v, values):
        name = values.get('name')
        now = values.get('now')
        if name:
            return os.path.join(v, name, TimeUtil.strftime(now, '%Y%m%d_%H%M%S'))
        else:
            return os.path.join(v, TimeUtil.strftime(now, '%Y%m%d_%H%M%S'))


T = TypeVar("T", bound=BaseConfig)

def auto_config(
        config_cls: Generic[T],
        script_path: str = None,
        file_path: str = None,
        pre_save=True,
        post_save=False
):
    """ auto load config """
    def out_wrapper(func):
        if len(sys.argv) >= 2:
            config_path = sys.argv[1]
        elif not file_path:
            basename = os.path.basename(script_path).replace('.py', '.env')
            dirname = os.path.basename(os.path.dirname(script_path))
            if dirname:
                config_path = f'../../../../resource/{dirname}/{basename}'
            else:
                config_path = f'../../../../resource/{basename}'
        else:
            config_path = file_path
        if '--local-rank=' in config_path:
            config_path = '../../../../resource/mlp/common_compare.env'
        print(config_path)
        config = config_cls(config_path)

        def wrapper(*args, **kwargs):
            if args:
                args = [config] + list(args[1:])  # noqa
            elif 'config' in kwargs:
                kwargs['config'] = config
            else:
                args = (config,)
            if pre_save:
                config.save()
            ret = func(*args, **kwargs)
            if post_save:
                config.save()
            return ret

        return wrapper
    return out_wrapper
