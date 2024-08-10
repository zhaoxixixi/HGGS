"""
@author: longlong.yu
@email: yulonglong.hz@qq.com
@date: 2023-04-05
@description: deal with date & time
"""

from datetime import datetime, timedelta

import pytz
from dateutil.relativedelta import relativedelta


class TimeUtil(object):

    LOCAL_TZ = pytz.timezone('Asia/Shanghai')
    UTC_TZ = pytz.utc

    @classmethod
    def now(cls) -> datetime:
        """ 返回本地时区当前时间 """
        return cls.UTC_TZ.localize(datetime.utcnow()).astimezone(cls.LOCAL_TZ)

    @classmethod
    def to_locale(cls, dt: datetime):
        """ 变换为本地时区 """
        if not dt:
            return dt
        return cls._validate(dt).astimezone(cls.LOCAL_TZ)

    @classmethod
    def to_utc(cls, dt: datetime):
        """ 变换为UTC时区 """
        if not dt:
            return dt
        return cls._validate(dt).astimezone(cls.UTC_TZ)

    @classmethod
    def strptime(cls, date_string, format):
        """ datetime.strptime, 按本地时区返回 """
        return cls.LOCAL_TZ.localize(datetime.strptime(date_string, format))

    @classmethod
    def strftime(cls, dt: datetime, fmt: str):
        """ datetime.strftime, 按本地时区返回 """
        return cls._validate(dt).astimezone(cls.LOCAL_TZ).strftime(fmt)

    @classmethod
    def localize(cls, dt: datetime):
        """ （将未指定时区的 datetime）指定为本地时区 """
        if not dt:
            return dt
        return cls.LOCAL_TZ.localize(dt)

    @classmethod
    def day_start(cls, dt: datetime) -> datetime:
        """ 返回本地时区当前时间 """
        dt = cls.to_locale(dt)
        return cls.strptime(dt.strftime('%Y-%m-%d'), '%Y-%m-%d')

    @classmethod
    def month_start(cls, dt: datetime) -> datetime:
        """ 返回本地时区当前时间 """
        dt = cls.to_locale(dt)
        return cls.strptime(dt.strftime('%Y-%m-01'), '%Y-%m-%d')

    @classmethod
    def next_month_start(cls, dt: datetime) -> datetime:
        """ 返回本地时区当前时间 """
        return cls.month_start(cls.month_start(dt) + timedelta(days=31))

    @classmethod
    def year_start(cls, dt: datetime) -> datetime:
        """ 返回本地时区当前时间 """
        dt = cls.to_locale(dt)
        return cls.strptime(dt.strftime('%Y-01-01'), '%Y-%m-%d')

    @classmethod
    def is_same_month(cls, dt_1: datetime, dt_2: datetime):
        if not dt_1 or not dt_1:
            return False
        dt_1 = cls.to_locale(dt_1)
        dt_2 = cls.to_locale(dt_2)
        return dt_1.year == dt_2.year and dt_1.month == dt_2.month

    @staticmethod
    def _validate(dt: datetime):
        """ 校验是否存在时区信息 """
        if not dt.tzinfo:
            raise Exception('datetime 类型必须指定时区')
        return dt

    @classmethod
    def get_date_month(cls, dt: datetime, mon=0):
        """
        获取几个月前/后 的时间
        """
        if not dt.tzinfo:
            raise Exception('datetime 类型必须指定时区')

        if mon < 0:
            return dt - relativedelta(months=-mon)
        else:
            return dt + relativedelta(months=mon)
