"""
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
        """ return current """
        return cls.UTC_TZ.localize(datetime.utcnow()).astimezone(cls.LOCAL_TZ)

    @classmethod
    def to_locale(cls, dt: datetime):
        """ transform to local """
        if not dt:
            return dt
        return cls._validate(dt).astimezone(cls.LOCAL_TZ)

    @classmethod
    def to_utc(cls, dt: datetime):
        """ to UTC """
        if not dt:
            return dt
        return cls._validate(dt).astimezone(cls.UTC_TZ)

    @classmethod
    def strptime(cls, date_string, format):
        """ datetime.strptime, return current """
        return cls.LOCAL_TZ.localize(datetime.strptime(date_string, format))

    @classmethod
    def strftime(cls, dt: datetime, fmt: str):
        """ datetime.strftime, return current """
        return cls._validate(dt).astimezone(cls.LOCAL_TZ).strftime(fmt)

    @classmethod
    def localize(cls, dt: datetime):
        """ Assign the local time zone to a datetime object when the time zone is not specified. """
        if not dt:
            return dt
        return cls.LOCAL_TZ.localize(dt)

    @classmethod
    def day_start(cls, dt: datetime) -> datetime:
        """ Return the current local time. """
        dt = cls.to_locale(dt)
        return cls.strptime(dt.strftime('%Y-%m-%d'), '%Y-%m-%d')

    @classmethod
    def month_start(cls, dt: datetime) -> datetime:
        """ Return the current local time. """
        dt = cls.to_locale(dt)
        return cls.strptime(dt.strftime('%Y-%m-01'), '%Y-%m-%d')

    @classmethod
    def next_month_start(cls, dt: datetime) -> datetime:
        """ Return the current local time. """
        return cls.month_start(cls.month_start(dt) + timedelta(days=31))

    @classmethod
    def year_start(cls, dt: datetime) -> datetime:
        """ Return the current local time. """
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
        """ Check if timezone information is present. """
        if not dt.tzinfo:
            raise Exception('datetime Type must specify a time zone')
        return dt

    @classmethod
    def get_date_month(cls, dt: datetime, mon=0):
        """
        Get the time several months ago/later
        """
        if not dt.tzinfo:
            raise Exception('datetime Type must specify a time zone')

        if mon < 0:
            return dt - relativedelta(months=-mon)
        else:
            return dt + relativedelta(months=mon)
