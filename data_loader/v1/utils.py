from datetime import datetime


def fromisoformat(dt: str) -> datetime:
    '''2020-03-31 16:00:28.467'''
    return datetime.strptime(dt, '%Y-%m-%d %H:%M:%S.%f')
