from datetime import datetime
from typing import Tuple, Union

from .config import HEIGHT_MAX, HEIGHT_MIN, TIMESTAMP_MAX, TIMESTAMP_MIN

from .rule import DataRule


class PlaneDataSimple:
    def __init__(self, dt: Union[datetime, str], longi: float, lati: float, h: float) -> None:
        if isinstance(dt, datetime):
            self.datetime = dt
        else:
            self.datetime = datetime.fromisoformat(dt)  # 时间
        self.datetime.timestamp()
        self.height = h             # 几何高度
        self.longitude = longi      # 经度
        self.latitude = lati        # 纬度
    
    @classmethod
    def from_tuple(cls, dt: Tuple[float, float, float, float]) -> 'PlaneDataSimple':
        timestamp = dt[0] * (TIMESTAMP_MAX - TIMESTAMP_MIN) / HEIGHT_MAX + TIMESTAMP_MIN
        longitude = dt[1] * 360 / HEIGHT_MAX - 180
        latitude = dt[2] * 180 / HEIGHT_MAX - 90
        height = dt[3] + HEIGHT_MIN
        return cls(datetime.fromtimestamp(timestamp) , longitude, latitude, height)
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        # 归一化
        return (self.datetime.timestamp() - TIMESTAMP_MIN) * HEIGHT_MAX / (TIMESTAMP_MAX - TIMESTAMP_MIN), \
            (self.longitude + 180) * HEIGHT_MAX / 360, \
            (self.latitude + 90) * HEIGHT_MAX / 180, \
            (self.height - HEIGHT_MIN)

    def __str__(self) -> str:
        return str([self.datetime.timestamp(), self.longitude, self.latitude, self.height])


class PlaneData(PlaneDataSimple):
    def __init__(self, dt: Union[datetime, str], fn: str, h: float, longi: float, lati: float, is_available: bool=True) -> None:
        super().__init__(dt, longi, lati, h)
        self.flight_number = fn                     # 航班号
        self.is_available = is_available    # 数据是否有效（航班号、高度、经纬度缺失的数据无效）

    @classmethod
    def from_str(cls, dt: str) -> 'PlaneData':
        cs = dt.split('\t')
        longi, lati = (cs[DataRule.longitude_latitude].strip() or '0,0').split(',')
        pd = cls(
            cs[DataRule.datetime].strip() or datetime.fromtimestamp(TIMESTAMP_MIN),
            cs[DataRule.flight_number].strip(),
            float(cs[DataRule.height].strip() or '0'),
            float(longi),
            float(lati)
        )
        if pd.datetime == datetime.fromtimestamp(TIMESTAMP_MIN) or pd.flight_number == '' or pd.height == 0 or (pd.longitude == 0 and pd.latitude == 0):
            pd.is_available = False
        return pd

    def __str__(self) -> str:
        result = [' '] * (DataRule.last + 1)
        result[DataRule.flight_number] = self.flight_number
        result[DataRule.datetime] = str(self.datetime)
        result[DataRule.longitude_latitude] = f'{self.longitude},{self.latitude}'
        result[DataRule.height] = str(self.height)
        return '\t'.join(result)
