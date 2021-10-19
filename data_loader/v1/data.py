from datetime import datetime
from typing import Tuple, Union

from .config import HEIGHT_MAX, HEIGHT_MIN, LATITUDE_MAX, LATITUDE_MIN, LONGITUDE_MAX, LONGITUDE_MIN, TIMESTAMP_MIN

from .rule import DataRule


class PlaneDataSimple:
    def __init__(self, longi: float, lati: float, h: float) -> None:
        self.height = h             # 几何高度
        self.longitude = longi      # 经度
        self.latitude = lati        # 纬度
    
    @classmethod
    def from_tuple(cls, dt: Tuple[float, float, float]) -> 'PlaneDataSimple':
        longitude = dt[0] * (LONGITUDE_MAX - LONGITUDE_MIN) / (HEIGHT_MAX - HEIGHT_MIN) + LONGITUDE_MIN
        latitude = dt[1] * (LATITUDE_MAX - LATITUDE_MIN) / (HEIGHT_MAX - HEIGHT_MIN) + LATITUDE_MIN
        height = dt[2] + HEIGHT_MIN
        return cls(longitude, latitude, height)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        # 归一化
        return (self.longitude - LONGITUDE_MIN) * (HEIGHT_MAX - HEIGHT_MIN) / (LONGITUDE_MAX - LONGITUDE_MIN), \
            (self.latitude - LATITUDE_MIN) * (HEIGHT_MAX - HEIGHT_MIN) / (LATITUDE_MAX - LATITUDE_MIN), \
            (self.height - HEIGHT_MIN)

    def __str__(self) -> str:
        return str([self.longitude, self.latitude, self.height])


class PlaneData(PlaneDataSimple):
    def __init__(self, dt: Union[datetime, str], fn: str, h: float, longi: float, lati: float, is_available: bool=True) -> None:
        super().__init__(longi, lati, h)
        if isinstance(dt, datetime):
            self.datetime = dt
        else:
            self.datetime = datetime.fromisoformat(dt)  # 时间
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
