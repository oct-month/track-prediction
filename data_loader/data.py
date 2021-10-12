from datetime import datetime
from typing import Tuple, Union, List

from data_loader.config import HEIGHT_MAX, HEIGHT_MIN

from .rule import DataRule

class PlaneData:
    def __init__(self, dt: Union[datetime, str], fn: str, h: float, longi: float, lati: float, is_available: bool=True) -> None:
        if isinstance(dt, datetime):
            self.datetime = dt
        else:
            self.datetime = datetime.fromisoformat(dt)  # 时间
        self.flight_number = fn                     # 航班号
        self.height = h                             # 几何高度
        self.longitude = longi                      # 经度
        self.latitude = lati                        # 纬度
        self.is_available = is_available    # 数据是否有效（航班号、高度、经纬度缺失的数据无效）

    @classmethod
    def from_str(cls, dt: str) -> 'PlaneData':
        cs = dt.split('\t')
        longi, lati = (cs[DataRule.longitude_latitude].strip() or '0,0').split(',')
        pd = cls(
            cs[DataRule.datetime].strip() or '2000-01-01 00:00:00.000000',
            cs[DataRule.flight_number].strip(),
            float(cs[DataRule.height].strip() or '0'),
            float(longi),
            float(lati)
        )
        if pd.flight_number == '' or pd.height == 0 or (pd.longitude == 0 and pd.latitude == 0):
            pd.is_available = False
        return pd

    def __str__(self) -> str:
        result = [' '] * (DataRule.last + 1)
        result[DataRule.flight_number] = self.flight_number
        result[DataRule.datetime] = str(self.datetime)
        result[DataRule.longitude_latitude] = f'{self.longitude},{self.latitude}'
        result[DataRule.height] = str(self.height)
        return '\t'.join(result)

    def to_list(self) -> List[float]:
        # 归一化
        return [(self.longitude + 180) / 360, (self.latitude + 90) / 180, (self.height - HEIGHT_MIN) / HEIGHT_MAX]
        # return [self.longitude, self.latitude, self.height]
    
    @classmethod
    def from_list(cls, dt: List[float]) -> Tuple[float, float, float]:
        longitude = dt[0] * 360 - 180
        latitude = dt[1] * 180 - 90
        height = dt[2] * HEIGHT_MAX + HEIGHT_MIN
        return longitude, latitude, height

    # def to_tensor(self) -> torch.Tensor:
    #     return torch.tensor(self.to_list())
