from datetime import datetime
from typing import Optional, Tuple, Union

from .utils import fromisoformat


class PlaneDataSimple:
    def __init__(self, longi: float, lati: float, h: float) -> None:
        self.height = h             # 几何高度
        self.longitude = longi      # 经度
        self.latitude = lati        # 纬度
    
    def __str__(self) -> str:
        return str([self.longitude, self.latitude, self.height])


class PlaneData(PlaneDataSimple):
    def __init__(self, dt: Optional[Union[datetime, str]], fn: Optional[str], h: float, longi: float, lati: float, h2: float, gs: float, course: float, is_available: bool=True) -> None:
        super().__init__(longi, lati, h)
        if dt is None or isinstance(dt, datetime):
            self.datetime = dt
        else:
            self.datetime = fromisoformat(dt)           # 时间
        self.flight_number = fn                         # 航班号
        self.height_2 = h2                              # 飞行高度
        self.ground_speed = gs                          # 地速
        self.course = course                            # 航向
        self.is_available = is_available                # 数据是否有效（时间、航班号缺失的数据无效）
    
    def to_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return self.height, self.longitude, self.latitude, self.height_2, self.ground_speed, self.course
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float, float, float, float]):
        return cls(None, None, t[0], t[1], t[2], t[3], t[4], t[5])
