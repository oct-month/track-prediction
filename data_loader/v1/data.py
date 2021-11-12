from datetime import datetime
from typing import Optional, Tuple, Union

from .utils import fromisoformat
# from .config import HEIGHT_MAX, HEIGHT_MIN, INT_NULL, LATITUDE_MAX, LATITUDE_MIN, LONGITUDE_MAX, LONGITUDE_MIN
# from .rule import DataRule


class PlaneDataSimple:
    def __init__(self, longi: float, lati: float, h: float) -> None:
        self.height = h             # 几何高度
        self.longitude = longi      # 经度
        self.latitude = lati        # 纬度
    
    # @classmethod
    # def from_tuple(cls, dt: Tuple[float, float, float]) -> 'PlaneDataSimple':
    #     print('方法已过时')
    #     longitude = dt[0] * (LONGITUDE_MAX - LONGITUDE_MIN) / (HEIGHT_MAX - HEIGHT_MIN) + LONGITUDE_MIN
    #     latitude = dt[1] * (LATITUDE_MAX - LATITUDE_MIN) / (HEIGHT_MAX - HEIGHT_MIN) + LATITUDE_MIN
    #     height = dt[2] + HEIGHT_MIN
    #     return cls(longitude, latitude, height)
    
    # def to_tuple(self) -> Tuple[float, float, float]:
    #     print('方法已过时')
    #     # 归一化
    #     return (self.longitude - LONGITUDE_MIN) * (HEIGHT_MAX - HEIGHT_MIN) / (LONGITUDE_MAX - LONGITUDE_MIN), \
    #         (self.latitude - LATITUDE_MIN) * (HEIGHT_MAX - HEIGHT_MIN) / (LATITUDE_MAX - LATITUDE_MIN), \
    #         (self.height - HEIGHT_MIN)

    def __str__(self) -> str:
        return str([self.longitude, self.latitude, self.height])


class PlaneData(PlaneDataSimple):
    def __init__(self, dt: Optional[Union[datetime, str]], fn: Optional[str], h: float, longi: float, lati: float, h2: float, gs: float, course: float, is_available: bool=True) -> None:
        super().__init__(longi, lati, h)
        if dt is None or isinstance(dt, datetime):
            self.datetime = dt
        else:
            self.datetime = fromisoformat(dt)  # 时间
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

    # @classmethod
    # def from_str(cls, dt: str) -> 'PlaneData':
    #     print('方法已过时')
    #     cs = dt.split('\t')
    #     if not cs[DataRule.longitude_latitude].strip():
    #         tap = cs[DataRule.longitude_latitude].split(',')
    #         longi, lati = float(tap[0]), float(tap[1])
    #     else:
    #         longi, lati = INT_NULL, INT_NULL
    #     pd = cls(
    #         cs[DataRule.datetime].strip() or None,
    #         cs[DataRule.flight_number].strip(),
    #         float(cs[DataRule.height]) if cs[DataRule.height].strip() else INT_NULL,
    #         longi,
    #         lati,
    #         float(cs[DataRule.heigth_2]) if cs[DataRule.heigth_2].strip() else INT_NULL,
    #         float(cs[DataRule.ground_speed]) if cs[DataRule.ground_speed].strip() else INT_NULL,
    #         float(cs[DataRule.course]) if cs[DataRule.course].strip() else INT_NULL
    #     )
    #     if pd.datetime is None or pd.flight_number == '':# or pd.height == INT_NULL or pd.longitude == INT_NULL or pd.latitude == INT_NULL:
    #         pd.is_available = False
    #     return pd

    # def __str__(self) -> str:
    #     print('方法已过时')
    #     result = [' '] * (DataRule.last + 1)
    #     result[DataRule.flight_number] = self.flight_number
    #     result[DataRule.datetime] = str(self.datetime) if self.datetime is not None else ''
    #     result[DataRule.longitude_latitude] = f'{self.longitude},{self.latitude}' if self.longitude != INT_NULL else ''
    #     result[DataRule.height] = str(self.height) if self.height != INT_NULL else ''
    #     result[DataRule.heigth_2] = str(self.height_2) if self.height_2 != INT_NULL else ''
    #     result[DataRule.ground_speed] = str(self.ground_speed) if self.ground_speed != INT_NULL else ''
    #     result[DataRule.course] = str(self.course) if self.course != INT_NULL else ''
    #     return '\t'.join(result)
