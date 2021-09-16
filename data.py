from datetime import datetime

from rule import DataRule

class PlaneData:
    def __init__(self, dt, fn, h, longi, lati, is_available=True) -> None:
        self.datetime = datetime.fromisoformat(dt)  # 时间
        self.flight_number = fn                     # 航班号
        self.height = h                             # 几何高度
        self.longitude = longi                      # 经度
        self.latitude = lati                        # 纬度
        self.is_available = is_available    # 数据是否有效

    @classmethod
    def from_str(cls, dt: str) -> 'PlaneData':
        flag = True
        cs = dt.split('\t')
        # for i in DataRule.iters:
        #     if cs[i].strip() == '':
        #         flag = False
        longi, lati = (cs[DataRule.longitude_latitude].strip() or '0,0').split(',')
        pd = cls(
            cs[DataRule.datetime].strip() or '2000-01-01 00:00:00.000000',
            cs[DataRule.flight_number].strip(),
            float(cs[DataRule.height].strip() or cs[DataRule.heigth_2].strip() or '0'),
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
