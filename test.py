from datetime import datetime
from typing import Union, List

class DataRule:
    flight_number = 0       # 航班号
    datetime = 13           # 时间
    longitude_latitude = 7  # 经纬度
    height = 15             # 几何高度
    heigth_2 = 17           # 飞行高度

    iters = [flight_number, datetime, longitude_latitude, height]
    last = 20

num_height, num_height2 = 0, 0
num_none, num_all = 0, 0

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

        global num_height2, num_height, num_all, num_none
        if cs[DataRule.height].strip() == '' and cs[DataRule.heigth_2].strip() == '':
            num_none += 1
        elif cs[DataRule.height].strip() != '' and cs[DataRule.heigth_2].strip() != '':
            num_all += 1
        elif cs[DataRule.height].strip() != '' and cs[DataRule.heigth_2].strip() == '':
            num_height += 1
        else:
            num_height2 += 1

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

    def to_list(self) -> List[float]:
        return [self.longitude, self.latitude, self.height]
    
    # def to_tensor(self) -> torch.Tensor:
    #     return torch.tensor(self.to_list())

if __name__ == '__main__':
    # with open('data/20210401.txt', 'r', encoding='UTF-8') as f:
    #     t = f.readlines()[1:]
    #     for s in t:
    #         pd = PlaneData.from_str(s)
    # print("num_all", num_all)
    # print("num_none", num_none)
    # print("num_height", num_height)
    # print("num_height2", num_height2)

    t = datetime.fromtimestamp(57628.47345)
    print(t)
