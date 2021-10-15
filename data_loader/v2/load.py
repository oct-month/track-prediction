from typing import Generator, List, Tuple
import os
from collections import defaultdict
import torch

from .data import PlaneData
from .config import DATA_DIR, TRACK_MIN_POINT_NUM, TRACK_TIME_INTERVAL_MAX


def process_data_a_day(path: str) -> Generator[List[PlaneData], None, None]:
    result = defaultdict(list)
    # 读取
    with open(path, 'r', encoding='UTF-8') as f:
        for s in f.readlines()[1:]:
            pd = PlaneData.from_str(s)
            if pd.is_available:
                result[pd.flight_number].append(pd)
    # 排序
    for k, v in result.items():
        result[k] = sorted(v, key=lambda t: t.datetime)
    # 拆分
    for v in result.values():
        now = None
        start = 0
        for i, pd in enumerate(v):
            if now is None:
                now = pd.datetime
            else:
                delta = (pd.datetime - now).total_seconds()
                now = pd.datetime
                if delta > TRACK_TIME_INTERVAL_MAX:     # 间隔超过600s，认为是不同航迹
                    # print('拆分 ', i, '-' , v[0].flight_number)
                    yield v[start:i]
                    start = i
        yield v[start:]


def data_track_iter() -> Generator[List[PlaneData], None, None]:
    # 一个txt表示一天的数据
    for s in os.listdir(DATA_DIR):
        if not s.endswith('.txt'):
            continue
        # day, ext = os.path.splitext(s)
        s = os.path.join(DATA_DIR, s)
        for track in process_data_a_day(s):
            if len(track) > TRACK_MIN_POINT_NUM:
                yield track


def data_iter(batch_size: int, num_steps: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    for track in data_track_iter():
        track_len = len(track)
        batch_len = track_len // batch_size
        steps = (batch_len - 1) // num_steps
        matrix = []
        for i in range(batch_size):
            matrix.append([t.to_tuple() for t in track[i * batch_len : (i + 1) * batch_len]])
        dts = torch.tensor(matrix)
        for i in range(steps):
            X = dts[:, i * num_steps : (i + 1) * num_steps]
            Y = dts[:, i * num_steps + 1 : (i + 1) * num_steps + 1]
            yield X, Y
