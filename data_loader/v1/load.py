from typing import Generator, List, Tuple
import os
from collections import defaultdict
from datetime import datetime, timedelta
import math
import torch

from .data import PlaneData
from .config import DATA_DIR, TRACK_MIN_POINT_NUM, TRACK_POINT_TIME_INTERVAL, TRACK_TIME_INTERVAL_MAX


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


def generate_track_point(a: PlaneData, b: PlaneData, t: datetime) -> PlaneData:
    '''基于两点生成中间需要的点'''
    at, bt = (t - a.datetime).total_seconds(), (b.datetime - t).total_seconds()
    longitude = (at * b.longitude + bt * a.longitude) / (at + bt) if a.longitude != b.longitude else a.longitude
    latitude = (at * b.latitude + bt * a.latitude) / (at + bt) if a.latitude != b.latitude else a.latitude
    height = (at * b.height + bt * a.height) / (at + bt) if a.height != b.height else a.height
    return PlaneData(t, a.flight_number, height, longitude, latitude)


def max_min_time_interval(track: List[PlaneData]) -> Tuple[float, float]:
    min_delta, max_delta = 100000.0, -1.0
    now = None
    for pd in track:
        if now is None:
            now = pd.datetime
        else:
            delta = pd.datetime - now
            now = pd.datetime
            min_delta = min(min_delta, delta.total_seconds())
            max_delta = max(max_delta, delta.total_seconds())
    return max_delta, min_delta


def find_pf_pr(track: List[PlaneData], t: datetime) -> Tuple[int, int]:
    start = 0
    end = len(track) - 1
    while start < end:
        media = (start + end) // 2
        if track[media].datetime < t:
            start = media + 1
        elif track[media].datetime > t:
            end = media
        else:
            return media, media
    return end - 1, end



def sampling_track(track: List[PlaneData], microseconds: int) -> List[PlaneData]:
    '''每隔microseconds微秒取一个航迹点'''
    start = track[0].datetime
    end = track[-1].datetime
    num = math.floor((end - start).total_seconds() * (1e6)) // microseconds
    result = [track[0]]
    pf, pr = 0, 0   # track指针
    for _ in range(num - 1):
        start += timedelta(microseconds=microseconds)
        pf, pr = find_pf_pr(track, start)
        if pf == pr:
            result.append(track[pf])
        else:
            try:
                result.append(generate_track_point(track[pf], track[pr], start))
            except Exception:
                sampling_track(track, microseconds)
    return result


def data_track_iter() -> Generator[List[PlaneData], None, None]:
    # 一个txt表示一天的数据
    for s in os.listdir(DATA_DIR):
        if not s.endswith('.txt'):
            continue
        # day, ext = os.path.splitext(s)
        s = os.path.join(DATA_DIR, s)
        for track in process_data_a_day(s):
            track_2 = sampling_track(track, TRACK_POINT_TIME_INTERVAL)
            if len(track_2) > TRACK_MIN_POINT_NUM:
                yield track_2
            # else:
            #     print('丢弃', track_2[0].flight_number, len(track))


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
