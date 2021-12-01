from typing import Generator, List, Tuple
import os
from collections import defaultdict
from datetime import datetime, timedelta
import math
import random
import pandas
from mxnet import nd

from .data import PlaneData
from .config import DATA_DIR, TRACK_MIN_POINT_NUM, TRACK_POINT_TIME_INTERVAL, TRACK_TIME_INTERVAL_MAX


FEATURES_COLUMNS = ['系统接收时间', '航班号', '几何高度', '经度', '纬度', '飞行高度', '地速', '航向']


def process_data_a_file(path: str) -> Generator[List[PlaneData], None, None]:
    result = defaultdict(list)
    # 读取
    df = pandas.read_csv(path, encoding='UTF-8')
    for vl in df.loc[:, FEATURES_COLUMNS].iterrows():
        pd = PlaneData(*vl[1].to_list())
        result[pd.flight_number].append(pd)
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
    h2 = (at * b.height_2 + bt * a.height_2) / (at + bt) if a.height_2 != b.height_2 else a.height_2
    gs = (at * b.ground_speed + bt * a.ground_speed) / (at + bt) if a.ground_speed != b.ground_speed else a.ground_speed
    course = (at * b.course + bt * a.course) / (at + bt) if a.course != b.course else a.course
    return PlaneData(t, a.flight_number, height, longitude, latitude, h2, gs, course)


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


def sampling_track(track: List[PlaneData], s: int) -> List[PlaneData]:
    '''每隔s秒取一个航迹点'''
    microseconds = s * int(1e6)
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
            result.append(generate_track_point(track[pf], track[pr], start))
    return result


def data_track_iter() -> Generator[List[PlaneData], None, None]:
    # 一个txt表示一天的数据
    for s in os.listdir(DATA_DIR):
        if not s.endswith('.csv'):
            continue
        # day, ext = os.path.splitext(s)
        s = os.path.join(DATA_DIR, s)
        for track in process_data_a_file(s):
            track_2 = sampling_track(track, TRACK_POINT_TIME_INTERVAL)
            if len(track_2) > TRACK_MIN_POINT_NUM:
                yield track_2
            # else:
            #     print('丢弃', track_2[0].flight_number, len(track))


def data_steps_iter(num_steps: int) -> Generator[Tuple[List[PlaneData], List[PlaneData]], None, None]:
    '''航迹序列遍历'''
    for track in data_track_iter():
        track_len = len(track)
        idxex = list(range((track_len - 1) // num_steps))
        random.shuffle(idxex)
        for i in idxex:
            X = track[i * num_steps : (i + 1) * num_steps]
            Y = track[i * num_steps + 1 : (i + 1) * num_steps + 1]
            yield X, Y


# def data_iter_pre(batch_size: int, num_steps: int) -> None:
#     Xs, Ys = [], []
#     matrix, matriy = [], []
#     for step_x, step_y in data_steps_iter(num_steps):
#         if len(matrix) >= batch_size:
#             X, Y = nd.array(matrix).asnumpy(), nd.array(matriy).asnumpy()
#             Xs.append(X)
#             Ys.append(Y)
#             matrix.clear()
#             matriy.clear()
#         matrix.append([x.to_tuple() for x in step_x])
#         matriy.append([y.to_tuple() for y in step_y])
#     np.savez(os.path.join(DATA_RUNTIME_DIR, 'X.npz'), *Xs)
#     np.savez(os.path.join(DATA_RUNTIME_DIR, 'Y.npz'), *Ys)


DATA_X = []
DATA_Y = []

def data_iter(batch_size: int, num_steps: int) -> Generator[Tuple[nd.NDArray, nd.NDArray], None, None]:
    if len(DATA_X) > 0 and len(DATA_Y) > 0:
        for X, Y in zip(DATA_X, DATA_Y):
            yield X, Y
    else:
        matrix, matriy = [], []
        for step_x, step_y in data_steps_iter(num_steps):
            if len(matrix) >= batch_size:
                X, Y = nd.array(matrix), nd.array(matriy)
                DATA_X.append(X)
                DATA_Y.append(Y)
                yield X, Y
                matrix.clear()
                matriy.clear()
            matrix.append([x.to_tuple() for x in step_x])
            matriy.append([y.to_tuple() for y in step_y])
