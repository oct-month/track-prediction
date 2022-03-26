'''归一化'''
from datetime import datetime
import pandas as pd
import os

DATA_PRE_DIR = './datasets/cache'
DATA_AFTER_DIR = './datasets/train'

FEATURES_COLUMNS = ['时间', '经度', '纬度', '速度', '高度', '航向']
LABEL_COLUMNS = ['时间', '经度', '纬度', '高度']
FEATURES_NORMALIZATION = [
    [1000000000, 4000000000],
    [-180, 180],
    [-90, 90],
    [0, 5000],
    [-10000, 20000],
    [0, 360]
]
FEATURES_NORMALIZATION_TIMES = 100


def convert_datetime_numric(value: datetime, **extra):
    return value.timestamp()

def datetime_normalization(value: float, **extra):
    return FEATURES_NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[0][0]) / (FEATURES_NORMALIZATION[0][1] - FEATURES_NORMALIZATION[0][0])

def longitude_normalization(value: float, **extra):
    return FEATURES_NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[1][0]) / (FEATURES_NORMALIZATION[1][1] - FEATURES_NORMALIZATION[1][0])

def latitude_normalization(value: float, **extra):
    return FEATURES_NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[2][0]) / (FEATURES_NORMALIZATION[2][1] - FEATURES_NORMALIZATION[2][0])

def speed_normalization(value: float, **extra):
    return FEATURES_NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[3][0]) / (FEATURES_NORMALIZATION[3][1] - FEATURES_NORMALIZATION[3][0])

def height_normalization(value: float, **extra):
    return FEATURES_NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[4][0]) / (FEATURES_NORMALIZATION[4][1] - FEATURES_NORMALIZATION[4][0])

def course_normalization(value: float, **extra):
    return FEATURES_NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[5][0]) / (FEATURES_NORMALIZATION[5][1] - FEATURES_NORMALIZATION[5][0])


if __name__ == '__main__':
    for pp in os.listdir(DATA_PRE_DIR):
        p = os.path.join(DATA_PRE_DIR, pp)
        df = pd.read_csv(p, sep=',')
        # 时间处理
        df['时间'] = df.loc[:, '时间'].apply(pd.to_datetime, errors='raise', format='%Y-%m-%d %H:%M:%S.%f')
        df['时间'] = df.loc[:, '时间'].apply(convert_datetime_numric)
        # 归一化
        df['时间'] = df.loc[:, '时间'].apply(datetime_normalization)
        df['经度'] = df.loc[:, '经度'].apply(longitude_normalization)
        df['纬度'] = df.loc[:, '纬度'].apply(latitude_normalization)
        df['速度'] = df.loc[:, '速度'].apply(speed_normalization)
        df['高度'] = df.loc[:, '高度'].apply(height_normalization)
        df['航向'] = df.loc[:, '航向'].apply(course_normalization)
        # 导出
        df.loc[:, FEATURES_COLUMNS].to_csv(os.path.join(DATA_AFTER_DIR, pp), index=False, encoding='UTF-8')
        # for i in range(df.shape[0] - 6 - 1 + 1):
        #     features = df.loc[i:i+6, FEATURES_COLUMNS]
        #     label = df.loc[i+7, LABEL_COLUMNS]

