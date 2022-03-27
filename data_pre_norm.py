'''归一化'''
from datetime import datetime
import pandas as pd
import os

from config import DATA_DIR_2 as DATA_PRE_DIR, DATA_DIR_3 as DATA_AFTER_DIR, FEATURES_COLUMNS, FEATURES_NORMALIZATION, NORMALIZATION_TIMES


def convert_datetime_numric(value: datetime, **extra):
    return value.timestamp()

def datetime_normalization(value: float, **extra):
    return NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[0][0]) / (FEATURES_NORMALIZATION[0][1] - FEATURES_NORMALIZATION[0][0])

def longitude_normalization(value: float, **extra):
    return NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[1][0]) / (FEATURES_NORMALIZATION[1][1] - FEATURES_NORMALIZATION[1][0])

def latitude_normalization(value: float, **extra):
    return NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[2][0]) / (FEATURES_NORMALIZATION[2][1] - FEATURES_NORMALIZATION[2][0])

def speed_normalization(value: float, **extra):
    return NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[3][0]) / (FEATURES_NORMALIZATION[3][1] - FEATURES_NORMALIZATION[3][0])

def height_normalization(value: float, **extra):
    return NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[4][0]) / (FEATURES_NORMALIZATION[4][1] - FEATURES_NORMALIZATION[4][0])

def course_normalization(value: float, **extra):
    return NORMALIZATION_TIMES * (value - FEATURES_NORMALIZATION[5][0]) / (FEATURES_NORMALIZATION[5][1] - FEATURES_NORMALIZATION[5][0])


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
