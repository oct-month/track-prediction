from datetime import datetime

from config import DATA_AFTER_DIR as DATA_DIR

NUM_FEATURES = 6    # 特征纬度
# INT_NULL = -9223372036854775808
# EXPAND_TIMES = 10000                # 扩大特征倍数
TRACK_TIME_INTERVAL_MAX = 600       # 间隔超过600秒的航迹点认为是两条航迹
TRACK_POINT_TIME_INTERVAL = 500_000 # 航迹点间隔设定为0.5秒
TRACK_MIN_POINT_NUM = 500           # 一条航迹最少应有500航迹点
# HEIGHT_MIN = 0                      # 飞机最低高度
# HEIGHT_MAX = 16000                  # 飞机最高高度
# LONGITUDE_MIN = 70                  # 最小经度
# LONGITUDE_MAX = 130                 # 最大经度
# LATITUDE_MIN = 20                   # 最小纬度
# LATITUDE_MAX = 50                   # 最大纬度
