from datetime import datetime

DATA_DIR = './data/'
NUM_FEATURES = 4        # 特征纬度
TRACK_TIME_INTERVAL_MAX = 900       # 间隔超过900秒的航迹点认为是两条航迹
TRACK_MIN_POINT_NUM = 1000          # 一条航迹最少应有1000航迹点
HEIGHT_MIN = 0                      # 飞机最低高度
HEIGHT_MAX = 20000                  # 飞机最高高度
TIMESTAMP_MIN = datetime.fromisoformat('2000-01-01 00:00:00.000000').timestamp()    # 最小时间戳
TIMESTAMP_MAX = datetime.fromisoformat('2100-01-01 00:00:00.000000').timestamp()    # 最大时间戳
