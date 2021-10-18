from datetime import datetime

DATA_DIR = './data/'
NUM_FEATURES = 3    # 特征纬度
TRACK_TIME_INTERVAL_MAX = 600       # 间隔超过600秒的航迹点认为是两条航迹
TRACK_POINT_TIME_INTERVAL = 500000  # 航迹点间隔设定为0.5秒
TRACK_MIN_POINT_NUM = 500           # 一条航迹最少应有500航迹点
HEIGHT_MIN = 0                      # 飞机最低高度
HEIGHT_MAX = 20000                  # 飞机最高高度
TIMESTAMP_MIN = datetime.fromisoformat('2000-01-01 00:00:00.000000').timestamp()    # 最小时间戳
