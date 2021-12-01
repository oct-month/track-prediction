from datetime import datetime

from config import DATA_AFTER_DIR as DATA_DIR

NUM_FEATURES = 6                    # 特征纬度
TRACK_TIME_INTERVAL_MAX = 600       # 间隔超过600秒的航迹点认为是两条航迹
TRACK_POINT_TIME_INTERVAL = 10      # 航迹点间隔设定为10秒
TRACK_MIN_POINT_NUM = 50            # 一条航迹最少应有50航迹点
