from collections import defaultdict
import math
from mxnet import nd, gpu, cpu
from mxnet.util import get_gpu_count

from model import HybridCNNLSTM, predict
from logger import logger
from config import PARAMS_PATH

from .api.intelligence import Iface


gpu_counts = get_gpu_count()
devices = [gpu(i) for i in range(gpu_counts)] if gpu_counts > 0 else [cpu()]

def get_model():
    model = HybridCNNLSTM()
    model.load_parameters(PARAMS_PATH, ctx=devices)
    return model


TIMES = 111000      # 1经度 = 111千米
model = get_model()

XY_TRACK_POINTS = defaultdict(list)
LL_TRACK_POINTS = defaultdict(list)

# 时间,经度,纬度,速度,高度,航向
class IntelligenceHandler(Iface):
    def forecast_xy(self, fn: str, t: float, x: float, y: float, h: float, v: float, course: float, dx: float, dy: float) -> float:
        global XY_TRACK_POINTS
        XY_TRACK_POINTS[fn].append((t, x, y, v, h, course))
        XY_TRACK_POINTS[fn] = XY_TRACK_POINTS[fn][-1000:]
        if len(XY_TRACK_POINTS[fn]) <= 6:
            return 0
        basics = nd.array(XY_TRACK_POINTS[fn][-6:])
        ends = predict(model, basics)
        logger.info(f'receive params: ({t}, {x}, {y}, {v}, {h}, {course})')
        return math.sqrt((ends[0] - dx) ** 2 + (ends[1] - dy) ** 2) / v
    
    def forecast_ll(self, fn: str, t: float, longi: float, lati: float, h: float, v: float, course: float, dlongi: float, dlati: float) -> float:
        global LL_TRACK_POINTS
        LL_TRACK_POINTS[fn].append((t, longi, lati, v, h, course))
        LL_TRACK_POINTS[fn] = LL_TRACK_POINTS[fn][-1000:]
        if len(LL_TRACK_POINTS[fn]) <= 6:
            return 0
        basics = nd.array(LL_TRACK_POINTS[fn][-6:])
        ends = predict(model, basics)
        logger.info(f'receive params: ({t}, {longi}, {lati}, {v}, {h}, {course})')
        return math.sqrt((ends[0] - dlongi) ** 2 + (ends[1] - dlati) ** 2) * TIMES / v
