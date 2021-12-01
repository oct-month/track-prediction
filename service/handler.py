from collections import defaultdict
import math

from .api.intelligence import Iface
from .model import get_model

from model import predict
from data_loader import PlaneData, TRACK_POINT_TIME_INTERVAL

TIMES = 111000      # 1经度 = 111千米
PREDICT_NUM = 10    # 预测的点
model = get_model()

XY_TRACK_POINTS = defaultdict(list)
LL_TRACK_POINTS = defaultdict(list)

class IntelligenceHandler(Iface):
    def forecast_xy(self, fn: str, x: float, y: float, h: float, v: float, course: float, dx: float, dy: float) -> float:
        global XY_TRACK_POINTS
        XY_TRACK_POINTS[fn].append(PlaneData.from_tuple((h, x, y, 0, v, course)))
        XY_TRACK_POINTS[fn] = XY_TRACK_POINTS[fn][-1000:]
        ends = predict(model, XY_TRACK_POINTS[fn], PREDICT_NUM)[-1]
        return math.sqrt((ends.longitude - dx) ** 2 + (ends.latitude - dy) ** 2) / v + PREDICT_NUM * TRACK_POINT_TIME_INTERVAL
    
    def forecast_ll(self, fn: str, longi: float, lati: float, h: float, v: float, course: float, dlongi: float, dlati: float) -> float:
        global LL_TRACK_POINTS
        LL_TRACK_POINTS[fn].append(PlaneData.from_tuple((h, longi, lati, 0, v, course)))
        LL_TRACK_POINTS[fn] = LL_TRACK_POINTS[fn][-1000:]
        ends = predict(model, LL_TRACK_POINTS[fn], PREDICT_NUM)[-1]
        return math.sqrt((ends.longitude - dlongi) ** 2 + (ends.latitude - dlati) ** 2) * TIMES / v + PREDICT_NUM * TRACK_POINT_TIME_INTERVAL
