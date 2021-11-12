from .api.intelligence import Iface


class IntelligenceHandler(Iface):
    def forecast_xy(self, fn, x, y, h, v, course):
        return super().forecast_xy(fn, x, y, h, v, course)
    
    def forecast_ll(self, fn, longi, lati, h, v, course):
        return super().forecast_ll(fn, longi, lati, h, v, course)
# TODO 实现
