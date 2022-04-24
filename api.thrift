service intelligence {
    double forecast_xy(1:string fn, 2:double t, 3:double x, 4:double y, 5:double h, 6:double v, 7:double course, 8:double dx, 9:double dy);
    double forecast_ll(1:string fn, 2:double t, 3:double longi, 4:double lati, 5:double h, 6:double v, 7:double course, 8:double dlongi, 9:double dlati);
}

// @Return 需要的时间（秒）

// fn 航班号
// t 时间戳
// x y 坐标
// longi lati 经纬度
// dx dy 目标坐标
// dlongi dlati 目标经纬度
/// ↓ 以下需要做转换
// h 高度
// v 速度
// course 朝向
