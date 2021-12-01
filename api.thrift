service intelligence {
    i32 forecast_xy(1:string fn, 2:double x, 3:double y, 4:double h, 5:double v, 6:double course, 7:double dx, 8:double dy);
    i32 forecast_ll(1:string fn, 2:double longi, 3:double lati, 4:double h, 5:double v, 6:double course, 7:double dlongi, 8:double dlati);
}

// @Return 需要的时间（秒）

// fn 航班号
// x y 坐标
// longi lati 经纬度
// dx dy 目标坐标
// dlongi dlati 目标经纬度
/// ↓ 以下需要做转换
// h 高度
// v 速度
// course 朝向
