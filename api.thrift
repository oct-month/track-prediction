service intelligence {
    i32 forecast_xy(1:string fn, 2:double x, 3:double y, 4:double h, 5:double v, 6:double course);
    i32 forecast_ll(1:string fn, 2:double longi, 3:double lati, 4:double h, 5:double v, 6:double course);
}
