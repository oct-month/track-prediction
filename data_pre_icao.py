import pandas as pd
import os

from config import DATA_DIR_1 as DATA_PRE_DIR, DATA_DIR_2 as DATA_AFTER_DIR, TRACK_MIN_POINT_NUM, FEATURES_COLUMNS, BASE_LATITUDE, BASE_LONGITUDE, PER_LATITUDE_M, PER_LONGITUDE_M


def convert_empty_str(value: pd.Series, **extra):
    return value.map(lambda x: x.strip() if isinstance(x, str) else x)


def get_longitude(value: str, **extra):
    if value is None or value is pd.NA or value.strip() == '':
        return pd.NA
    return float(value.split(',')[0])


def get_latitude(value: str, **extra):
    if value is None or value is pd.NA or value.strip() == '':
        return pd.NA
    return float(value.split(',')[1])

def longitude_tom(value: float, **extra):
    return (value - BASE_LONGITUDE) * PER_LONGITUDE_M

def latitude_tom(value: float, **extra):
    return (value - BASE_LATITUDE) * PER_LATITUDE_M


def main():
    for idx, pp in enumerate(os.listdir(DATA_PRE_DIR)):
        p = os.path.join(DATA_PRE_DIR, pp)
        df: pd.DataFrame = pd.read_table(p, sep='\t', encoding='UTF-8')

        # 空串替换为NA
        df.replace(to_replace=r'^\s*$', value=pd.NA, regex=True, inplace=True)
        # 0.00替换为NA
        df.replace(to_replace=r'^\s*?0*?\.?0*?\s*?$', value=pd.NA, regex=True, inplace=True)
        # 无航班号的删除
        df.dropna(axis=0, how='any', subset=['航班号(I170)'], inplace=True)
        # 字符串精简
        df = df.apply(convert_empty_str)

        pnames = set(df['航班号(I170)'].to_list())
        for i, pn in enumerate(pnames):
            dt = df.loc[df['航班号(I170)'] == pn]
            # 经纬度-高精度经纬度
            if all(dt['经纬度(I130)'].isna()):
                dt['经纬度(I130)'] = dt.loc[:, '高精度经纬度(I131)']
            # 几何高度-飞行高度
            if all(dt['几何高度(I140)'].isna()):
                dt['几何高度(I140)'] = dt.loc[:, '飞行高度(I145)']
            # 经纬度提取
            dt['经度'] = dt.loc[:, '经纬度(I130)'].apply(get_longitude)
            dt['纬度'] = dt.loc[:, '经纬度(I130)'].apply(get_latitude)
            # 取有用列
            dt.rename(columns={
                '系统接收时间': '时间',
                '航班号(I170)': '航班号',
                '地速(I160)': '速度',
                '几何高度(I140)': '高度',
                '航向(I160)': '航向'
            }, inplace=True)
            dt = dt.loc[:, FEATURES_COLUMNS]
            # 类型
            # dt['航班号'] = dt.loc[:, '航班号'].astype(str)
            dt['时间'] = dt.loc[:, '时间'].apply(pd.to_datetime, errors='raise', format='%Y-%m-%d %H:%M:%S.%f')
            dt.drop_duplicates(subset=['时间'], keep='last', inplace=True)
            dt.set_index('时间', inplace=True)
            dt[['经度', '纬度', '速度', '高度', '航向']] = dt.loc[:, ['经度', '纬度', '速度', '高度', '航向']].apply(pd.to_numeric, errors='coerce')
            # 丢弃无用数据
            dt.dropna(axis=0, how='any', inplace=True)
            # 丢弃重复行
            dt.drop_duplicates(subset=FEATURES_COLUMNS[1:3], keep='first', inplace=True)
            # 丢弃离谱数据
            # TODO
            # 经纬度转米
            dt['经度'] = dt.loc[:, '经度'].apply(longitude_tom)
            dt['纬度'] = dt.loc[:, '纬度'].apply(latitude_tom)
            # 丢弃平飞数据和过短数据
            # 保存csv
            if dt.shape[0] >= TRACK_MIN_POINT_NUM: # and abs(dt['高度'][-1] - dt['高度'][0]) >= 500:
                file_name = os.path.join(DATA_AFTER_DIR, pn + '-' + str(idx) + '.csv')
                dt.to_csv(file_name, index=True, encoding='UTF-8')
    print('data pre icao done.')

if __name__ == '__main__':
    main()
