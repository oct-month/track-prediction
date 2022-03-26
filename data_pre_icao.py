import pandas as pd
import os

DATA_PRE_DIR = './datasets/radar'
DATA_AFTER_DIR = './datasets/cache'
TRACK_MIN_POINT_NUM = 50                # 一条航迹最少应有50航迹点

FEATURES_COLUMNS = ['时间', '航班号', '经度', '纬度', '速度', '高度', '航向']

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


p = os.path.join(DATA_PRE_DIR, '20210401.txt')
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
    flag = True
    for feature in FEATURES_COLUMNS[2:]:
        if all(dt[feature].isna()):
            flag = False
            break
    if flag:
        # 插值
        dt.interpolate(method='time', axis=0, inplace=True)
        dt.dropna(axis=0, how='any', subset=FEATURES_COLUMNS[2:], inplace=True)
        # 保存csv
        if dt.shape[0] >= TRACK_MIN_POINT_NUM:
            file_name = os.path.join(DATA_AFTER_DIR, pn + '-' + i + '.csv')
            dt.to_csv(file_name, index=True, encoding='UTF-8')
