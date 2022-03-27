'''数据清洗'''
import pandas as pd
import os

from config import DATA_PRE_DIR, DATA_AFTER_DIR

FEATURES_COLUMNS = ['航班号(I170)', '经纬度(I130)', '系统接收时间', '地速(I160)', '几何高度(I140)', '飞行高度(I145)', '航向(I160)']


def convert_empty_str(value: pd.Series, **extra):
    return value.map(lambda x: x.strip())


def get_longitude(value: str, **extra):
    if value.strip() == '':
        return None
    return float(value.split(',')[0])


def get_latitude(value: str, **extra):
    if value.strip() == '':
        return None
    return float(value.split(',')[1])


if __name__ == '__main__':
    for p in os.listdir(DATA_PRE_DIR):
        if p.endswith('.txt'):
            p = os.path.join(DATA_PRE_DIR, p)
            df = pd.read_table(p, sep='\t', encoding='UTF-8')
            # df = df.dropna(axis=1, how='all')   # 删除空列
            df.dropna(axis=0, subset=['航班号(I170)', '系统接收时间'], inplace=True)
            df = df.loc[:, FEATURES_COLUMNS]
            df = df.apply(convert_empty_str)
            df.replace('', None, inplace=True)
            df['经度'] = df.loc[:, '经纬度(I130)'].apply(get_longitude).astype(float)
            df['纬度'] = df.loc[:, '经纬度(I130)'].apply(get_latitude).astype(float)
            df.drop(['经纬度(I130)'], axis=1, inplace=True)

            df[['航班号(I170)', '系统接收时间']] = df[['航班号(I170)', '系统接收时间']].astype(str)
            df[['地速(I160)', '几何高度(I140)', '飞行高度(I145)', '航向(I160)']] = df[['地速(I160)', '几何高度(I140)', '飞行高度(I145)', '航向(I160)']].astype(float)
            pnames = set(df['航班号(I170)'].to_list())

            df.rename(columns={
                '航班号(I170)': '航班号',
                '地速(I160)': '地速',
                '几何高度(I140)': '几何高度',
                '飞行高度(I145)': '飞行高度',
                '航向(I160)': '航向'
            }, inplace=True)

            for pn in pnames:
                if pn is not None and pn.strip() != '':
                    dt = df.loc[df['航班号'] == pn]
                    dt = dt.interpolate()
                    file_name = os.path.join(DATA_AFTER_DIR, pn.strip() + '.csv')
                    dt.to_csv(file_name, index=False, encoding='UTF-8')
