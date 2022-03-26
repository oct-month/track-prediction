import os
import pandas as pd
from mxnet import nd

DATA_DIR = './datasets/train'

FEATURES_COLUMNS = ['时间', '经度', '纬度', '速度', '高度', '航向']
LABEL_COLUMNS = ['时间', '经度', '纬度', '高度']


X = []
Y = []

def data_iter(batch_size: int, ctx=None):
    for pp in os.listdir(DATA_DIR):
        p = os.path.join(DATA_DIR, pp)
        df = pd.read_csv(p, sep=',')
        for i in range(df.shape[0] - 7):
            features = df.loc[i:i+6, FEATURES_COLUMNS].T
            label = df.loc[i+7, LABEL_COLUMNS]
            X.append(nd.array(features.to_numpy().tolist()))
            Y.append(nd.array(label.to_numpy().tolist()))
            if len(X) >= batch_size:
                yield nd.array(X, ctx=ctx), nd.array(Y, ctx=ctx)
                X.clear()
                Y.clear()
