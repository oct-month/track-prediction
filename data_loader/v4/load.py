import os
import pandas as pd
from mxnet import nd

DATA_DIR = './datasets/train'

FEATURES_COLUMNS = ['时间', '经度', '纬度', '速度', '高度', '航向']
LABEL_COLUMNS = ['时间', '经度', '纬度', '高度']


def data_iter(batch_size, ctx=None):
    X = []
    Y = []
    for pp in os.listdir(DATA_DIR):
        p = os.path.join(DATA_DIR, pp)
        df = pd.read_csv(p, sep=',')
        batch_length = (df.shape[0] - 7 + 1) // batch_size
        for i in range(batch_length):
            for j in range(batch_size):
                jx = i + j * batch_length
                feature = df.loc[jx:jx+5, FEATURES_COLUMNS].T
                label = df.loc[jx+6, LABEL_COLUMNS]
                X.append(feature.to_numpy().tolist())
                Y.append(label.to_numpy().tolist())
            yield nd.array(X, ctx=ctx), nd.array(Y, ctx=ctx)
            X.clear()
            Y.clear()


def data_iter_order(batch_size, ctx=None):
    X = []
    Y = []
    for pp in os.listdir(DATA_DIR):
        p = os.path.join(DATA_DIR, pp)
        df = pd.read_csv(p, sep=',')
        for i in range(df.shape[0] - 6):
            features = df.loc[i:i+5, FEATURES_COLUMNS].T
            label = df.loc[i+6, LABEL_COLUMNS]
            X.append(features.to_numpy().tolist())
            Y.append(label.to_numpy().tolist())
            if len(X) >= batch_size:
                yield nd.array(X, ctx=ctx), nd.array(Y, ctx=ctx)
                X.clear()
                Y.clear()

