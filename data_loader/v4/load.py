import os
import pickle
import pandas as pd
from mxnet import nd

from config import DATA_DIR_3 as DATA_PRE_DIR, DATA_DIR_4 as DATA_AFTER_DIR, FEATURES_COLUMNS, LABEL_COLUMNS

DATA_DIR = DATA_PRE_DIR
BASE_INDEX_FILE = 1000000000000


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


def data_iter_pre(batch_size):
    idx = 1
    X = []
    Y = []
    for pp in os.listdir(DATA_PRE_DIR):
        p = os.path.join(DATA_PRE_DIR, pp)
        df = pd.read_csv(p, sep=',')
        batch_length = (df.shape[0] - 7 + 1) // batch_size
        for i in range(batch_length):
            for j in range(batch_size):
                jx = i + j * batch_length
                feature = df.loc[jx:jx+5, FEATURES_COLUMNS].T
                label = df.loc[jx+6, LABEL_COLUMNS]
                X.append(feature.to_numpy().tolist())
                Y.append(label.to_numpy().tolist())
            with open(os.path.join(DATA_AFTER_DIR, 'f' + str(1000000000000 + idx) + '.pt'), 'wb') as f:
                pickle.dump(X, f)
            with open(os.path.join(DATA_AFTER_DIR, 'l' + str(1000000000000 + idx) + '.pt'), 'wb') as f:
                pickle.dump(Y, f)
            X.clear()
            Y.clear()
            idx += 1
            print(idx, end=' ')


def data_iter_load(ctx=None):
    for pp in os.listdir(DATA_AFTER_DIR):
        px = os.path.join(DATA_AFTER_DIR, pp)
        if pp.startswith('f'):
            py = os.path.join(DATA_AFTER_DIR, 'l' + pp[1:])
            with open(px, 'rb') as f:
                X = nd.array(pickle.load(f), ctx=ctx)
            with open(py, 'rb') as f:
                Y = nd.array(pickle.load(f), ctx=ctx)
            yield X, Y
