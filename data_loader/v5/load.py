import os
import shutil
import pickle
import pandas as pd
import torch

from config import DATA_DIR_2 as DATA_PRE_DIR, DATA_DIR_4 as DATA_AFTER_DIR, FEATURES_COLUMNS, LABEL_COLUMNS

DATA_DIR = DATA_PRE_DIR
BASE_INDEX_FILE = 1000000000000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_iter(batch_size):
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
            yield torch.tensor(X, device=device), torch.tensor(Y, device=device)
            X.clear()
            Y.clear()


def data_iter_order(batch_size):
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
                yield torch.tensor(X, device=device), torch.tensor(Y, device=device)
                X.clear()
                Y.clear()


def data_iter_pre(batch_size):
    data_dir = DATA_AFTER_DIR + '-' + str(batch_size)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    idx = 0
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
            with open(os.path.join(data_dir, 'f' + str(1000000000000 + idx) + '.pt'), 'wb') as f:
                pickle.dump(X, f)
            with open(os.path.join(data_dir, 'l' + str(1000000000000 + idx) + '.pt'), 'wb') as f:
                pickle.dump(Y, f)
            X.clear()
            Y.clear()
            idx += 1
        print(idx, end=' ')


def data_iter_load(batch_size):
    data_dir = DATA_AFTER_DIR + '-' + str(batch_size)
    for pp in os.listdir(data_dir):
        px = os.path.join(data_dir, pp)
        if pp.startswith('f'):
            py = os.path.join(data_dir, 'l' + pp[1:])
            with open(px, 'rb') as f:
                X = torch.tensor(pickle.load(f), device=device)
            with open(py, 'rb') as f:
                Y = torch.tensor(pickle.load(f), device=device)
            yield X, Y
