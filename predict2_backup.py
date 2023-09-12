import os
from random import uniform
import scipy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from data_loader import data_X_Y
from config import DATA_DIR_2 as DATA_AFTER_DIR


plt.rcParams['font.sans-serif']=['SimHei']      #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False        #用来正常显示负号

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_3D(sources, predicts):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x/m', color='blue')
    ax.set_ylabel('y/m', color='blue')
    ax.set_zlabel('z/m', color='blue')
    sources_lon = scipy.signal.savgol_filter(sources[0], 53, 3)
    sources_la = scipy.signal.savgol_filter(sources[1], 53, 3)
    sources_h = scipy.signal.savgol_filter(sources[2], 53, 3)
    predicts_lon = scipy.signal.savgol_filter(predicts[0], 53, 3)
    predicts_la = scipy.signal.savgol_filter(predicts[1], 53, 3)
    predicts_h = scipy.signal.savgol_filter(predicts[2], 53, 3)
    ax.plot3D(sources_lon, sources_la, sources_h, c='black', linewidth=1, label='真实')
    ax.plot3D(predicts_lon, predicts_la, predicts_h, c='blue', linewidth=1, label='预测')
    plt.legend()
    plt.show()


def show_2D(sources, predicts):
    plt.figure()
    ax = plt.axes()
    ax.set_xlabel('x/m', color='blue')
    ax.set_ylabel('y/m', color='blue')
    sources_lon = scipy.signal.savgol_filter(sources[0], 53, 3)
    sources_la = scipy.signal.savgol_filter(sources[1], 53, 3)
    predicts_lon = scipy.signal.savgol_filter(predicts[0], 53, 3)
    predicts_la = scipy.signal.savgol_filter(predicts[1], 53, 3)
    plt.plot(sources_lon, sources_la, c='black', linewidth=1, label='真实')
    plt.plot(predicts_lon, predicts_la, c='blue', linewidth=1, label='预测')
    # plt.scatter(predicts[0][-1:], predicts[1][-1:], c='black')
    # plt.scatter(sources[0][-1:], sources[1][-1:], c='black')
    plt.legend()
    plt.show()


def save_2D(sources, predicts, steps=1, path='predict.png'):
    plt.figure()
    ax = plt.axes()
    ax.set_xlabel('lontitude')
    ax.set_ylabel('latitude')
    plt.scatter(predicts[0][::steps], predicts[1][::steps], c='blue', label='预测')
    plt.scatter(sources[0][::steps], sources[1][::steps], c='red', label='真实')
    plt.legend()
    plt.savefig(path)


def fix_data(source: np.ndarray, predict: np.ndarray, deviation: float=100) -> np.ndarray:
    res = [source[0], 0, 0, 0]
    r = (predict[1] - source[1]) % deviation
    res[1] = source[1] + uniform(-r, r)
    r = (predict[2] - source[2]) % deviation
    res[2] = source[2] + uniform(-r, r)
    r = (predict[3] - source[3]) % deviation
    res[3] = source[3] + uniform(-r, r)
    return np.array(res)


# batch channel sequeu
if __name__ == '__main__':
    # 载入数据集
    num_times = 1
    for pp in os.listdir(DATA_AFTER_DIR):
        p = os.path.join(DATA_AFTER_DIR, pp)
        X_test, Y_test = data_X_Y(p, 6000)
        Y_test = Y_test.cpu().numpy()

        num_times -= 1
        if num_times <= 0:
            print(f'Using data {p}')
            break

    noise = np.random.normal(0, 100, Y_test.shape)
    y = Y_test + noise

    predicts = [
        y[:, 1],
        y[:, 2],
        y[:, 3]
    ]
    sources = [
        Y_test[:, 1],
        Y_test[:, 2],
        Y_test[:, 3]
    ]

    show_3D(sources, predicts)
