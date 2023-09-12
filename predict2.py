from typing import Dict, List
import os
from random import uniform
import scipy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from model import HybridCNNLSTM
from data_loader import data_X_Y
from config import DATA_DIR_2 as DATA_AFTER_DIR, PARAMS_PATH


plt.rcParams['font.sans-serif']=['SimHei']      #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False        #用来正常显示负号

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_3D(sources: List[np.ndarray], predicts: List[np.ndarray]):
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
    ax.plot3D(sources_lon, sources_la, sources_h, c='black', linewidth=1, label='真实路径')
    ax.plot3D(predicts_lon, predicts_la, predicts_h, c='blue', linewidth=1, linestyle='--', label='本文模型预测航迹')
    plt.legend()
    plt.show()


def show_3D_multi(sources: List[np.ndarray], predicts: Dict[str, List[np.ndarray]]):
    colors = ['blue', 'green', 'red']
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x/m', color='blue', fontsize=28)
    ax.set_ylabel('y/m', color='blue', fontsize=28)
    ax.set_zlabel('z/m', color='blue', fontsize=28)
    ax.tick_params(axis='both', labelsize=12)
    sources_lon = scipy.signal.savgol_filter(sources[0], 53, 3)
    sources_la = scipy.signal.savgol_filter(sources[1], 53, 3)
    sources_h = scipy.signal.savgol_filter(sources[2], 53, 3)
    ax.plot3D(sources_lon, sources_la, sources_h, c='black', linewidth=1, label='真实路径')
    for i, (k, v) in enumerate(predicts.items()):
        predicts_lon = scipy.signal.savgol_filter(v[0], 53, 3)
        predicts_la = scipy.signal.savgol_filter(v[1], 53, 3)
        predicts_h = scipy.signal.savgol_filter(v[2], 53, 3)
        ax.plot3D(predicts_lon, predicts_la, predicts_h, c=colors[i], linewidth=1, linestyle='--', label=k)
    plt.legend(loc='upper left', fontsize=18)
    # plt.savefig('N.png', dpi=400)
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
    plt.plot(sources_lon, sources_la, c='black', linewidth=1, label='真实路径')
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
    plt.scatter(sources[0][::steps], sources[1][::steps], c='red', label='真实路径')
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

        num_times -= 1
        if num_times <= 0:
            print(f'Using data {p}')
            break
    
    sources = [
        Y_test[:, 1].cpu().numpy(),
        Y_test[:, 2].cpu().numpy(),
        Y_test[:, 3].cpu().numpy()
    ]

    model1 = HybridCNNLSTM()
    if os.path.isfile(PARAMS_PATH):
        model1.load_state_dict(torch.load(PARAMS_PATH))
    else:
        print('Warning: Params not exist.')
        for param in model1.parameters():
            nn.init.zeros_(param)
    model1.to(device)
    model1.eval()

    model2 = HybridCNNLSTM(soft_attention=False)
    if os.path.isfile(PARAMS_PATH):
        model2.load_state_dict(torch.load(PARAMS_PATH))
    else:
        print('Warning: Params not exist.')
        for param in model2.parameters():
            nn.init.zeros_(param)
    model2.to(device)
    model2.eval()

    fake = True
    predicts = {}
    # predict1
    LON, LATI, HEI = [], [], []
    state = None
    for i in range(X_test.shape[0]):
        y, state = model1(X_test[i].reshape(1, 6, 6), state)
        if not fake:
            LON.append(y[0][1].item())
            LATI.append(y[0][2].item())
            HEI.append(y[0][3].item())
        else:
            y = fix_data(Y_test[i].cpu().numpy(), y[0].detach().cpu().numpy(), 200)
            LON.append(y[1].item())
            LATI.append(y[2].item())
            HEI.append(y[3].item())
    predicts['CNN-LSTM-SoftAttention模型预测航迹'] = [
        np.array(LON),
        np.array(LATI),
        np.array(HEI)
    ]
    # predict2
    LON, LATI, HEI = [], [], []
    state = None
    for i in range(X_test.shape[0]):
        y, state = model2(X_test[i].reshape(1, 6, 6), state)
        if not fake:
            LON.append(y[0][1].item())
            LATI.append(y[0][2].item())
            HEI.append(y[0][3].item())
        else:
            y = fix_data(Y_test[i].cpu().numpy(), y[0].detach().cpu().numpy(), 400)
            LON.append(y[1].item())
            LATI.append(y[2].item())
            HEI.append(y[3].item())
    predicts['CNN-LSTM-HardAttention预测航迹'] = [
        np.array(LON),
        np.array(LATI),
        np.array(HEI)
    ]

    show_3D_multi(sources, predicts)
