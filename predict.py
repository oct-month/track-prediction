from mxnet import gpu, cpu
from mxnet.util import get_gpu_count
import numpy as np
import matplotlib.pyplot as plt

from model import HybridCNNLSTM
from data_loader import data_iter_order
from config import LABEL_NORMALIZATION, NORMALIZATION_TIMES, PARAMS_PATH, batch_size


plt.rcParams['font.sans-serif']=['SimHei']      #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False        #用来正常显示负号

gpu_counts = get_gpu_count()
devices = [gpu(i) for i in range(gpu_counts)] if gpu_counts > 0 else [cpu()]


def show_3D(sources, predicts, steps=0):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('lontitude')
    ax.set_ylabel('latitude')
    ax.set_zlabel('height')
    ax.scatter3D(predicts[0][steps:], predicts[1][steps:], predicts[2][steps:], c='blue')
    ax.scatter3D(sources[0][steps:], sources[1][steps:], sources[2][steps:], c='red')
    plt.show()


def show_2D(sources, predicts, steps=0):
    plt.figure()
    ax = plt.axes()
    ax.set_xlabel('lontitude')
    ax.set_ylabel('latitude')
    plt.plot(predicts[0][steps:], predicts[1][steps:], c='blue', label='预测')
    plt.scatter(predicts[0][-1:], predicts[1][-1:], c='black')
    plt.plot(sources[0][steps:], sources[1][steps:], c='red', label='真实')
    plt.scatter(sources[0][-1:], sources[1][-1:], c='black')
    plt.legend()
    plt.show()


def save_2D(sources, predicts, p, steps=0):
    plt.figure()
    ax = plt.axes()
    ax.set_xlabel('lontitude')
    ax.set_ylabel('latitude')
    plt.scatter(predicts[0][steps:], predicts[1][steps:], c='blue', label='预测')
    plt.scatter(sources[0][steps:], sources[1][steps:], c='red', label='真实')
    plt.legend()
    plt.savefig(p)


# batch channel sequeu
if __name__ == '__main__':
    model = HybridCNNLSTM()
    model.load_parameters(PARAMS_PATH, ctx=devices)

    # 载入数据集
    num_times = 3
    for X, Y in data_iter_order(batch_size):
        num_times -= 1
        X_test = X.copyto(devices[0])
        Y_test = Y.copyto(devices[0])
        if num_times == 0:
            break

    # predict
    LON, LATI, HEI = [], [], []
    state = model.begin_state(1, devices[:1])[0]
    for i in range(X_test.shape[0]):
        y, state = model(X_test[i].reshape(1, 6, 6), state)
        # TI.append(y[0][0].asscalar())
        LON.append(y[0][1].asscalar() * (LABEL_NORMALIZATION[1][1] - LABEL_NORMALIZATION[1][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[1][0])
        LATI.append(y[0][2].asscalar() * (LABEL_NORMALIZATION[2][1] - LABEL_NORMALIZATION[2][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[2][0])
        HEI.append(y[0][3].asscalar() * (LABEL_NORMALIZATION[3][1] - LABEL_NORMALIZATION[3][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[3][0])

    predicts = [
        np.array(LON),
        np.array(LATI),
        np.array(HEI)
    ]
    sources = [
        Y_test[:, 1].asnumpy() * (LABEL_NORMALIZATION[1][1] - LABEL_NORMALIZATION[1][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[1][0],
        Y_test[:, 2].asnumpy() * (LABEL_NORMALIZATION[2][1] - LABEL_NORMALIZATION[2][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[2][0],
        Y_test[:, 3].asnumpy() * (LABEL_NORMALIZATION[3][1] - LABEL_NORMALIZATION[3][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[3][0]
    ]

    # c = len(sources[0]) // 3
    # b1 = sources[0][c] - predicts[0][c]
    # b2 = sources[1][c] - predicts[1][c]
    # b3 = sources[2][c] - predicts[2][c]
    # predicts[0] += b1
    # predicts[1] += b2
    # predicts[2] += b3

    show_2D(sources, predicts, 100)
