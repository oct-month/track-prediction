from mxnet import nd, gpu, cpu
from mxnet.util import get_gpu_count
import matplotlib.pyplot as plt

from model import HybridCNNLSTM
from data_loader import data_iter


LABEL_COLUMNS = ['时间', '经度', '纬度', '高度']
LABEL_NORMALIZATION = [
    [1000000000, 4000000000],
    [-180, 180],
    [-90, 90],
    [-10000, 20000],
]
LABEL_NORMALIZATION_TIMES = 100

gpu_counts = get_gpu_count()
devices = [gpu(i) for i in range(gpu_counts)] if gpu_counts > 0 else [cpu()]

batch_size = 1200
num_epochs = 10
PARAMS_PATH = './params-hybrid.pt'


def show_3D(sources, predicts):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('lontitude')
    ax.set_ylabel('latitude')
    ax.set_zlabel('height')
    ax.scatter3D(predicts[0], predicts[1], predicts[2], c='blue')
    ax.scatter3D(sources[0], sources[1], sources[2], c='red')
    plt.show()


def show_2D(sources, predicts):
    plt.figure()
    ax = plt.axes()
    ax.set_xlabel('lontitude')
    ax.set_ylabel('latitude')
    plt.scatter(predicts[0], predicts[1], c='blue')
    plt.scatter(sources[0], sources[1], c='red')
    plt.show()


# batch channel sequeu
if __name__ == '__main__':
    model = HybridCNNLSTM()
    model.load_parameters(PARAMS_PATH, ctx=devices)

    # 载入数据集
    for X, Y in data_iter(batch_size):
        X_test: nd.NDArray = X.copyto(devices[0])
        Y_test: nd.NDArray = Y.copyto(devices[0])
        break

    # predict
    LON, LATI, HEI = [], [], []
    state = model.begin_state(1, devices[:1])[0]
    for i in range(X_test.shape[0]):
        y, state = model(X_test[i].reshape(1, 6, 6), state)
        # TI.append(y[0][0].asscalar())
        LON.append(y[0][1].asscalar() * (LABEL_NORMALIZATION[1][1] - LABEL_NORMALIZATION[1][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[1][0])
        LATI.append(y[0][2].asscalar() * (LABEL_NORMALIZATION[2][1] - LABEL_NORMALIZATION[2][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[2][0])
        HEI.append(y[0][3].asscalar() * (LABEL_NORMALIZATION[3][1] - LABEL_NORMALIZATION[3][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[3][0])
    
    predicts = [LON, LATI, HEI]
    sources = [
        Y_test[:, 1].asnumpy() * (LABEL_NORMALIZATION[1][1] - LABEL_NORMALIZATION[1][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[1][0],
        Y_test[:, 2].asnumpy() * (LABEL_NORMALIZATION[2][1] - LABEL_NORMALIZATION[2][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[2][0],
        Y_test[:, 3].asnumpy() * (LABEL_NORMALIZATION[3][1] - LABEL_NORMALIZATION[3][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[3][0],
    ]

    show_2D(sources, predicts)
