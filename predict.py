from mxnet import nd, gpu, cpu
from mxnet.gluon.utils import split_and_load
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


# batch channel sequeu
if __name__ == '__main__':
    model = HybridCNNLSTM()
    model.initialize(ctx=devices)

    # 载入训练数据集
    datasets = []
    for X, Y in data_iter(batch_size):
        X_list = split_and_load(X, devices, batch_axis=0, even_split=True)
        Y_list = split_and_load(Y, devices, batch_axis=0, even_split=True)
        datasets.append((X_list, Y_list))
        break
    # 测试集
    datasets.pop()
    X_test: nd.NDArray = X.copyto(devices[0])
    Y_test: nd.NDArray = Y.copyto(devices[0])

    # predict
    LON, LATI, HEI = [], [], []
    state = model.begin_state(1, devices[:1])[0]
    for i in range(X_test.shape[0]):
        y, state = model(X_test[i].reshape(1, 6, 6), state)
        # TI.append(y[0][0].asscalar())
        LON.append(y[0][1].asscalar() * (LABEL_NORMALIZATION[1][1] - LABEL_NORMALIZATION[1][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[1][0])
        LATI.append(y[0][2].asscalar() * (LABEL_NORMALIZATION[2][1] - LABEL_NORMALIZATION[2][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[2][0])
        HEI.append(y[0][3].asscalar() * (LABEL_NORMALIZATION[3][1] - LABEL_NORMALIZATION[3][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[3][0])
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('lontitude')
    ax.set_ylabel('latitude')
    ax.set_zlabel('height')
    ax.scatter3D(LON, LATI, HEI, c='blue')
    ax.scatter3D(
        Y_test[:, 0].asnumpy() * (LABEL_NORMALIZATION[1][1] - LABEL_NORMALIZATION[1][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[1][0],
        Y_test[:, 1].asnumpy() * (LABEL_NORMALIZATION[2][1] - LABEL_NORMALIZATION[2][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[2][0],
        Y_test[:, 2].asnumpy() * (LABEL_NORMALIZATION[3][1] - LABEL_NORMALIZATION[3][0]) / LABEL_NORMALIZATION_TIMES + LABEL_NORMALIZATION[3][0]
    )
    plt.show()
