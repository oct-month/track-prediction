from time import time
from mxnet import autograd, gpu, cpu
from mxnet.gluon import Trainer
from mxnet.gluon.utils import split_and_load
from mxnet.util import get_gpu_count
import matplotlib.pyplot as plt

from model import HybridCNNLSTM, loss
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
    states = model.begin_state(batch_size, devices)

    optimizer = Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.0001})

    # 载入训练数据集
    datasets = []
    for X, Y in data_iter(batch_size):
        X_list = split_and_load(X, devices, batch_axis=0, even_split=True)
        Y_list = split_and_load(Y, devices, batch_axis=0, even_split=True)
        datasets.append((X_list, Y_list))
    # 测试集
    datasets.pop()
    X_test = X.copyto(devices[0])
    Y_test = Y.copyto(devices[0])

    # 训练
    for epoch in range(1, num_epochs):
        print(f'epoch {epoch}, ', end='')
        l_sum, n, start = 0.0, 0, time()

        for X_list, Y_list in datasets:
            losses = []
            with autograd.record():
                for i, (Xs, Ys) in enumerate(zip(X_list, Y_list)):
                    y, states[i] = model(Xs, states[i])
                    l = loss(y, Ys)
                    losses.append(l)
            autograd.backward(losses)
            optimizer.step(batch_size)
            l_sum += sum([l.sum().asscalar() for l in losses])
            n += batch_size
        # 测试
        state_test = model.begin_state(batch_size, devices[:1])[0]
        y, state_test = model(X_test, state_test)
        l_test = loss(y, Y_test).sum().asscalar() / batch_size
        # 输出
        print(f'loss {l_sum / n}, time {time() - start}, n {n}, test loss {l_test}.')
    model.save_parameters(PARAMS_PATH)

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
    plt.savefig('predict.png')
