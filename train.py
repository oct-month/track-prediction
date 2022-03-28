import os
from time import time
from mxnet import autograd, init, gpu, cpu
from mxnet.gluon import Trainer
from mxnet.gluon.utils import split_and_load
from mxnet.util import get_gpu_count

from model import HybridCNNLSTM, loss
from data_loader import data_iter_load
from config import PARAMS_PATH, batch_size, num_epochs


gpu_counts = get_gpu_count()
devices = [gpu(i) for i in range(gpu_counts)] if gpu_counts > 0 else [cpu()]


# batch channel sequeu
if __name__ == '__main__':
    model = HybridCNNLSTM()
    if os.path.isfile(PARAMS_PATH):
        model.load_parameters(PARAMS_PATH, ctx=devices)
        print('Warning: Using the existing params to train.')
    else:
        model.initialize(init=init.Normal(sigma=0.5), ctx=devices)
    states = model.begin_state(batch_size, devices)

    # print(model.collect_params())
    optimizer = Trainer(model.collect_params(), 'sgd', {'learning_rate': 1000})

    # 载入训练数据集
    datasets = []
    for X, Y in data_iter_load():
        X_list = split_and_load(X, devices, batch_axis=0, even_split=True)
        Y_list = split_and_load(Y, devices, batch_axis=0, even_split=True)
        datasets.append((X_list, Y_list))
    # 测试集
    datasets.pop()
    X_test = X.copyto(devices[0])
    Y_test = Y.copyto(devices[0])

    # 训练
    for epoch in range(1, num_epochs + 1):
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
