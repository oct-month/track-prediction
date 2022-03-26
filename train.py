from time import time
from mxnet import autograd, gpu, cpu
from mxnet.gluon import Trainer
from mxnet.gluon.utils import split_and_load
from mxnet.util import get_gpu_count

from model import HybridCNNLSTM, loss

from data_loader import data_iter

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

    optimizer = Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.001})

    # 载入数据集
    datasets = []
    for X, Y in data_iter(batch_size):
        X_list = split_and_load(X, devices, batch_axis=0, even_split=True)
        Y_list = split_and_load(Y, devices, batch_axis=0, even_split=True)
        datasets.append((X_list, Y_list))

    for epoch in range(1, num_epochs):
        print(f'epoch {epoch}, ', end='')
        l_sum, n, start = 0.0, 0, time()

        # 训练
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
            n += Y.shape[0]
        print(f'loss {l_sum / n}, time {time() - start}, n {n}.')
    
    model.save_parameters(PARAMS_PATH)
