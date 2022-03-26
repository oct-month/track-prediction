from time import time
from mxnet import autograd, gpu, cpu
from mxnet.gluon import Trainer
from mxnet.gluon.utils import split_and_load
from mxnet.util import get_gpu_count

from model import HybridCNNLSTM, loss

from data_loader import data_iter

gpu_counts = get_gpu_count()
devices = [gpu(i) for i in range(gpu_counts)] if gpu_counts > 0 else [cpu()]

batch_size = 100
num_epochs = 10
PARAMS_PATH = './params-hybrid.pt'

# batch channel sequeu
if __name__ == '__main__':
    model = HybridCNNLSTM()
    model.initialize(batch_size, ctx=devices)

    optimizer = Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.001})

    for epoch in range(1, num_epochs):
        l_sum, n, start = 0.0, 0, time()
        for X, Y in data_iter(batch_size):
            X_list = split_and_load(X, devices, batch_axis=0, even_split=True)
            Y_list = split_and_load(Y, devices, batch_axis=0, even_split=True)
            with autograd.record():
                losses = [loss(model(Xs), Ys) for Xs, Ys in zip(X_list, Y_list)]
                # l = loss(model(X), Y).sum()
            autograd.backward(losses)
            optimizer.step(batch_size)
            l_sum += sum([l.sum().asscalar() for l in losses])
            n += Y.shape[0]
        print(f'epoch {epoch}, loss {l_sum / n}, time {time() - start}, n {n}.')
    
    model.save_parameters()
