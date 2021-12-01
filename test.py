from mxnet import nd, gpu
from mxnet.gluon.parameter import Parameter
import numpy as np

from data_loader import data_iter
from model import PlaneLSTMModule


if __name__ == '__main__':
    # for X, Y in data_iter(batch_size=256, num_steps=5):
    #     X, Y = X.copyto(gpu()), Y.copyto(gpu())
    #     print(X.shape, Y.shape)
    #     break
    model = PlaneLSTMModule(144, 6)
    model.initialize()
   
    for k, v in model.collect_params().items():
        print(type(k), type(v))
        print(k, v)
        print("++++++++")
        print(v.data()[0, 0])
        print("++++++++")
        print(type(v.data()))
        v._data[0][0] = 90
        print(v.data()[0, 0])
        break
