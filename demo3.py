from mxnet import nd, autograd
from mxnet.gluon import Trainer
from model import HybridCNNLSTM, loss

# batch channel sequeu
if __name__ == '__main__':
    model = HybridCNNLSTM()
    model.initialize(2)

    optimizer = Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.001})

    # with autograd.record():
    #     pass
    # autograd.backward()
    # optimizer.step()

    m = model(nd.random_normal(shape=(2, 6, 6)))
    print(m.shape)
