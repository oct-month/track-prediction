import os
from mxnet import autograd
from mxnet.gluon.trainer import Trainer
from mxnet.gluon.utils import split_and_load
from time import time

from model import PlaneLSTMModule, loss, grad_clipping, predict, draw_2d, draw_3d, test_loss, devices, gpu_counts
from data_loader import data_track_iter, data_iter, NUM_FEATURES
from config import PARAMS_PATH, num_hiddens, lr, clipping_theta, num_epochs, batch_size, num_steps


if __name__ == '__main__':
    model = PlaneLSTMModule(num_hiddens, NUM_FEATURES)
    if os.path.isfile(PARAMS_PATH):
        model.load_parameters(PARAMS_PATH, ctx=devices)
    else:
        model.initialize(ctx=devices)

    optimizer = Trainer(model.collect_params(), 'sgd', {'learning_rate': lr})

    state = model.begin_state(batch_size // gpu_counts)
    loss_list = []
    for epoch in range(1, num_epochs + 1):
        l_sum, n, start = 0.0, 0, time()
        for Xs, Ys in data_iter(batch_size, num_steps):
            X_list = split_and_load(Xs, devices, batch_axis=0, even_split=True)
            Y_list = split_and_load(Ys, devices, batch_axis=0, even_split=True)
            # X, Y = X.copyto(device), Y.copyto(device)
            with autograd.record():
                losses = [loss(model(X, state)[0], Y.reshape(-1, Y.shape[-1])) for X, Y in zip(X_list, Y_list)]
                # outputs, _ = model(X, state)
                # y = Y.reshape(-1, Y.shape[-1])
                # l = loss(outputs, y)
            autograd.backward(losses)
            # for l in losses:
            #     l.backward()
            # grad_clipping(model.collect_params(), clipping_theta)
            optimizer.step(batch_size)
            l_sum += sum([l.sum().asscalar() for l in losses])
            # l_sum += l.asscalar() / NUM_FEATURES
            n += Ys.shape[0]
        loss_list.append(l_sum / n)
        print(f'epoch {epoch}, loss {loss_list[-1]}, time {time() - start}, n {n}.')

    model.save_parameters(PARAMS_PATH)

    for track in data_track_iter():
        track_pred = predict(model, track[:100], len(track) - 100)
        draw_2d(track[::10], track_pred[100::10], loss_list)
        print('test loss', test_loss(track, track_pred))
        print(len(track_pred))
        # print(track[-1].to_tuple())
        # print([t.to_tuple() for t in track_pred[-10:]])
        break
