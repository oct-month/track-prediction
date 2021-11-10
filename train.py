import os
import torch
from torch import nn, optim
from time import time

from model import device, PlaneLSTMModule, loss, grad_clipping, predict, draw_2d, draw_3d, test_loss
from data_loader import data_track_iter, data_iter, NUM_FEATURES
from config import PARAMS_PATH, num_hiddens, lr, clipping_theta, num_epochs, batch_size, num_steps


if __name__ == '__main__':
    model = PlaneLSTMModule(num_hiddens, NUM_FEATURES).cpu()
    if os.path.isfile(PARAMS_PATH):
        model.load_state_dict(torch.load(PARAMS_PATH))
    else:
        for param in model.parameters():
            nn.init.normal_(param, mean=0, std=1)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_list = []
    for epoch in range(1, num_epochs + 1):
        l_sum, n, start = 0.0, 0, time()
        for X, Y in data_iter(batch_size, num_steps):
            X, Y = X.to(device), Y.to(device)
            outputs, state = model(X, None)
            y = Y.reshape(-1, Y.shape[-1])
            l = loss(outputs, y).sum()

            optimizer.zero_grad()
            l.backward()
            # grad_clipping(model.parameters(), clipping_theta)
            optimizer.step()

            l_sum += l.item() / NUM_FEATURES
            n += y.shape[0]
        loss_list.append(l_sum / n)
        print(f'epoch {epoch}, loss {loss_list[-1]}, time {time() - start}, n {n}.')

    torch.save(model.cpu().state_dict(), PARAMS_PATH)
    model = model.to(device)

    for track in data_track_iter():
        track_pred = predict(model, track[:1000], len(track) - 1000)
        draw_2d(track[::100], track_pred[1000::10], loss_list)
        print('test loss', test_loss(track, track_pred))
        print(len(track_pred))
        # print(track[-1].to_tuple())
        # print([t.to_tuple() for t in track_pred[-10:]])
        break
