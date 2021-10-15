import torch
from torch import nn, optim
from time import time

from model import device, PlaneLSTMModule, loss, grad_clipping, predict, draw_2d, draw_3d
from data_loader import data_track_iter, data_iter, NUM_FEATURES


PARAMS_PATH = './params.pt'

if __name__ == '__main__':
    num_hiddens = 144
    lr = 1
    clipping_theta = 1e-2
    num_epochs = 128
    batch_size = 32
    num_steps = 64

    model = PlaneLSTMModule(num_hiddens, NUM_FEATURES).to(device)
    # if os.path.isfile(PARAMS_PATH):
    #      model.load_state_dict(torch.load(PARAMS_PATH))
    # else:
    #     for param in model.parameters():
    #         nn.init.normal_(param)
    for param in model.parameters():
        nn.init.normal_(param)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        l_sum, n, start = 0.0, 0, time()
        state = None
        for X, Y in data_iter(batch_size, num_steps):
            X, Y = X.to(device), Y.to(device)
            if state is not None:
                for s in state:
                    s.detach_()
            outputs, state = model(X, state)
            y = Y.reshape(-1, Y.shape[-1])
            l = loss(outputs, y)

            optimizer.zero_grad()
            l.sum().backward()
            if epoch > 8:
                grad_clipping(model.parameters(), clipping_theta)
            optimizer.step()

            l_sum += l.sum().item() / NUM_FEATURES
            n += y.shape[0]
        print(f'epoch {epoch}, loss {l_sum / n}, time {time() - start}, n {n}.')

    torch.save(model.state_dict(), PARAMS_PATH)

    for track in data_track_iter():
        track_pred = predict(model, track[:1000], len(track))
        draw_2d(track, track_pred)
        # print(len(track_pred))
        # print(track[-1].to_tuple())
        # print([t.to_tuple() for t in track_pred[-120:]])
        break
