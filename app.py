from typing import Iterator
import torch
from torch import nn, optim
from time import time

from model import PlaneLSTMModule, loss
from data_loader import data_iter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def grad_clipping(params: Iterator[nn.Parameter], theta: float) -> None:
    '''梯度和裁剪为theta'''
    norm = torch.tensor(0.0, device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt()
    if norm.item() > theta:
        for param in params:
            param.grad.data *= theta / norm.item()


if __name__ == '__main__':
    num_hiddens = 144
    num_features = 3
    lr = 0.1
    # clipping_theta = 1e2
    num_epochs = 10
    batch_size = 256
    num_steps = 100

    model = PlaneLSTMModule(num_hiddens, num_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs + 1):
        l_sum, n, start = 0.0, 0, time()
        state = None
        for X, Y in data_iter(batch_size, num_steps):
            X, Y = X.to(device), Y.to(device)
            if state is not None:
                for s in state:
                    s.detach_()
            outputs, state = model(X, state)
            y = Y.reshape(-1, Y.shape[-1])
            # y = torch.transpose(Y, 0, 1).contiguous().view(-1, 3)
            l = loss(outputs, y)

            optimizer.zero_grad()
            l.backward()
            # grad_clipping(model.parameters(), clipping_theta)
            optimizer.step()

            l_sum += l.item()
            n += y.shape[-1]
        print(f'epoch {epoch}, loss {l_sum / n}, time {time() - start}')
