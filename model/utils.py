from typing import Iterator, List, Sequence
import torch
from torch import nn
import matplotlib.pyplot as plt

from model import PlaneLSTMModule
from data_loader import PlaneDataSimple

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


def predict(model: PlaneLSTMModule, basics: Sequence[PlaneDataSimple], num_pred: int) -> List[PlaneDataSimple]:
    '''使用model基于basics预测num_pred个航迹点'''
    model.eval()
    result = []
    state = None
    for pd in basics:
        X = torch.tensor(pd.to_tuple(), device=device)
        X = X.view(1, 1, X.shape[-1])
        Y, state = model(X, state)
    t = tuple(Y.view(-1).detach().cpu().numpy().tolist())
    result.append(PlaneDataSimple.from_tuple(t))
    for _ in range(num_pred):
        X = torch.tensor(result[-1].to_tuple(), device=device)
        X = X.view(1, 1, X.shape[-1])
        Y, state = model(X, state)
        t = tuple(Y.view(-1).detach().cpu().numpy().tolist())
        result.append(PlaneDataSimple.from_tuple(t))
    model.train()
    return result


def draw_3d(track: Sequence[PlaneDataSimple], track_pred: Sequence[PlaneDataSimple]) -> None:
    ax = plt.axes(projection='3d')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_zlabel('height')

    X = [dt.longitude for dt in track]
    Y = [dt.latitude for dt in track]
    Z = [dt.height for dt in track]
    ax.scatter3D(X, Y, Z, c='blue')

    X_p = [dt.longitude for dt in track_pred]
    Y_p = [dt.latitude for dt in track_pred]
    Z_p = [dt.height for dt in track_pred]
    ax.scatter3D(X_p, Y_p, Z_p, c='red')

    plt.show()


def draw_2d(track: Sequence[PlaneDataSimple], track_pred: Sequence[PlaneDataSimple]) -> None:
    X = [dt.longitude for dt in track]
    Y = [dt.latitude for dt in track]
    plt.scatter(X, Y, c='blue', s=0.5)

    X_p = [dt.longitude for dt in track_pred]
    Y_p = [dt.latitude for dt in track_pred]
    plt.scatter(X_p, Y_p, c='red', s=0.5)

    plt.show()
