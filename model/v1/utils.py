from typing import Iterator, List, Optional, Sequence
import torch
from torch import nn
import matplotlib.pyplot as plt

from .model import PlaneLSTMModule, loss
from data_loader import PlaneData, NUM_FEATURES

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


def predict(model: PlaneLSTMModule, basics: Sequence[PlaneData], num_pred: int, device: torch.device = device) -> List[PlaneData]:
    '''使用model基于basics预测num_pred个航迹点'''
    model.eval()
    state = None
    result = [basics[0]]
    for t in range(num_pred + len(basics) - 1):
        X = torch.tensor(result[0].to_tuple(), device=device).float().view(1, 1, NUM_FEATURES)
        Y, state = model(X, state)
        if t < len(basics) - 1:
            result.append(basics[t + 1])
        else:
            tap = tuple(Y.view(-1).detach().cpu().numpy().tolist())
            result.append(PlaneData.from_tuple(tap))
    model.train()
    return result


def test_loss(track: Sequence[PlaneData], track_pred: Sequence[PlaneData]) -> float:
    y = torch.tensor([t.to_tuple() for t in track])
    y_p = torch.tensor([t.to_tuple() for t in track_pred])
    l = loss(y_p, y)
    return l.sum().item() / NUM_FEATURES / len(track)


def draw_3d(track: Sequence[PlaneData], track_pred: Sequence[PlaneData]) -> None:
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


def draw_2d(track: Sequence[PlaneData], track_pred: Sequence[PlaneData], loss_list: Optional[Sequence[float]]=None) -> None:
    if loss_list is not None:
        plt.subplot(1, 2, 2)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        X_l = range(1, len(loss_list) + 1)
        Y_l = loss_list
        plt.plot(X_l, Y_l, label='loss')

        plt.subplot(1, 2, 1)

    plt.xlabel('longitude')
    plt.ylabel('latitude')
    X = [dt.longitude for dt in track]
    Y = [dt.latitude for dt in track]
    plt.scatter(X, Y, c='blue', s=0.5, label='base')
    X_p = [dt.longitude for dt in track_pred]
    Y_p = [dt.latitude for dt in track_pred]
    plt.scatter(X_p, Y_p, c='red', s=0.5, label='predict')

    plt.legend()
    plt.show()
