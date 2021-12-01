from typing import Iterator, List, Optional, Sequence
from mxnet import nd, gpu, cpu, Context, autograd
from mxnet.gluon.parameter import ParameterDict
from mxnet.util import get_gpu_count
import matplotlib.pyplot as plt

from data_loader import PlaneData, NUM_FEATURES
from .model import PlaneLSTMModule, loss
from .config import device


def grad_clipping(params: ParameterDict, theta: float) -> None:
    '''梯度和裁剪为theta（可能有问题）'''
    norm = nd.array(0.0, ctx=device)    # TODO
    for param in params.values():
        norm += (param.grad() ** 2).sum()
    norm = norm.sqrt()
    if norm.asscalar() > theta:
        for param in params.values():
            param._grad =  (param.grad() * theta / norm.asscalar()).asnumpy().tolist()


def predict(model: PlaneLSTMModule, basics: Sequence[PlaneData], num_pred: int, device: Context = device) -> List[PlaneData]:
    '''使用model基于basics预测num_pred个航迹点'''
    if autograd.is_training():  # TODO
        raise RuntimeError("存在丢弃")
    state = None
    result = [basics[0]]
    for t in range(num_pred + len(basics) - 1):
        X = nd.array(result[0].to_tuple(), ctx=device, dtype='float32').reshape(1, 1, NUM_FEATURES)
        Y, state = model(X, state)
        if t < len(basics) - 1:
            result.append(basics[t + 1])
        else:
            tap = tuple(Y.reshape(-1).detach().copyto(cpu()).asnumpy().tolist())
            result.append(PlaneData.from_tuple(tap))
    return result


def test_loss(track: Sequence[PlaneData], track_pred: Sequence[PlaneData]) -> float:
    y = nd.array([t.to_tuple() for t in track])
    y_p = nd.array([t.to_tuple() for t in track_pred])
    l = loss(y_p, y)
    return l.sum().asscalar() / NUM_FEATURES / len(track)


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

    # plt.show()
    plt.savefig("draw_3d.png")


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
    plt.savefig("draw_2d.png")
    # plt.show()
