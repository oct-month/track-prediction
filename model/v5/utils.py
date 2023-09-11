from typing import Tuple
import torch

from .model import HybridCNNLSTM


# 时间,经度,纬度,速度,高度,航向
def predict(model: HybridCNNLSTM, basics: torch.Tensor) -> Tuple[float, float, float]:
    '''使用model基于basics预测'''
    state = None
    # 预测
    X = basics.reshape(1, 6, 6)
    y, state = model(X, state)
    lon = y[0][1].item()
    lati = y[0][2].item()
    hei = y[0][3].item()
    return lon, lati, hei
