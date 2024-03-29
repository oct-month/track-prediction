from typing import Tuple
import torch

from config import FEATURES_NORMALIZATION, NORMALIZATION_TIMES, LABEL_NORMALIZATION, NORMALIZATION_TIMES

from .model import HybridCNNLSTM


# 时间,经度,纬度,速度,高度,航向
def predict(model: HybridCNNLSTM, basics: torch.Tensor) -> Tuple[float, float, float]:
    '''使用model基于basics预测'''
    state = None
    # 归一化
    basics[:, 0] = NORMALIZATION_TIMES * (basics[:, 0] - FEATURES_NORMALIZATION[0][0]) / (FEATURES_NORMALIZATION[0][1] - FEATURES_NORMALIZATION[0][0])
    basics[:, 1] = NORMALIZATION_TIMES * (basics[:, 1] - FEATURES_NORMALIZATION[1][0]) / (FEATURES_NORMALIZATION[1][1] - FEATURES_NORMALIZATION[1][0])
    basics[:, 2] = NORMALIZATION_TIMES * (basics[:, 2] - FEATURES_NORMALIZATION[2][0]) / (FEATURES_NORMALIZATION[2][1] - FEATURES_NORMALIZATION[2][0])
    basics[:, 3] = NORMALIZATION_TIMES * (basics[:, 3] - FEATURES_NORMALIZATION[3][0]) / (FEATURES_NORMALIZATION[3][1] - FEATURES_NORMALIZATION[3][0])
    basics[:, 4] = NORMALIZATION_TIMES * (basics[:, 4] - FEATURES_NORMALIZATION[4][0]) / (FEATURES_NORMALIZATION[4][1] - FEATURES_NORMALIZATION[4][0])
    basics[:, 5] = NORMALIZATION_TIMES * (basics[:, 5] - FEATURES_NORMALIZATION[5][0]) / (FEATURES_NORMALIZATION[5][1] - FEATURES_NORMALIZATION[5][0])
    # 预测
    X = basics.reshape(1, 6, 6)
    y, state = model(X, state)
    lon = y[0][1].item() * (LABEL_NORMALIZATION[1][1] - LABEL_NORMALIZATION[1][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[1][0]
    lati = y[0][2].item() * (LABEL_NORMALIZATION[2][1] - LABEL_NORMALIZATION[2][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[2][0]
    hei = y[0][3].item() * (LABEL_NORMALIZATION[3][1] - LABEL_NORMALIZATION[3][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[3][0]
    return lon, lati, hei
