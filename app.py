import os
import torch
from torch import nn

from model import PlaneLSTMModule, predict, draw_2d, draw_3d, test_loss
from data_loader import data_track_iter, NUM_FEATURES
from config import PARAMS_PATH, num_hiddens


device = torch.device('cpu')

if __name__ == '__main__':
    model = PlaneLSTMModule(num_hiddens, NUM_FEATURES).to(device)
    model.load_state_dict(torch.load(PARAMS_PATH))

    idx = 3
    for i, track in enumerate(data_track_iter()):
        if i < idx:
            continue
        track_pred = predict(model, track[:1000], len(track) - 1000, device)
        draw_2d(track[::100], track_pred[1000::10])
        print('test loss', test_loss(track, track_pred))
        print(len(track_pred))
        break
