import os

from model import device, PlaneLSTMModule, predict, draw_2d, draw_3d
from data_loader import data_track_iter, NUM_FEATURES
from config import PARAMS_PATH, num_hiddens


if __name__ == '__main__':
    model = PlaneLSTMModule(num_hiddens, NUM_FEATURES)
    if os.path.isfile(PARAMS_PATH):
        model.load_parameters(PARAMS_PATH, ctx=device)
    else:
        model.initialize(ctx=device)

    steps = 1
    for i, track in enumerate(data_track_iter()):
        if i == steps:
            track_pred = predict(model, track[:1000], len(track) - 1000)
            draw_2d(track[:1000], track_pred, [100, 50, 1])
            print(len(track_pred))
            # print(track[-1].to_tuple())
            # print([t.to_tuple() for t in track_pred[-120:]])
            break
