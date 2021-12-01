from model import PlaneLSTMModule, device, predict, draw_2d, draw_3d, test_loss
from data_loader import data_track_iter, NUM_FEATURES
from config import PARAMS_PATH, num_hiddens


if __name__ == '__main__':
    model = PlaneLSTMModule(num_hiddens, NUM_FEATURES)
    # model.initialize(ctx=device)
    model.load_parameters(PARAMS_PATH, ctx=device)

    idx = 3
    for i, track in enumerate(data_track_iter()):
        if i < idx:
            continue
        track_pred = predict(model, track[:100], len(track) - 100, device)
        draw_2d(track[::10], track_pred[100::10])
        print('test loss', test_loss(track, track_pred))
        print(len(track_pred))
        break
