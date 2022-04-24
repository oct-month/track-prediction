from mxnet import nd

from model import predict, HybridCNNLSTM
from config import PARAMS_PATH

model = HybridCNNLSTM()
model.load_parameters(PARAMS_PATH)


r = predict(model, nd.array([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6]
]))

print(r)
