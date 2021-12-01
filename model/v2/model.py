from typing import List, Optional, Tuple
from mxnet import nd
from mxnet.gluon import nn, rnn, loss as gloss

from .config import device


class PlaneLSTMModule(nn.Block):
    def __init__(self, hidden_size: int, feature_size: int, prefix=None, params=None):
        super().__init__(prefix, params)
        self.lstm = rnn.LSTM(hidden_size, input_size=feature_size, num_layers=1, layout='NTC', dropout=0.2, bidirectional=False)
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.dense = nn.Dense(feature_size, activation=None, use_bias=True)

    def forward(self, X: nd.NDArray, state: Optional[List[nd.NDArray]]):
        if state is not None:
            Y, state = self.lstm(X, state)
        else:
            Y = self.lstm(X, state)
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, batch_size: int) -> List[nd.NDArray]:
        return [st.copyto(device) for st in self.lstm.begin_state(batch_size)]


loss = gloss.L2Loss()
