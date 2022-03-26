from mxnet import Context
from mxnet.gluon import nn, rnn, loss as gloss

class HybridCNNLSTM(nn.Block):
    def __init__(self, prefix=None, params=None):
        super().__init__(prefix, params)
        # CNN层
        self.cnn_net = nn.Sequential()
        with self.cnn_net.name_scope():
            self.cnn_net.add(nn.Conv1D(
                channels=32,
                kernel_size=3,
                strides=1,
                padding=1,
                layout='NCW',
                in_channels=6,
                activation='relu',
                use_bias=True
            ))
            self.cnn_net.add(nn.MaxPool1D(
                pool_size=2,
                strides=2,
                padding=0,
                layout='NCW',
                ceil_mode=False
            ))
            self.cnn_net.add(nn.Conv1D(
                channels=32,
                kernel_size=3,
                strides=1,
                padding=1,
                layout='NCW',
                in_channels=32,
                activation='relu',
                use_bias=True
            ))
            self.cnn_net.add(nn.MaxPool1D(
                pool_size=2,
                strides=2,
                padding=0,
                layout='NCW',
                ceil_mode=False
            ))
        # LSTM层
        self.lstm_net = rnn.SequentialRNNCell()
        with self.lstm_net.name_scope():
            self.lstm_net.add(rnn.LSTM(
                hidden_size=50,
                num_layers=2,
                layout='NTC',
                dropout=0.2,
                bidirectional=False,
                state_clip_nan=True,
                input_size=32
            ))
            self.lstm_net.add(rnn.DropoutCell(
                rate=0.2
            ))
        # Dense层
        self.dense_net = nn.Dense(
            units=4,
            use_bias=True,
            flatten=True,
            in_units=50
        )
    
    def forward(self, X, state):
        Xc = self.cnn_net(X)
        Xc = Xc.reshape(0, -1, Xc.shape[-2])
        Xl, state = self.lstm_net(Xc, state)
        return self.dense_net(Xl), state

    def begin_state(self, batch_size, devices):
        gpu_counts = len(devices)
        states = []
        for i in range(gpu_counts):
            t = self.lstm_net.begin_state(batch_size=batch_size // gpu_counts)
            for j, ts in enumerate(t):
                t[j] = ts.copyto(devices[i])
            states.append(t)
        return states


loss = gloss.L2Loss()
