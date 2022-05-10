from typing import Optional, Tuple
import torch
from torch import nn


class HybridCNNLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # CNN层
        self.cnn_net = nn.Sequential(
            nn.Conv1d(
                in_channels=6,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2,
                padding=0,
                ceil_mode=False
            ),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2,
                padding=0,
                ceil_mode=False
            )
        )
        # LSTM层
        self.lstm_net = nn.LSTM(
            input_size=32,
            hidden_size=50,
            bias=True,
            batch_first=True,
            bidirectional=False,
            num_layers=2,
            dropout=0.2
        )
        # 全连接层
        self.linear_net = nn.Linear(
            in_features=50,
            out_features=4,
            bias=True
        )
    
    def forward(self, X: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        Xc = self.cnn_net(X)
        Xl, state = self.lstm_net(Xc.view(-1, 1, Xc.shape[-2]), state)
        return self.linear_net(Xl.view(-1, Xl.shape[-1])), state


loss = nn.MSELoss()
