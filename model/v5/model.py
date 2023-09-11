from typing import Optional, Tuple
import torch
from torch import nn


class HybridCNNLSTM(nn.Module):
    def __init__(self, lstm_num: int=1, lstm_hidden: int=8, soft_attention: bool=True) -> None:
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
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2,
                padding=0,
                ceil_mode=False
            )
        )
        # LSTM层
        self.lstm_hidden_size = lstm_hidden
        self.soft_attention_flag = soft_attention
        self.lstm_net = nn.LSTM(
            input_size=3,
            hidden_size=self.lstm_hidden_size,
            bias=True,
            batch_first=True,
            bidirectional=False,
            num_layers=lstm_num,
            dropout=0.2
        )
        # 全连接层
        self.linear_net = nn.Linear(
            in_features=self.lstm_hidden_size,
            out_features=4,
            bias=True
        )
    
    def forward(self, X: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        # CNN
        Xc = self.cnn_net(X)
        # LSTM
        Xl, state = self.lstm_net(Xc, state)
        if self.soft_attention_flag:
            # Soft-Attention 机制
            score = (Xl * state[0].sum(0, keepdim=True).transpose(0, 1)).sum(2).softmax(1)
            Xl_p = (Xl * score.unsqueeze(2)).sum(1)
        else:
            # Hard-Attention 机制
            score = (Xl * state[0].sum(0, keepdim=True).transpose(0, 1)).sum(2)
            indices = score.argmax(1)
            Xl_p = torch.gather(Xl, 1, indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, Xl.size(2)))
        # 全连接层
        return self.linear_net(Xl_p), state


loss = nn.MSELoss()
