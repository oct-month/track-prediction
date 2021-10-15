from typing import Optional, Tuple
import torch
from torch import nn

class PlaneLSTMModule(nn.Module):
    def __init__(self, hidden_size: int, feature_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size)
        self.hidden_size = self.lstm.hidden_size * (2 if self.lstm.bidirectional else 1)
        self.feature_size = feature_size
        self.dense = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(in_features=self.hidden_size, out_features=feature_size, bias=True)
            )

    def forward(self, X: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        Y, state = self.lstm(X, state)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, state

# loss = nn.MSELoss()


def loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    l = (((y_hat - y) ** 2).sum(dim=0) / y_hat.shape[0]).sqrt()
    return l
