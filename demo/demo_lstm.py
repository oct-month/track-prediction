from typing import Iterator, Optional, Tuple, List, Dict, Generator
import torch
from torch import nn, optim
import zipfile
import time
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data_jay_lyrics() -> Tuple[List[int], Dict[str, int], List[str], int]:
    with zipfile.ZipFile('./test/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(c, i) for i, c in enumerate(idx_to_char)])
    vocab_size = len(idx_to_char)
    corpus_indices = [char_to_idx[c] for c in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

def data_iter_consecutive(corpus_indices_: List[int], batch_size: int, num_steps: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    corpus_indices = torch.tensor(corpus_indices_, dtype=torch.int, device=device)
    data_len = len(corpus_indices_)
    batch_len = data_len // batch_size
    indices = corpus_indices[:batch_size * batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    
    for i in range(epoch_size):
        i *= num_steps
        X = indices[:, i:i + num_steps]
        Y = indices[:, i + 1:i + num_steps + 1]
        yield X, Y

def one_hot(X: torch.Tensor, n_class: int) -> torch.Tensor:
    x = X.view(-1, 1).long()
    res = torch.zeros(x.shape[0], n_class, dtype=torch.float, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res

def to_one_hot(X: torch.Tensor, n_class: int) -> List[torch.Tensor]:
    if (len(X.shape) < 2):
        return [one_hot(X, n_class)]
    else:
        return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

def grad_clipping(params: Iterator[nn.Parameter], theta: float) -> None:
    '''梯度和裁剪为theta'''
    norm = torch.tensor(0.0, device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt()
    if norm.item() > theta:
        for param in params:
            param.grad.data *= theta / norm.item()

def predict_rnn(model: nn.Module, prefix: str, num_chars: int) -> str:
    '''基于prefix预测之后的num_chars个字符'''
    state = None
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor(output[-1], device=device).float()
        Y, state = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y.argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])

class RNNModel(nn.Module):
    def __init__(self, rnn_layer: nn.RNNBase, vocab_size: int):
        super().__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None
    
    def forward(self, inputs: torch.Tensor, state: Optional[torch.Tensor]):
        X = to_one_hot(inputs, self.vocab_size)
        Y, self.state = self.rnn(torch.stack(X), state)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

lr = 1e-2
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = RNNModel(lstm_layer, vocab_size)

num_epochs, num_steps, batch_size, clipping_theta = 160, 35, 32, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']


if __name__ == '__main__':
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(1, num_epochs + 1):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps)
        for X, Y in data_iter:
            if state is not None:
                for s in state:
                    s.detach_()
            output, state = model(X, state)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            grad_clipping(model.parameters(), clipping_theta)
            optimizer.step()

            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        perplexity = math.exp(l_sum / n)
        if epoch % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(model, prefix, pred_len))
