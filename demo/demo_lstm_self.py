from typing import Generator, Optional, Tuple, List, Dict, Iterable
import time
import math
import random
import numpy as np
import torch
from torch import nn, optim
import zipfile

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

def data_iter_random(corpus_indices: List[int], batch_size: int, num_steps: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    for i in range(epoch_size):
        i *= batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [corpus_indices[j * num_steps:(j + 1) * num_steps] for j in batch_indices]
        Y = [corpus_indices[j * num_steps + 1:(j + 1) * num_steps + 1] for j in batch_indices]
        yield torch.tensor(X, dtype=torch.int, device=device), torch.tensor(Y, dtype=torch.int, device=device)

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

corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

def get_params() -> nn.ParameterList:
    def _tree():
        return nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens)), device=device, dtype=torch.float), requires_grad=True), \
            nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens, num_hiddens)), device=device, dtype=torch.float), requires_grad=True),   \
            nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float), requires_grad=True)
    W_xf, W_hf, b_f = _tree()   # 遗忘门
    W_xi, W_hi, b_i = _tree()   # 输入门
    W_xc, W_hc, b_c = _tree()   # 候选记忆细胞
    W_xo, W_ho, b_o = _tree()   # 输出门
    # 输出层
    W_hq = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens, num_outputs)), device=device, dtype=torch.float), requires_grad=True)
    b_q = nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float), requires_grad=True)

    return nn.ParameterList([W_xf, W_hf, b_f, W_xi, W_hi, b_i, W_xc, W_hc, b_c, W_xo, W_ho, b_o, W_hq, b_q])


def init_lstm_state(batch_size: int):
    return torch.zeros(batch_size, num_hiddens, device=device), torch.zeros(batch_size, num_hiddens, device=device)


def lstm(inputs: List[torch.Tensor], state: Tuple[torch.Tensor, torch.Tensor], params: nn.ParameterList) -> Tuple[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    W_xf, W_hf, b_f, W_xi, W_hi, b_i, W_xc, W_hc, b_c, W_xo, W_ho, b_o, W_hq, b_q = params
    H, C = state
    outputs = []
    for X in inputs:
        F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
        I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
        C_tilda = torch.tanh(X @ W_xc + H @ W_hc + b_c)
        O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
        C = C * F + I * C_tilda
        H = O * C.tanh()
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return outputs, (H, C)

def grad_clipping(params: nn.ParameterList, theta: float) -> None:
    '''梯度和裁剪为theta'''
    norm = torch.tensor(0.0, device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt()
    if norm.item() > theta:
        for param in params:
            param.grad.data *= theta / norm.item()

def predict_rnn(prefix: str, num_chars: int, params: nn.ParameterList) -> str:
    '''基于prefix预测之后的num_chars个字符'''
    state = init_lstm_state(1)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_one_hot(torch.tensor(output[-1], device=device).view(1, 1), vocab_size)
        Y, state = lstm(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y[0].argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])

num_epochs, num_steps, batch_size, lr, clipping_theta = 240, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']


if __name__ == '__main__':
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        state = init_lstm_state(batch_size)
        l_sum, n, start = 0.0, 0, time.time()

        for X, Y in data_iter_consecutive(corpus_indices, batch_size, num_steps):
            for s in state:
                s.detach_()
            inputs = to_one_hot(X, vocab_size)
            outputs_, state = lstm(inputs, state, params)
            outputs = torch.cat(outputs_, dim=0)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(outputs, y.long())

            if params[0].grad is not None:
                for param in params:
                    param.grad.zero_()
            
            l.backward()
            grad_clipping(params, clipping_theta)

            for param in params:
                param.data -= lr * param.grad
            
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
    
        if epoch % pred_period == 0:
            print(f'epoch {epoch}, perplexity {math.exp(l_sum / n)}, time {time.time() - start} sec')
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, params))
