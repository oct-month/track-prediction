import os
from time import time
import torch
from torch import nn, optim

from model import HybridCNNLSTM, loss
from data_loader import data_iter_load
from config import PARAMS_PATH, batch_size, num_epochs, lr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# batch channel sequeue
if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Note: Using cuda to train.')
    else:
        print('Note: Using cpu to train.')
    
    model = HybridCNNLSTM()
    if os.path.isfile(PARAMS_PATH):
        model.load_state_dict(torch.load(PARAMS_PATH))
        print('Warning: Using the existing params to train.')
    else:
        for param in model.parameters():
            nn.init.normal_(param, mean=0, std=1)
            # nn.init.zeros_(param)
    model.to(device)
    model.train()

    # 输出模型参数
    # print('****** ↓ Model Parameters ↓ ******')
    # for k, v in model.named_parameters():
    #     print(k, v.data)
    # print('****** ↑ Model Parameters ↑ ******')

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    for epoch in range(1, num_epochs + 1):
        print(f'epoch {epoch}, ', end='')
        l_sum, n, start = 0.0, 0, time()

        grad_flag = True
        states = None
        for X, Y in data_iter_load(batch_size):
            y, states = model(X, states)
            l = loss(Y, y)

            optimizer.zero_grad()
            l.backward()
            # 输出梯度
            if grad_flag:
                # print('****** ↓ Parameters\' grad ↓ ******')
                # for k, v in model.named_parameters():
                #     print(k, v.grad)
                # print('****** ↑ Parameters\' grad ↑ ******')
                grad_flag = False
            for state in states:
                state.detach_()
            optimizer.step()
            l_sum += l.item()
            n += y.shape[0]
        # 输出
        print(f'loss {l_sum / n}, time {time() - start}, n {n}.')
    
    # 输出模型参数
    # print('****** ↓ Model Parameters ↓ ******')
    # for k, v in model.named_parameters():
    #     print(k, v.data)
    # print('****** ↑ Model Parameters ↑ ******')
    # 保存模型
    torch.save(model.cpu().state_dict(), PARAMS_PATH)
    # 打印参数值
    # for k, v in model.named_parameters():
    #     print(k, v.data)
