import matplotlib.pyplot as plt

from data_loader import data_iter_order
from config import LABEL_NORMALIZATION, NORMALIZATION_TIMES, batch_size


plt.rcParams['font.sans-serif']=['SimHei']      #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False        #用来正常显示负号


def show_3D(sources, steps=0):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('lontitude')
    ax.set_ylabel('latitude')
    ax.set_zlabel('height')
    ax.plot3D(sources[0][steps:], sources[1][steps:], sources[2][steps:], c='red')
    plt.show()


def show_2D(sources, steps=0):
    plt.figure()
    ax = plt.axes()
    ax.set_xlabel('lontitude')
    ax.set_ylabel('latitude')
    plt.plot(sources[0][steps:], sources[1][steps:], c='red', label='真实')
    plt.scatter(sources[0][-1:], sources[1][-1:], c='black')
    plt.legend()
    plt.show()


# batch channel sequeu
if __name__ == '__main__':
    # 载入数据集
    num_times = 10
    for X, Y in data_iter_order(batch_size):
        num_times -= 1

        if num_times <= 0:
            break
        sources = [
            Y[:, 1].numpy() * (LABEL_NORMALIZATION[1][1] - LABEL_NORMALIZATION[1][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[1][0],
            Y[:, 2].numpy() * (LABEL_NORMALIZATION[2][1] - LABEL_NORMALIZATION[2][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[2][0],
            Y[:, 3].numpy() * (LABEL_NORMALIZATION[3][1] - LABEL_NORMALIZATION[3][0]) / NORMALIZATION_TIMES + LABEL_NORMALIZATION[3][0]
        ]
        show_3D(sources)
