import matplotlib.pyplot as plt
import pandas as pd
import os

from config import DATA_DIR_2 as DATA_AFTER_DIR


plt.rcParams['font.sans-serif']=['SimHei']      #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False        #用来正常显示负号


def show_3D(sources, steps=0):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x/m', color='blue')
    ax.set_ylabel('y/m', color='blue')
    ax.set_zlabel('z/m', color='blue')
    ax.plot3D(sources[0], sources[1], sources[2], c='black', label='真实')
    # ax.scatter3D(sources[0][::steps], sources[1][::steps], sources[2][::steps], c='blue', marker='o', s=3)
    plt.legend()
    plt.show()


def show_2D(sources, steps=0):
    plt.figure()
    ax = plt.axes()
    ax.set_xlabel('lontitude')
    ax.set_ylabel('latitude')
    plt.plot(sources[0][steps:], sources[1][steps:], c='red', label='真实')
    # plt.scatter(sources[0][steps:], sources[1][steps:], c='red', marker='o', s=1, label='真实')
    # plt.scatter(sources[0][-1:], sources[1][-1:], c='black')
    plt.legend()
    plt.show()


# batch channel sequeu
if __name__ == '__main__':
    num_times = 1
    # 载入数据集
    for pp in os.listdir(DATA_AFTER_DIR):
        p = os.path.join(DATA_AFTER_DIR, pp)
        df = pd.read_csv(p, sep=',')

        sources = [
            df.loc[3200:, '经度'].to_numpy(),
            df.loc[3200:, '纬度'].to_numpy(),
            df.loc[3200:, '高度'].to_numpy()
        ]
        show_3D(sources, 200)

        num_times -= 1
        if num_times <= 0:
            break
