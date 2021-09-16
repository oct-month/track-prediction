from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from load import load_data

TUDE_M = 111000     # 一个经纬度对应的米数
SPARSE = 10         # 稀疏倍数

ATHLETIC_TRACK_1 = [(108.75124502525209, 34.421111039119765), (108.78197765997065, 34.44334435911524)]
ATHLETIC_TRACK_2 = [(108.74011042767648, 34.43817675412275), (108.76431414029567, 34.455728492291556)]

def show_3D():
    dts = load_data()

    dts = dts[::SPARSE]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    X = [d.longitude for d in dts]
    Y = [d.latitude for d in dts]
    Z = [d.height for d in dts]
    ax.scatter3D(X, Y, Z, c='blue')

    ax.scatter3D([X[-1], 108.75121335514869], [Y[-1], 34.42109991733429], [Z[-1], Z[-1]], c='red')
    
    print(X[-1], Y[-1])

    plt.show()


def y(a:float, b:float, X: np.ndarray) -> np.ndarray:
    return a * X + b


def gen_line(A: Tuple[float, float], B: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    a = (A[1] - B[1]) / (A[0] - B[0])
    b = (A[0] * B[1] - B[0] * A[1]) / (A[0] - B[0])
    X = np.linspace(A[0], B[0], 100, endpoint=True, dtype=np.float64)
    Y = y(a, b, X)
    return X, Y


def show_2D():
    dts = load_data()
    # dts = dts[::SPARSE]
    
    fig = plt.figure()

    X = [d.longitude for d in dts]
    Y = [d.latitude for d in dts]
    plt.scatter(X, Y, c='blue', s=0.5)

    plt.plot(*gen_line(ATHLETIC_TRACK_1[0], ATHLETIC_TRACK_1[1]), color='red', linewidth=5)
    plt.plot(*gen_line(ATHLETIC_TRACK_2[0], ATHLETIC_TRACK_2[1]), color='red', linewidth=5)

    plt.show()


if __name__ == '__main__':
    show_2D()
