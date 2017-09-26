#-*-coding: utf8 -*-
# import matplotlib
# matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import copy

NEGATIVE = -1
POSITIVE = 1


def calculate_distance(line_vec, point):
    '''
    计算点point到线line_vec的距离，含符号表示在线的哪一侧
    '''
    # line_vec[w0, w1, b]
    # point[x, y]
    lv = np.array(line_vec)
    pv = np.array(point + [1])
    d = np.dot(lv, pv) / np.sqrt(np.sum(lv**2))
    return d


def generate_dataset(line_vec, min_distance, point_count=5):
    '''
    在线line_vec的两侧各生成point_count个不同的点，每个点到线的距离绝对值不能小于min_distance
    生成的数据集中是每个点的坐标，以及点与线的位置关系[x, y, POSITIVE/NEGATIVE]
    '''
    data_set = []
    positive_count = 0
    negative_count = 0

    while positive_count < point_count or negative_count < point_count:
        point = [np.random.rand() * 10, np.random.rand() * 10]
        x_lst = [pt[0] for pt in data_set]
        y_lst = [pt[1] for pt in data_set]
        if point[0] in x_lst or point[1] in y_lst:
            continue
        
        dist = calculate_distance(line_vec, point)
        abs_dist = abs(dist)
        if abs_dist < min_distance:
            continue

        if dist > 0 and positive_count < point_count:
            p = point + [POSITIVE]
            positive_count += 1
            data_set.append(p)
        elif dist < 0 and negative_count < point_count:
            p = point + [NEGATIVE]
            negative_count += 1
            data_set.append(p)

    return data_set


# ============================================================

def sign(line_vec, pt):
    '''
    res=y(w.x + b), w和x为向量
    检测line_vec是否将点pt正确分类
    返回值<=0表示pt被误分
    返回值>0表示pt被正确分类
    '''
    lv = np.array(line_vec)
    ptv = np.array(pt[:2] + [1])
    res = lv.dot(ptv) # w.x + b
    res *= pt[2] # y*(w.x + b)
    return res


def perceptron(data_set, line_hist):
    '''
    感知机：遍历数据集，每遇到一个误判的点，及时做出适当的调整；
    直到找到一条线，将所有的点都正确地分隔在线的两边
    '''
    print 'default perceptron'
    hw = [0, 0, 0]
    while True:
        incorrect = 0
        for point in data_set:
            s = sign(hw, point)
            if s > 0:
                continue

            incorrect += 1
            hw[0] += point[2] * point[0]
            hw[1] += point[2] * point[1]
            hw[2] += point[2]
            line_hist.append(copy.deepcopy(hw))

        if incorrect == 0:
            break
    return hw


def perceptron_illustrate(perceptron_type='default', min_distance=0.5, point_count=3):
    original_line_vec = [2, 1, -12]  # w0, w1, b
    perceptron_line_hist = []
    data_set = generate_dataset(original_line_vec, min_distance, point_count)
    if perceptron_type == 'duality':
        duality_perceptron(data_set, perceptron_line_hist)
    else:
        perceptron(data_set, perceptron_line_hist)

    fig = plt.figure()
    ax = plt.axes(xlim=(-1, 10), ylim=(-1, 10), title="Perceptron",
                  xlabel="X Axis", ylabel="Y Axis",)
    ax.grid(True)

    perceptron_line, = ax.plot([], [], 'g-')

    def init():
        # original point
        ax.plot(0, 0, 'ro')

        # negative points
        nX = [p[0] for p in data_set if p[2] == NEGATIVE]
        nY = [p[1] for p in data_set if p[2] == NEGATIVE]
        ax.plot(nX, nY, 'rx')

        # positive points
        pX = [p[0] for p in data_set if p[2] == POSITIVE]
        pY = [p[1] for p in data_set if p[2] == POSITIVE]
        ax.plot(pX, pY, 'bo')

        # original line
        x = -10
        y = -(x * original_line_vec[0] + original_line_vec[2]) / original_line_vec[1]
        x_ = 60
        y_ = -(x_ * original_line_vec[0] + original_line_vec[2]) / original_line_vec[1]
        ax.plot([x, x_], [y, y_], 'r-')

        perceptron_line.set_data([], [])
        return perceptron_line,

    def animate(i, last, lines):
        line = lines[i]
        x = -10
        y = -(x * line[0] + line[2]) / line[1]
        x_ = 60
        y_ = -(x_ * line[0] + line[2]) / line[1]
        perceptron_line.set_data([x, x_], [y, y_])

        if i == last:
            perceptron_line.set_color('g')
            perceptron_line.set_linestyle('-')
        else:
            perceptron_line.set_color('b')
            perceptron_line.set_linestyle('-.')

        return perceptron_line,

    anim = FuncAnimation(fig, animate, frames=len(perceptron_line_hist), init_func=init,
                         fargs=(len(perceptron_line_hist) - 1, perceptron_line_hist),
                         repeat=True, repeat_delay=3000, blit=True, interval=500)

    anim.save('./illustrators/%s_perceptron.gif' % (perceptron_type,), dpi=80, writer='imagemagick')
    plt.show()


# ==============================duality perceptron=============================

def cal_gramm(data_set):
    g = np.empty((len(data_set), len(data_set)), np.float)
    for i in range(len(data_set)):
        for j in range(len(data_set)):
            g[i][j] = np.dot(np.array(data_set[i][:-1]), np.array(data_set[j][:-1]))
    return g


def duality_sign(a, b, g, y, idx):
    res = np.dot(a * y, g[idx]) # w.x_i
    res = (res + b) * y[idx] # y_i(w.x_i + b)
    return res


def update_duality_perceptron(a, b, x, y, idx, line_hist):
    a[idx] += 1
    b += y[idx]
    new_line = np.dot(a*y, x).tolist()
    new_line.append(b)
    line_hist.append(new_line)
    return b


def duality_perceptron(data_set, line_hist):
    print 'duality perceptron'

    A = [0.0] * len(data_set)
    B = 0.0
    X = np.array([p[:-1] for p in data_set])
    Y = np.array([p[-1] for p in data_set])
    G = cal_gramm(data_set)

    while True:
        incorrect = 0
        for i in range(len(data_set)):
            s = duality_sign(A, B, G, Y, i)
            if s > 0:
                continue

            incorrect += 1

            B = update_duality_perceptron(A, B, X, Y, i, line_hist)

        if incorrect == 0:
            break


if __name__ == '__main__':
    perceptron_illustrate('duality')

