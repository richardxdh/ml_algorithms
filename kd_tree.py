# -*- coding: utf8 -*-
# REF: http://www.hankcs.com/ml/k-nearest-neighbor-method.html
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import itertools
import copy


class node:
    def __init__(self, point):
        self.left = None
        self.right = None
        self.point = point


def median(data_list):
    m = len(data_list) / 2
    return data_list[m], m


def build_kdtree(data_list, dim, square, square_list):
    square_list.append(square)
    data_list = sorted(data_list, key=lambda x: x[dim])
    point, m = median(data_list)
    tree_node = node(point)

    print point

    if m >= 0:
        sub_square = copy.deepcopy(square)
        if dim == 0:
            sub_square[1][0] = point[0]
        else:
            sub_square[1][1] = point[1]
        square_list.append(sub_square)
        if m > 0:
            tree_node.left = build_kdtree(data_list[:m], not dim, sub_square, square_list)

    if len(data_list) > m + 1:
        sub_square = copy.deepcopy(square)
        if dim == 0:
            sub_square[0][0] = point[0]
        else:
            sub_square[0][1] = point[1]
        tree_node.right = build_kdtree(data_list[m + 1:], not dim, sub_square, square_list)

    return tree_node


def kd_tree_illustrate():
    T = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    colors = itertools.cycle(["#FF6633", "g", "#3366FF", "c", "m", "y", '#EB70AA', '#0099FF', '#66FFFF'])

    square_list = []
    build_kdtree(T, 0, [[0, 0], [10, 10]], square_list)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 10), ylim=(0, 10), title="BUILD KD TREE",
                  xlabel="X Axis", ylabel="Y Axis")
    ax.grid(True)

    def init():
        X = [p[0] for p in T]
        Y = [p[1] for p in T]
        ax.plot(X, Y, 'bo')
        for pt in T:
            ax.text(pt[0], pt[1] - 0.5, str(pt))
        return []

    def animate(i):
        square = square_list[i]
        print square
        ax.add_patch(Rectangle((square[0][0], square[0][1]), 
                     square[1][0] - square[0][0], 
                     square[1][1] - square[0][1],
                     color=next(colors)))
        return []

    anim = FuncAnimation(fig, animate, frames=len(square_list), init_func=init,
                         interval=1000, blit=True, repeat=True, repeat_delay=5000)
    plt.show()
    anim.save('build_kd_tree.gif', dpi=80, writer='imagemagick')


if __name__ == '__main__':
    kd_tree_illustrate()
