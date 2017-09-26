# -*- coding: utf8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import itertools
import copy


class kd_node:
    '''
    KD树的节点
    '''
    def __init__(self, point, region, dim):
        self.point = point   # point stored in this kd-tree node
        self.region = region # [[lower_left_x, lower_left_y], [upper_right_x, upper_right_y]]
        self.split_dim = dim # on which dimension this point splits
        self.left = None
        self.right = None


def median(data_lst, split_dim):
    '''找出data_lst的中位数'''
    d = len(data_lst) / 2
    # make sure that on split_dim dimension all the points in right subtree 
    # are equal or greater than the point stored in root node
    l = 0
    h = d
    while l < h:
        m = (l + h) / 2
        if data_lst[m][split_dim] < data_lst[h][split_dim]:
            l = m + 1
        else:
            h = m
    return data_lst[h], h


def get_split_dim(data_lst):
    """
    计算points在每个维度上的方差, 选择在方差最大的维度上进行切割
    """
    var_lst = np.var(data_lst, axis=0)
    split_dim = 0
    for v in range(1, len(var_lst)):
        if var_lst[v] > var_lst[split_dim]:
            split_dim = v
    return split_dim


def build_kdtree(data_lst, region, square_list):
    '''构建kd树'''
    split_dim = get_split_dim(data_lst)
    data_lst = sorted(data_lst, key=lambda x: x[split_dim])
    point, m = median(data_lst, split_dim)
    tree_node = kd_node(point, region, split_dim)
    square_list.append(region)

    print 'split point: %s, split_dim: %s' % (point, split_dim)

    if m > 0:
        sub_region = copy.deepcopy(region)
        sub_region[1][split_dim] = point[split_dim]
        tree_node.left = build_kdtree(data_lst[:m], sub_region, square_list)

    if len(data_lst) > m + 1:
        sub_region = copy.deepcopy(region)
        sub_region[0][split_dim] = point[split_dim]
        tree_node.right = build_kdtree(data_lst[m + 1:], sub_region, square_list)

    return tree_node


def illustrate_build_kd_tree():
    colors = itertools.cycle(["#FF6633", "g", "#3366FF", "c", "m", "y", '#EB70AA', '#0099FF', '#66FFFF'])

    fig = plt.figure(figsize=(4, 4), dpi=128, facecolor='w')
    ax = plt.axes(xlim=(0, 10), ylim=(0, 10), title="BUILD KD TREE",
                  xlabel="X Axis", ylabel="Y Axis")
    ax.grid(False)

    def draw_static_elements():
        # draw static elements
        global T, TL
        X = [p[0] for p in T]
        Y = [p[1] for p in T]
        ax.plot(X, Y, 'bo')
        for pt in T:
            ax.text(pt[0], pt[1] - 0.5, '%s %s' % (TL.get(str(pt), 'X'), pt))

    def init():
        return []

    def animate(i):
        square = square_list[i]
        print 'draw square: %s' % square
        ax.add_patch(Rectangle((square[0][0], square[0][1]), 
                     square[1][0] - square[0][0], 
                     square[1][1] - square[0][1],
                     edgecolor='r', facecolor='w'))
                     #color=next(colors)))
        return []

    draw_static_elements()
    anim = FuncAnimation(fig, animate, frames=len(square_list), init_func=init,
                         interval=1000, blit=True, repeat=True, repeat_delay=5000)
    anim.save('./illustrators/build_kd_tree.gif', dpi=80, writer='imagemagick')
    plt.show()


def euclid_distance(d1, d2):
    dist = np.linalg.norm(np.array(d1) - np.array(d2))
    print 'check %s - %s; distance: %s' % (d1, d2, dist)
    return dist


class NeiNode:
    '''neighbor node'''
    def __init__(self, p, d):
        self.__point = p
        self.__dist = d

    def get_point(self):
        return self.__point

    def get_dist(self):
        return self.__dist

class BPQ:
    '''bounded priority queue'''
    def __init__(self, k):
        self.__K = k
        self.__pos = 0
        self.__bpq = [0] * (k + 2)

    def add_neighbor(self, neighbor):
        self.__pos += 1
        self.__bpq[self.__pos] = neighbor
        self.__swim_up(self.__pos)
        if self.__pos > self.__K:
            self.__exchange(1, self.__pos)
            self.__pos -= 1
            self.__sink_down(1)

    def get_knn_points(self):
        return [neighbor.get_point() for neighbor in self.__bpq[1:self.__pos + 1]]

    def get_max_distance(self):
        if self.__pos > 0:
            return self.__bpq[1].get_dist()
        return 0

    def is_full(self):
        return self.__pos >= self.__K

    def print_bpq(self):
        if self.__pos < 1:
            print 'no neighbor'
        print 'nearest %d neighbors: ' % self.__K
        for p in self.__bpq[1: self.__pos + 1]:
            print '    %s: %s' % (p.get_point(), p.get_dist())
        print 'max distance: %s' % self.get_max_distance()
        print ''

    def __swim_up(self, n):
        while n > 1 and self.__less(n/2, n):
            self.__exchange(n/2, n)
            n = n/2

    def __sink_down(self, n):
        while 2*n <= self.__pos:
            j = 2*n
            if j < self.__pos and self.__less(j, j+1):
                j += 1
            if not self.__less(n, j):
                break
            self.__exchange(n, j)
            n = j

    def __less(self, m, n):
        return self.__bpq[m].get_dist() < self.__bpq[n].get_dist()

    def __exchange(self, m, n):
        tmp = self.__bpq[m]
        self.__bpq[m] = self.__bpq[n]
        self.__bpq[n] = tmp


def knn_search_kd_tree_recursively(knn_bpq, tree, target, search_track):
    if not tree:
        return

    search_track.append([tree.point, knn_bpq.get_knn_points(), knn_bpq.get_max_distance()])
    dist = euclid_distance(tree.point, target)
    knn_bpq.add_neighbor(NeiNode(tree.point, dist))
    knn_bpq.print_bpq()
    search_track.append([None, knn_bpq.get_knn_points(), knn_bpq.get_max_distance()])
 
    split_dim = tree.split_dim
    if target[split_dim] < tree.point[split_dim]:
        knn_search_kd_tree_recursively(knn_bpq, tree.left, target, search_track)
        opposite_branch = tree.right
    else:
        knn_search_kd_tree_recursively(knn_bpq, tree.right, target, search_track)
        opposite_branch = tree.left

    if not knn_bpq.is_full() or \
            abs(target[split_dim] - tree.point[split_dim]) < knn_bpq.get_max_distance():
        knn_search_kd_tree_recursively(knn_bpq, opposite_branch, target, search_track)


def knn_search_kd_tree_non_recursively(knn_bpq, tree, target, search_track):
    track_node = []
    node_ptr = tree
    while node_ptr:
        while node_ptr:
            track_node.append(node_ptr)
            search_track.append([node_ptr.point, knn_bpq.get_knn_points(), knn_bpq.get_max_distance()])
            dist = euclid_distance(node_ptr.point, target)
            knn_bpq.add_neighbor(NeiNode(node_ptr.point, dist))
            knn_bpq.print_bpq()
            search_track.append([None, knn_bpq.get_knn_points(), knn_bpq.get_max_distance()])

            split_dim = node_ptr.split_dim
            if target[split_dim] < node_ptr.point[split_dim]:
                node_ptr = node_ptr.left
            else:
                node_ptr = node_ptr.right

        while track_node:
            iter_node = track_node[-1]
            del track_node[-1]

            split_dim = iter_node.split_dim
            if not knn_bpq.is_full() or \
                    abs(iter_node.point[split_dim] - target[split_dim]) < knn_bpq.get_max_distance():
                if target[split_dim] < iter_node.point[split_dim]:
                    node_ptr = iter_node.right
                else:
                    node_ptr = iter_node.left

            if node_ptr:
                break


def illustrate_search_kd_tree(k):
    global T, S, square_list
    fig = plt.figure(figsize=(4, 4), dpi=128, facecolor="white")
    ax = plt.axes(xlim=(0, 12), ylim=(0, 12), title="SEARCH KD TREE",
                  xlabel="X Axis", ylabel="Y Axis", )
    ax.grid(False)

    def draw_static_elements():
        X = [p[0] for p in T]
        Y = [p[1] for p in T]
        ax.plot(X, Y, 'bo')
        for pt in T:
            ax.text(pt[0], pt[1] - 0.6, '%s %s' % (TL.get(str(pt), 'X'), pt))
        ax.plot(S[0], S[1], 'r*')
        ax.text(S[0], S[1] - 0.6, '%s %s' % ('S', S))
        for square in square_list:
            ax.add_patch(Rectangle((square[0][0], square[0][1]),
                         square[1][0] - square[0][0],
                         square[1][1] - square[0][1],
                         edgecolor='r', facecolor='w'))

    def init():
        test_point_plot.set_data([], [])
        nearest_point_plot.set_data([], [])
        nearest_dist_circle.set_radius(0)
        return [test_point_plot, nearest_point_plot, nearest_dist_circle]

    def animate(i):
        cur_track = search_track[i]
        if cur_track[0]:
            test_point_plot.set_data(cur_track[0][0], cur_track[0][1])
        else:
            test_point_plot.set_data([], [])
        if cur_track[1]:
            nearest_point_plot.set_data([x[0] for x in cur_track[1]], [y[1] for y in cur_track[1]])
        else:
            nearest_point_plot.set_data([], [])
        if cur_track[2] < float('inf'):
            nearest_dist_circle.set_radius(cur_track[2])
        else:
            nearest_dist_circle.set_radius(0)
        return [test_point_plot, nearest_point_plot, nearest_dist_circle]

    draw_static_elements()
    test_point_plot, = ax.plot([], [], 'ro')
    nearest_point_plot, = ax.plot([], [], 'yo')
    nearest_dist_circle = Circle(S, radius=0, fill=False, ls='dashdot')
    ax.add_patch(nearest_dist_circle)

    anim = FuncAnimation(fig, animate, frames=len(search_track), init_func=init,
                         interval=1000, blit=True, repeat=True, repeat_delay=5000)
    anim.save('./illustrators/%dNN_search_kd_tree.gif' % k, dpi=80, writer='imagemagick')
    plt.show()


def print_kd_tree(tree):
    level = 0
    node_list = [tree]
    while node_list:
        print 'level %s: %s' % (level, ' '.join([str(node.point) for node in node_list]))
        next_node_list = []
        for node in node_list:
            if node.left:
                next_node_list.append(node.left)
            if node.right:
                next_node_list.append(node.right)
        node_list = next_node_list


if __name__ == '__main__':
    #T = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    T = [[3, 5], [6, 2], [5, 8], [9, 3], [8, 6], [1, 1], [2, 9]]
    S = [8.2, 4.6]
    TL = {}
    lable = 65
    for t in T:
        TL[str(t)] = chr(lable)
        lable += 1
    square_list = []

    print 'building kd-tree ...'
    kd_tree = build_kdtree(T, [[0, 0], [12, 12]], square_list)
    print_kd_tree(kd_tree)
    print 'end of building kd-tree\n'
    #illustrate_build_kd_tree()

    
    k = 2
    knn_bpq = BPQ(k)
    search_track = []
    print 'begin to search target point %s in kd-tree' % S
    knn_search_kd_tree_non_recursively(knn_bpq, kd_tree, S, search_track)
    print '==========final result=========='
    print 'in data set %s' % T
    print 'nearest %d neighbors of %s listed belown' % (k, S)
    knn_bpq.print_bpq()
    illustrate_search_kd_tree(k)


