# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.patches import Rectangle


COLOR_LIST = ['y', 'r', 'g', 'b', 'c', 'm', 'k']
# ===============plot section===============
def plotKDTree2D(ax, kdtree, area):
    '''
    在二维平面上画出kdtre的所有分割线以及数据点文本描述
    '''
    if kdtree is None:
        return

    curx = kdtree.data[0]
    cury = kdtree.data[1]
    color = COLOR_LIST[int(kdtree.data[-1]) % len(COLOR_LIST)]

    # plot data text
    ax.text(curx + 0.01, cury + 0.01, '[%s, %s]' % (curx, cury), color=color)

    # draw split line
    if kdtree.splitDim == 0:
        ax.plot([curx, curx], [area[0][1], area[1][1]], color+'--', linewidth=1.0)
        lessThanArea = [area[0], [curx, area[1][1]]]
        greaterThanArea = [[curx, area[0][1]], area[1]]

    else:
        ax.plot([area[0][0], area[1][0]],[cury, cury],  color+'--', linewidth=1.0)
        lessThanArea = [[area[0][0], cury], area[1]]
        greaterThanArea = [area[0], [area[1][0], cury]]

    # draw less than
    plotKDTree2D(ax, kdtree.lessThan, lessThanArea)
    # draw greater than
    plotKDTree2D(ax, kdtree.greaterThan, greaterThanArea)


def plotTarget2D(ax, target, maxRadius, target_label):
    '''
    在二维图上画出测试点，用星号表示，颜色分类器预测的结果保持一致
    以K近邻中最远的邻居距离为半径画圆，圆圈将包含所有的K近邻数据点
    '''
    # plot target and circle with maxradius as radius
    color=COLOR_LIST[int(target_label) % len(COLOR_LIST)]
    ax.text(target[0] + 0.01, target[1] + 0.01, '[%s, %s]' % (target[0], target[1]), color=color)
    ax.plot(target[0], target[1], color+'*', markersize=8)
    circle = plt.Circle((target[0], target[1]), maxRadius, color=color, fill=False)
    ax.add_artist(circle)


def plotDataSet2D(ax, dataset):
    '''
    画出所有的二维数据点，颜色表示数据标签
    '''
    label_list = set(dataset[:, -1])
    for label in label_list:
        label_idx = np.where(dataset[:, -1] == label)
        feature_list = dataset[label_idx]
        color = COLOR_LIST[int(label) % len(COLOR_LIST)]
        ax.plot(feature_list[:, 0], feature_list[:, 1], color+'.', 
                label='TYPE%d' % label, markersize=8.0)


def plotDataSet3D(dataset, target, target_label):
    '''
    在三维图中画出所有的数据点和测试点
    '''
    fig = plt.figure(figsize=(10, 10))
    # plot 3D
    ax = fig.add_subplot(111, projection='3d')

    label_list = set(dataset[:, -1])
    for label in label_list:
        label_idx = np.where(dataset[:, -1] == label)
        feature_list = dataset[label_idx]
        color = COLOR_LIST[int(label) % len(COLOR_LIST)]
        ax.plot(feature_list[:, 0], feature_list[:, 1], feature_list[:, 2], 
                color+'.', label='TYPE%d' % label, markersize=2.0)

    # plot target
    if target is not None:
        color = COLOR_LIST[int(target_label) % len(COLOR_LIST)]
        ax.plot([target[0],], [target[1],], [target[2],], color+'*', 
                label='TYPE%d' % target_label, markersize=8.0)

    ax.legend()
    ax.set_xlabel('feature1')
    ax.set_ylabel('feature2')
    ax.set_zlabel('feature3')

    plt.show()


def plotKVsErr(k2Err):
    items = k2Err.items()
    items = sorted(items, key=lambda x: x[0])
    x = [item[0] for item in items]
    y = [item[1] for item in items]
    fig = plt.figure()
    plt.plot(x, y, 'g-')
    plt.xlabel('K')
    plt.ylabel('error rate')
    plt.show()


