# -*- coding: utf8 -*-
import os
import random
import heapq
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.patches import Rectangle


COLOR_LIST = ['y', 'r', 'g', 'b', 'c', 'm', 'k']


# ===============data load section===============
def normalizeDataSet(dataset):
    '''
    数据标准化: (value - min) / range
    dataset是样本数据，标签不需要标准化
    返回标准化后的数据，以及每个特征的数据范围和最小值（用于标准化测试数据）
    '''
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    rangevals = 1.0 * (maxvals - minvals)
    dataset_count = dataset.shape[0]
    norm_dataset = dataset - np.tile(minvals, (dataset_count, 1))
    norm_dataset = norm_dataset / np.tile(rangevals, (dataset_count, 1))
    return np.around(norm_dataset, 2), np.around(rangevals, 2), minvals


def loadDataSetOriginally(dataset_path):
    '''
    加载数据集, 不做任何处理
    '''
    dataset = []

    with open(dataset_path) as dsf:
        for line in dsf:
            data = line.strip().split('\t')
            dataset.append(data)

    return dataset


def loadDataSetNorm(dataset_path):
    '''
    加载数据集，进行标准化处理
    返回标准化后的数据集，以及用于标准化未来测试数据的minvals和rangevals
    '''
    feature_list = []
    label_list = []

    with open(dataset_path) as dsf:
        for line in dsf:
            data = line.strip().split('\t')
            feature_list.append(data[0:-1])
            label_list.append(data[-1])

    dataset = np.array(feature_list, dtype='float32')
    norm_dataset, rangevals, minvals = normalizeDataSet(dataset)
    label_list = np.array(label_list, dtype='int8')
    norm_dataset = np.column_stack((norm_dataset, label_list))

    return norm_dataset, rangevals, minvals


def img2List(file_path):
    digit_data = ''
    with open(file_path) as df:
        for line in df:
            digit_data += line.strip()
    return list(digit_data)


# ===============kNN classifier section===============
class KDNode:
    '''
    KD树上的节点
    '''
    def __init__(self, data, splitDim, lessThan, greaterThan):
        self.data = data # 节点所包含的样本数据
        self.splitDim = splitDim # 当前节点的分裂维度
        self.lessThan = lessThan # 在分裂维度上小于当前节点的分支
        self.greaterThan = greaterThan # 在分裂维度上大于当前节点的分支
        self.parent = None # 父节点
        self.distance = float('inf') # 当前节点与测试点的距离，这个是短时有效，随着测试点变化
        pass

    def __cmp__(self, other):
        if self.distance < other.distance:
            return -1
        elif self.distance > other.distance:
            return 1
        return 0


def buildKDTree(dataset):
    '''
    构建KD树
    '''
    if dataset.shape[0] < 1:
        return None
    elif dataset.shape[0] == 1:
        return KDNode(dataset[0], -1, None, None)

    # 计算dataset每个纬度上的方差，选择方差最大的维度作为分裂维度
    varList = np.var(dataset, axis=0, ddof=1)
    splitDim = np.argmax(varList[0:-1])  # 最后一列是标签，不参与比较

    # 按照分裂维度排序后，取得分裂维度的中位数
    dataset = dataset[dataset[:, splitDim].argsort()]
    median = dataset[:, splitDim][len(dataset) / 2]

    # 切割数据
    lessThanCount = np.where(dataset[:, splitDim] < median)[0].size
    lessThan = dataset[0:lessThanCount]
    curDataInstance = dataset[lessThanCount]
    greaterThan = dataset[lessThanCount + 1:]

    # 创建子树
    lessThanBranch = buildKDTree(lessThan)
    greaterThanBranch = buildKDTree(greaterThan)

    # 创建当前节点
    curNode = KDNode(curDataInstance, splitDim, lessThanBranch, greaterThanBranch)
    if lessThanBranch:
        lessThanBranch.parent = curNode
    if greaterThanBranch:
        greaterThanBranch.parent = curNode

    return curNode


def searchKDTree(kdtree, k, target, kNeighbours):
    '''
    在kdtree中查找与target最近的k个数据实例
    返回最近的k个节点
    '''
    if kdtree is None:
        return kNeighbours

    # 先在子树上查找
    splitDim = kdtree.splitDim
    curNodeData = kdtree.data
    oppositeBranch = None
    if target[splitDim] < curNodeData[splitDim]:
        kNeighbours = searchKDTree(kdtree.lessThan, k,
                target, kNeighbours)
        oppositeBranch = kdtree.greaterThan
    else:
        kNeighbours = searchKDTree(kdtree.greaterThan, k,
                target, kNeighbours)
        oppositeBranch = kdtree.lessThan

    # 计算当前节点与target之间的距离
    kdtree.distance = np.linalg.norm(target - curNodeData[:-1])

    # 将当前节点添加到最近邻中
    # 如果最近邻树大于k， 从中去除最远的一个
    kNeighbours.append(kdtree)
    heapq._heapify_max(kNeighbours)
    if len(kNeighbours) > k:
        del kNeighbours[0]
    maxRadius = heapq.nlargest(1, kNeighbours)[0].distance

    # 当最近邻数小于k或者当前最大距离大于目标节点与当前节点在分裂维度上的距离
    # 则进入另一侧继续查找
    if np.abs(target[splitDim] - curNodeData[splitDim]) < maxRadius or \
            len(kNeighbours) < k:
        kNeighbours = searchKDTree(oppositeBranch, k, target, kNeighbours) 

    return kNeighbours


def verifyLabel(kNeighbours):
    '''
    根据K近邻找出数据标签
    1.）如果每种类型个数不相等，选择最多类型作为结果；
    2.）否则选择最近类型作为结果标签；
    '''
    if len(kNeighbours) < 1:
        return -1
    if len(kNeighbours) == 1:
        return kNeighbours[0].data[-1]

    label_dict = {}
    for neighbor in kNeighbours:
        label_dict.setdefault(neighbor.data[-1], 0)
        label_dict[neighbor.data[-1]] += 1
    label_count_tuple = label_dict.items()
    label_count_tuple = sorted(label_count_tuple, key=lambda x: x[1], reverse=True)

    max_count = label_count_tuple[0][1]
    if any([item[1] != max_count for item in label_count_tuple]):
        return label_count_tuple[0][0]

    # 如果所有的类型的邻居一样多，就返回最近的那个类型
    heapq.heapify(kNeighbours)
    return kNeighbours[0].data[-1]


def knnClassifier(kdtree, k, testData):
    '''
    K近邻分类器
    根据kdtree，对testData进行分类
    '''
    kNeighbours = searchKDTree(kdtree, k, testData, [])
    label = verifyLabel(kNeighbours)
    maxRadius = heapq.nlargest(1, kNeighbours)[0].distance
    return label, maxRadius


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


# ===============在给定的样本集上进行交叉验证测试，找出最佳的K===============
def validateK(sampleDataSet, testDataSet, k):
    '''
    使用sampleDataSet构建kd树，在testDataSet上验证k的错误率
    sampleDataSet和testDataSet都是经过标准化的数据
    返回k对应的错误率
    '''
    kdtree = buildKDTree(sampleDataSet)
    err = 0.0
    for testData in testDataSet:
        label, maxRadius = knnClassifier(kdtree, k, testData[0:-1])
        if label != testData[-1]:
            err += 1
            print 'predict: %s, actual: %s' % (label, testData[-1])

    error_rate = err / len(testDataSet)
    print 'error rate for %d: %s' % (k, error_rate)
    return error_rate


def cvSelectBestK(orgDataset):
    '''
    K的取值范围为1~10，通过交叉验证找出最佳的K值
    将数据打乱顺序后分成10份，一份作为测试数据，其它的作为样本数据用于
    构建kd树，用测试数据在kd树上计算出错误率；
    针对每一个K值，每一份数据都要作为测试数据测一遍，累计当前K值的错误率后求平均值；
    最终找出最佳K值
    '''
    random.shuffle(orgDataset)
    orgDatasetList = []
    for start in range(0, 10):
        orgDatasetList.append(orgDataset[start::10])

    k2Err = {}
    for k in range(1, 11):
        print '==========k is %d==========' % k
        err_rate = 0.0
        for i in range(len(orgDatasetList)):
            testDataSet = orgDatasetList[i]
            sampleDataSet = []
            for j in range(len(orgDatasetList)):
                if i == j:
                    continue
                sampleDataSet.extend(orgDatasetList[j])

            sampleDataSet = np.array(sampleDataSet, dtype='float32')
            testDataSet = np.array(testDataSet, dtype='float32')
            # normalize sample data 
            norm_samples, rangevals, minvals = normalizeDataSet(sampleDataSet[:, 0:-1])
            norm_samples = np.column_stack((norm_samples, sampleDataSet[:, -1]))
            # normalize test data
            test_count = len(testDataSet)
            norm_test = testDataSet[:, 0:-1] - np.tile(minvals, (test_count, 1))
            norm_test = np.around(norm_test / np.tile(rangevals, (test_count, 1)), 2)
            norm_test = np.column_stack((norm_test, testDataSet[:, -1]))

            err_rate += validateK(norm_samples, norm_test, k)
        err_rate = np.around(err_rate / len(orgDatasetList), 4)
        print 'error rate for %d is %f' % (k, err_rate)
        k2Err[k] = err_rate
    
    kErrTupleList = sorted(k2Err.items(), key=lambda x: x[1])
    print kErrTupleList
    print 'The best K is %d, its error rate is %f' % (kErrTupleList[0][0], kErrTupleList[0][1])
    plotKVsErr(k2Err)
    return kErrTupleList[0][0]


# ===============demo section===============
def demoRandomly(dataset_count, K):
    '''
    随机生成dataset_count个样本，每个样本数据有两个特征和一个标签
    特征为1~100间的随机整数，标签为1~3的随机整数， 构建KD树；
    生成一个随机target数据，在KD树上找到离target最近的K个节点，
    以K个节点中比例最大的标签作为target的标签；
    如果所有类型一样多，则以离target最近的节点标签做为target的标签
    '''

    # generate smaple data set
    feature_list = np.random.randint(1, 100, (dataset_count, 2))
    label_list = np.random.randint(1, 4, (dataset_count, 1))
    norm_dataset, rangevals, minvals = normalizeDataSet(feature_list)
    norm_dataset = np.column_stack((norm_dataset, label_list))
    print norm_dataset
    print rangevals
    print minvals

    # generate target data
    target = np.random.randint(1, 100, (1, 2))[0]
    target = np.around((target - minvals) / rangevals, 2)
    print target

    kdtree = buildKDTree(norm_dataset)
    target_label, maxRadius = knnClassifier(kdtree, K, target)

    # plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # plot border for data domain
    ax.add_patch(Rectangle((0, 0), 1.0, 1.0, fill=False, color='y'))
    plotKDTree2D(ax, kdtree, [[0, 1.0], [1.0, 0]])
    plotDataSet2D(ax, norm_dataset)
    plotTarget2D(ax, target, maxRadius, target_label)

    ax.legend()
    ax.set_xlabel('feature1')
    ax.set_ylabel('feature2')
    plt.show()


# 以下数据来自《Machine Learning in Action》第二章
def demoDatingTestSet():
    # 在datingTestSet上进行交叉验证后找出最佳K
    orgDataset = loadDataSetOriginally('./datingTestSet.txt')
    bestK = cvSelectBestK(orgDataset)
    print 'the best K for datingTestSet2.txt is %d' % bestK


def predictDatingPerson(K=5):
    '''
    使用datingTestSet.txt中样本数据构建kd树，
    对输入的数据进行分类
    默认K值是用demoDatingTestSet找出最佳K值
    '''
    # 输入数据
    ffMiles = float(raw_input("frequent flier miles earned per year?")) 
    percentTats = float(raw_input("percentage of time spent playing video games?")) 
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    personData = np.array([ffMiles, percentTats, iceCream], dtype='float32')

    # 使用datingTestSet构建KD树
    norm_dataset, rangevals, minvals = loadDataSetNorm('./datingTestSet.txt')
    kdtree = buildKDTree(norm_dataset)

    # 预测对约会对象的喜欢程度
    target = np.around((personData - minvals) / rangevals, 2)
    label, maxRadius = knnClassifier(kdtree, K, target)
    resultList = ['not at all', 'in small doses', 'in large doses']
    print 'predict label is %s' % (resultList[int(label) - 1])
    plotDataSet3D(norm_dataset, target, label)


# handwriting recognition system
def classifyDigit():
    # 从trainingDigits中加载样本数据
    digit_file_list = os.listdir('./trainingDigits')
    digit_data_list = []
    for digit_file in digit_file_list:
        digit_data = img2List(os.path.join('./trainingDigits', digit_file))
        digit_label = int(os.path.basename(digit_file)[0])
        digit_data.append(digit_label)
        digit_data_list.append(digit_data)
    data_set = np.array(digit_data_list, dtype='int8')        

    # 构建KD树
    kdtree = buildKDTree(data_set)

    # 对testDigits下的数字文件进行分类
    test_file_list = os.listdir('./testDigits')
    test_count = len(test_file_list)
    err = 0.0
    for test_digit in test_file_list:
        test_features = np.array(img2List(os.path.join('./testDigits', test_digit)), dtype='int8')
        test_label = int(os.path.basename(test_digit)[0])
        classified_label, maxRadius = knnClassifier(kdtree, 3, test_features)
        if test_label != classified_label:
            print 'actual: %s, predict: %s' % (test_label, classified_label)
            err += 1
    print 'the total number of error is: %d' % err
    print 'the total error rate is %s' % (np.around(err/test_count, 6))

if __name__ == '__main__':
    pass
