# -*- coding: utf8 -*-
import os
import random
import heapq
import numpy as np


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
    return kErrTupleList[0][0], k2Err


if __name__ == '__main__':
    pass
