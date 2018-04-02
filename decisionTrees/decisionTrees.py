# -*- coding: utf8 -*-
import operator
import numpy as np


def calcShannonEntropy(dataset):
    '''
    计算数据集的熵
    @param dataset numpy.array
    '''
    labels = set(dataset[:, -1])
    label_counts = np.array([np.count_nonzero(dataset[:, -1] == label) for
        label in labels])
    label_probs = label_counts / float(dataset.shape[0])

    entropy = -np.sum(label_probs * np.log2(label_probs))

    return entropy


def splitDataSet(dataset, dim):
    '''
    将数据集dataset在维度dim上进行划分
    @param dataset numpy.array
    '''
    vals = set(dataset[:, dim])
    subs = {}
    for val in vals:
        sub_data_set = dataset[dataset[:, dim] == val]
        subs[val] = np.column_stack((sub_data_set[:, :dim],
            sub_data_set[:, dim+1:]))

    return subs        


def chooseBestFeatureAndSplit(dataset):
    '''
    选择最佳的划分特征，既信息增益最大的特征，然后进行划分
    @param dataset numpy.array
    '''
    before_split_entropy = calcShannonEntropy(dataset)
    best_info_gain = 0.0
    best_dim = -1
    best_split = {}

    dims = dataset.shape[1] - 1
    for dim in range(dims):
        after_split_entropy = 0.0
        subs = splitDataSet(dataset, dim)
        for val in subs.keys():
            prob = len(subs[val]) / float(dataset.shape[0])
            after_split_entropy += prob * calcShannonEntropy(subs[val])

        cur_info_gain = before_split_entropy - after_split_entropy
        if cur_info_gain > best_info_gain:
            best_info_gain = cur_info_gain
            best_dim = dim
            best_split = subs

    return best_dim, best_split


def majorityVote(dataset):
    '''
    当所有的特征都检查过后，如果dataset中的label还不统一，此时需要通过多数票决的
    方式来选出label作为最终叶节点的label
    @param dataset numpy.array
    '''
    label2Count = {}
    for label in dataset:
        label2Count.setdefault(label, 0)
        label2Count[label] += 1
    sortedLabel2Count = sorted(label2Count.iteritems(),
            key=operator.itemgetter(1), reverse=True)
    return sortedLabel2Count[0][0]


def createDecisionTreeID3(dataset, featNames):
    '''
    创建决策树
    @param dataset numpy.array
    @param featNames list 
    '''

    # 样本集中的标签已经统一了
    if dataset.shape[0] == np.count_nonzero(dataset[:, -1] == dataset[0][-1]):
        return dataset[0][-1]

    # 样本集的所有特征都已经检查过了，但是标签还不统一，需要进行多数票决
    if len(dataset[0]) == 1:
        return majorityVote(dataset)

    # 构建当前树
    bestDim, bestSplit = chooseBestFeatureAndSplit(dataset)
    bestFeatName = featNames[bestDim]
    curTree = {bestFeatName: {}}
    del featNames[bestDim]

    # 构建子树
    for featval, subset in bestSplit.iteritems():
        curTree[bestFeatName][featval] = createDecisionTreeID3(subset, featNames[:])

    return curTree


def classify(decisionTree, featNames, testData):
    '''
    使用决策树decisionTree对测试样本testData进行分类
    '''
    decisionFeat = decisionTree.keys()[0]
    nextLevel = decisionTree[decisionFeat]
    decisionFeatIdx = featNames.index(decisionFeat)

    testLabel = 'unknown'
    for featVal in nextLevel.keys():
        if testData[decisionFeatIdx] == int(featVal):
            if type(nextLevel[featVal]) is dict:
                testLabel = classify(nextLevel[featVal], featNames, testData)
            else:
                testLabel = nextLevel[featVal]

    return testLabel


def createTestDataSet():
    dataSet = [[1, 1, 'yes'], 
            [1, 1, 'yes'], 
            [1, 0, 'no'], 
            [0, 1, 'no'], 
            [0, 1, 'no']] 
    labels = ['no surfacing','flippers'] 
    return dataSet, labels


if __name__ == '__main__':
    myDat, labels = createTestDataSet()
    myTree = createDecisionTreeID3(np.array(myDat), labels[:])
    print myTree
    testLabel = classify(myTree, labels[:], [1,0])
    print '[1,0] label is %s' % testLabel

