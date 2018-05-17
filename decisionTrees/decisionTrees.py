# -*- coding: utf8 -*-
import operator
import numpy as np


ID3 = "decision_tree_ID3"
C45 = "decision_tree_C45"


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
    将数据集dataset在维度dim上进行划分，每一个取值对应一个分支
    @param dataset numpy.array
    '''
    vals = set(dataset[:, dim])
    subs = {}
    for val in vals:
        subs[val] = dataset[dataset[:, dim] == val]

    return subs


def chooseBestFeatureAndSplit(dataset, featIdxList, dtType):
    '''
    选择最佳的划分特征，即信息增益(ID3)或信息增益比(C45)最大的特征，然后进行划分
    @param dataset numpy.array
    @param featIdxList list 还未参与分裂的特征索引
    @param dtType str 决策树类型: ID3 / C4.5
    '''
    before_split_entropy = calcShannonEntropy(dataset)
    best_info_gain = 0.0
    best_dim = -1
    best_split = {}

    for dim in featIdxList:
        after_split_entropy = 0.0
        subs = splitDataSet(dataset, dim)
        for val in subs.keys():
            prob = len(subs[val]) / float(dataset.shape[0])
            after_split_entropy += prob * calcShannonEntropy(subs[val])

        cur_info_gain = before_split_entropy - after_split_entropy
        if dtType == C45:
            # use information gain ratio for C45
            cur_info_gain /= before_split_entropy

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
    labels = dataset[:, -1]
    for label in set(labels):
        label2Count[label] = np.count_nonzero(labels == label)
    sortedLabel2Count = sorted(label2Count.iteritems(),
            key=operator.itemgetter(1), reverse=True)
    return sortedLabel2Count[0][0]


def createDecisionTree(dataset, featIdxList, dtType=ID3):
    '''
    创建决策树
    @param dataset numpy.array
    @param featIdxList list 
    '''

    # 样本集中的标签已经统一了
    if dataset.shape[0] == np.count_nonzero(dataset[:, -1] == dataset[0][-1]):
        return {'leaf': True, 'dataset': dataset, 'cls': dataset[0][-1]}

    # 样本集的所有特征都已经检查过了，但是标签还不统一，需要进行多数票决
    if len(featIdxList) == 0:
        return {'leaf': True, 'dataset': dataset, 'cls': majorityVote(dataset)}

    # 构建当前树
    bestDim, bestSplit = chooseBestFeatureAndSplit(dataset, featIdxList, dtType)
    curTree = {'splitFeat': bestDim, 'dataset': dataset,
            'cls': majorityVote(dataset), 'branches': {}}
    featIdxList.remove(bestDim)

    # 构建子树
    for featval, subset in bestSplit.iteritems():
        curTree['branches'][featval] = createDecisionTree(subset, featIdxList[:], dtType)

    return curTree


def classify(decisionTree, testData):
    '''
    使用决策树decisionTree对测试样本testData进行分类
    '''
    # 获取当前节点的预测类别
    testLabel = decisionTree.get('cls', 'unknown')

    # 如果到达叶子节点直接返回
    if decisionTree.get('leaf', False):
        return testLabel

    # 沿着分支继续前进
    branches = decisionTree.get('branches', {})
    splitFeat = decisionTree.get('splitFeat', -1)

    for featVal in branches.keys():
        if testData[splitFeat] == featVal:
            testLabel = classify(branches[featVal], testData)

    return testLabel


def pruneDT(node, alpha):
    '''
    剪枝
    '''
    # 如果到达叶子节点就计算自己的分类损失
    if node.get('leaf', False):
        # leaf node
        dataset = node['dataset']
        return dataset.shape[0] * calcShannonEntropy(dataset) + alpha

    # 如果不是叶子节点就沿着各个分支继续前进
    branches = node['branches']
    costBefore = 0.0
    leafNum = 0
    for featVal in branches.keys():
        leafCost = pruneDT(branches[featVal], alpha)
        if leafCost is not None:
            costBefore += leafCost
            leafNum += 1

    # 如果所有子节点都变成了叶子节点则继续修剪该节点
    if leafNum == len(branches.keys()):
        dataset = node['dataset']
        costAfter = dataset.shape[0] * calcShannonEntropy(dataset) + alpha
        # 如果修剪后分类的损失减小则修剪该节点，否则当前分支停止修剪
        if costAfter < costBefore:
            # prune current node
            node['leaf'] = True
            del node['branches']
            del node['splitFeat']
            return costAfter

    return None

def loadCarData():
    '''
    数据来源：Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
    [Car Evaluation] https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    '''
    dataset = []
    featNames = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    with open('./car/car.data') as df:
        for line in df:
            fields = line.strip().split(',')
            dataset.append(fields)
    return np.array(dataset), featNames


def testDT(alpha=0.):
    '''
    测试决策树
    '''
    dataSet, featNames = loadCarData()
    dataIdxList = range(dataSet.shape[0])
    featIdxList = range(dataSet.shape[1] - 1)
    np.random.shuffle(dataIdxList)
    trainCount = len(dataIdxList) * 2 / 3
    trainIdxList = dataIdxList[0:trainCount]
    testIdxList = dataIdxList[trainCount:]
    trainDataSet = dataSet[trainIdxList]
    testDataSet = dataSet[testIdxList]

    dt = createDecisionTree(trainDataSet, featIdxList, ID3)
    pruneDT(dt, alpha)
    errnum = 0.
    for testData in testDataSet:
        cls = classify(dt, testData[:-1])
        if cls != testData[-1]:
            errnum += 1
    errRate = errnum / testDataSet.shape[0]
    print ('test count is %d\nerror count is %d\nerror rate is %f') % \
            (testDataSet.shape[0], errnum, errRate)
    return errRate


if __name__ == '__main__':
    errRate = 0.
    for i in range(100):
        errRate += testDT(0.)
    print ('average error rate is %f') % (errRate / i)

