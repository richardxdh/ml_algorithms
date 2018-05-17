# -*- coding: utf8 -*-

import copy
import numpy as np
from sklearn import tree as SKTREE
from sklearn.externals.six import StringIO
from sklearn.datasets import load_iris
import graphviz

NODE = 'node'
LEAF = 'leaf'
CLF = "classifier"
REG = "regression"
DISCRETE = "discrete"
CONTINUOUS = "continuous"
COMPARATOR = {DISCRETE: np.equal, CONTINUOUS: np.less_equal}


def plotDTree(dTree, treeName):
    '''
    Visualize decision tree
    '''
    dot_data = StringIO()
    dot_data.write("graph dtree {\n")
    nodeList = [dTree]
    nodeIdxList = [0]
    while len(nodeList) > 0:
        latestIdx = nodeIdxList[-1]

        node = nodeList[0]
        del nodeList[0]
        nodeIdx = nodeIdxList[0]
        del nodeIdxList[0]

        if node['type'] == LEAF:
            nodeLabel = "gini: %.3f\nforecast: %d" % (node['gini'], node['fcVal'])
            dot_data.write("%d [label=\"%s\" shape=box]" % (nodeIdx, nodeLabel))
        else:
            opName = "<=" if node['comparator'] == np.less_equal else "=="
            nodeLabel = "X[%d]%s%.2f\ngini: %.3f" % (node['splitFeature'], 
                    opName, node['splitValue'], node['gini'],)
            dot_data.write("%d [label=\"%s\" shape=box]" % (nodeIdx, nodeLabel))

        for key in ['left', 'right']:
            branch = node.get(key, None)
            if branch:
                latestIdx += 1
                nodeList.append(branch)
                nodeIdxList.append(latestIdx)
                dot_data.write("%d -- %d\n" % (nodeIdx, latestIdx))
    dot_data.write("}")

    filename = "./trees/%s" % treeName 
    graph = graphviz.Source(dot_data.getvalue(), filename=filename, format="png")
    graph.view()


def giniImpurity(dl):
    '''
    Calculate Gini Impurity.

    Parameters
    ----------
    dl : array
        data label vector

    Returns
    ----------
    Gini Impurity which is calculated on dl.
    '''
    labelSet = set(dl)
    gini = 1.
    for label in labelSet:
        count = np.count_nonzero(dl == label)
        gini -= (float(count) / dl.size) ** 2
    return gini


def clfSplitPoint(df, dl, ft):
    '''
    Choose the best split point for classifier with gini impurity as metric.

    Parameters
    ----------
    df : array
        data feature matrix
    dl : array
        data label vector
    ft : list
        feature type (DISCRETE / CONTINUOUS)

    Returns
    ----------
    splitFeature : int
        Index of feature which is the best one to be splitted.
    splitValue : float
        It is the best split point on feature splitFeature.
    minGini : float
        gini impurity after splitting
    '''
    minGini = np.inf
    splitFeature = -1
    splitValue = -1
    splitLeftIdxList = None
    splitRightIdxList = None
    m, n = df.shape
    idxList = np.arange(m)
    for feat in range(n):
        comparator = COMPARATOR[ft[feat]]
        for val in set(df[:, feat]):
            curGini = 0.

            # left branch
            leftIdxList = np.where(comparator(df[:, feat], val))[0]
            if leftIdxList.size > 0:
                curGini += giniImpurity(dl[leftIdxList]) * leftIdxList.size

            # right branch
            rightIdxList = np.array([i for i in idxList if i not in leftIdxList])
            if rightIdxList.size > 0:
                curGini += giniImpurity(dl[rightIdxList]) * rightIdxList.size

            if curGini < minGini:
                minGini = curGini
                splitFeature = feat
                splitValue = val
                splitLeftIdxList = leftIdxList
                splitRightIdxList = rightIdxList

    if ft[splitFeature] == CONTINUOUS:
        # optimize split point
        valList = np.sort(list(set(df[:, splitFeature])))
        valIdx = np.where(valList == splitValue)[0][0]
        # case 1
#        midVal = (np.min(valList) + np.max(valList)) / 2
#        if splitValue < midVal:
#            splitValue = (splitValue + valList[valIdx + 1]) / 2
#        else:
#            splitValue = (splitValue + valList[valIdx - 1]) / 2

        # case 2
        if valIdx < len(valList):
            splitValue = (splitValue + valList[valIdx + 1]) / 2

    return splitFeature, splitValue, minGini, comparator, splitLeftIdxList, splitRightIdxList


def clfCreateNode(df, dl, ft):
    '''
    Create tree node for classifier.

    Parameters
    ----------
    df : array
        data feature matrix
    dl : array
        data label vector
    ft : list
        feature type (DISCRETE / CONTINUOUS)

    Returns
    ----------
    node : dict
        A dictionary represents tree node or leaf.
        {
        'type': string node / leaf
        'gini': float gini impurity
        'fcVal' : int forecasted class
        'splitFeature' : int splitting feature
        'splitValue' : float splitting value
        'comparator' : operator on splitFeature
        'left' : node contains data which splitFeature is equal to splitValue
        'right': node contains data which splitFeature is not equal to splitValue
        'giniError' : float decrease of impurity after splitting, used for pruning
        }
    '''
    node = {'type': LEAF}
    node['gini'] = giniImpurity(dl)
    # set the most label as current node label
    labels = list(set(dl))
    countList = [np.count_nonzero(dl == label) for label in labels]
    node['fcVal'] = labels[np.argmax(countList)]

    # check end condition (implement prepruning here)
    if len(labels) == 1:
        return node

    # create subnodes
    node['type'] = NODE
    splitFeature, splitValue, splitGini, comparator, leftIdxList, \
            rightIdxList = clfSplitPoint(df, dl, ft)
    node['splitFeature'] = splitFeature
    node['splitValue'] = splitValue
    node['comparator'] = comparator
    node['giniError'] = node['gini'] * dl.shape[0] - splitGini

    # create left subnode
    if leftIdxList.size > 0:
        node['left'] = clfCreateNode(df[leftIdxList], dl[leftIdxList], ft)

    # create right subnode
    if rightIdxList.size > 0:
        node['right'] = clfCreateNode(df[rightIdxList], dl[rightIdxList], ft)

    return node


def findLeafParents(node, parentNodes):
    '''
    查找carTree上所有叶子结点的父节点
    '''
    left = node['left']
    right = node['right']

    if left['type'] == LEAF and right['type'] == LEAF:
        parentNodes.append(node)
        return

    if left['type'] == NODE:
        findLeafParents(node.get('left'), parentNodes)

    if right['type'] == NODE:
        findLeafParents(node.get('right'), parentNodes)


def pruneOneBranch(carTree):
    '''
    对分类树进行剪枝，剪去carTree上误差增益最小的分支
    '''
    leafParents = []
    findLeafParents(carTree, leafParents)
    alpha = np.inf
    prunedNode = None
    for parentNode in leafParents:
        giniError = parentNode.get('giniError')
        if giniError < alpha:
            alpha = giniError
            prunedNode = parentNode
    if prunedNode:
        prunedNode['type'] = LEAF
        del prunedNode['splitFeature']
        del prunedNode['splitValue']
        del prunedNode['comparator']
        del prunedNode['left']
        del prunedNode['right']
        del prunedNode['giniError']


def genSubTreeList(carTree, saveTree=False):
    '''
    剪枝，生成子树列表
    '''
    treeList = [carTree]
    subTree = carTree
    while subTree['type'] != LEAF:
        subTree = copy.deepcopy(subTree)
        pruneOneBranch(subTree)
        treeList.append(subTree)
 
    if saveTree:
        for i in range(len(treeList)):
            plotDTree(treeList[i], "tree%d" % i)

    return treeList


def regSplitPoint(df, dl, fs):
    '''
    Choose the best split point for regression with the least square error as metric.

    Parameters
    ----------
    df : array
        data feature matrix
    dl : array
        data label vector
    fs : list
        feature space vector

    Returns
    ----------
    splitFeature : int
        Index of feature which is the best one to be splitted.
    splitValue : float
        It is the best split point on feature splitFeature.
    '''
    squareError = np.inf
    splitFeature = -1
    splitValue = -1
    for feat in fs:
        for val in set(df[:, feat]):
            curSqureError = 0.
            # greater than branch
            idxList = np.where(df[:, feat] > val)
            if idxList.size > 0:
                curSqureError += np.var(dl[idxList]) * idxList.size
            # less than or equal branch
            idxList = np.where(df[:, feat] <= val)
            if idxList.size > 0:
                curSqureError += np.var(dl[idxList]) * idxList.size

            if curSqureError < squareError:
                squareError = curSqureError
                splitFeature = feat
                splitValue = val
    return splitFeature, splitValue


def regCreateNode(df, dl, fs):
    '''
    Create tree node for regression.

    Parameters
    ----------
    df : array
        data feature matrix
    dl : array
        data label vector
    fs : list
        feature space vector

    Returns
    ----------
    node : dict
        A dictionary represents tree node or leaf.
        {
        'type': string node / leaf
        'fcVal' : int forecasted class
        'splitFeature' : int splitting feature
        'splitValue' : float splitting value
        'left' : node contains data which splitFeature is greater than splitValue
        'right': node contains data which splitFeature is less than or equal to splitValue
        }

    '''
    node = {'type': LEAF}
    # set label mean as current forecasted value
    node['fcVal'] = np.mean(dl)

    # check end condition
    if fs.size == 0 or np.var(dl) == 0.:
        return node

    node['type'] = NODE
    # choose the best splitting point
    splitFeature, splitValue = regSplitPoint(df, dl, fs)
    node['splitFeature'] = splitFeature
    node['splitValue'] = splitValue

    # create left subnode
    idxList = np.where(df[:, splitFeature] > splitValue)[0]
    if idxList.size > 0:
        node['left'] = regCreateNode(df[idxList], dl[idxList],
                fs[:].remove(splitFeature))

    # create right subnode
    idxList = np.where(df[:, splitFeature] <= splitValue)[0]
    if idxList.size > 0:
        node['right'] = regCreateNode(df[idxList], dl[idxList],
                fs[:].remove(splitFeature))

    return node


def createTree(dataFeats, dataLabels, featTypes, ctType=CLF):
    '''
    Create decision tree for different scenario, classifier or regression.
    '''
    if ctType == REG:
        createNode = regCreateNode
    else:
        createNode = clfCreateNode

    dTree = createNode(dataFeats, dataLabels, featTypes)
    return dTree


def forecastData(dTree, data):
    '''
    Using dTree to forecast data with correlative comparator, np.equal for classifier or np.greater for regression.
    '''
    point = dTree
    while True:
        if point is None:
            return None

        if point['type'] == LEAF:
            return point['fcVal']

        comparator = point['comparator']
        splitFeature = point['splitFeature']
        splitValue = point['splitValue']

        if comparator(data[splitFeature], splitValue):
            point = point.get('left', None)
        else:
            point = point.get('right', None)
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


def testCart():
    '''
    测试决策树
    '''
#   data, featNames = loadCarData()
#   featTypes = [DISCRETE] * 6
#   dataSet = data[:, :-1]
#   labels = data[:, -1]

    iris = load_iris()
    featTypes = [CONTINUOUS, CONTINUOUS, CONTINUOUS, CONTINUOUS]
    dataSet = iris.data
    labels = iris.target

    dataIdxList = list(range(dataSet.shape[0]))
    np.random.shuffle(dataIdxList)
    trainCount = int(len(dataIdxList) * 2 / 3)
    trainIdxList = dataIdxList[0:trainCount]
    testIdxList = dataIdxList[trainCount:]
    trainDataSet = dataSet[trainIdxList]
    trainLabels = labels[trainIdxList]
    testDataSet = dataSet[testIdxList]
    testLabels = labels[testIdxList]

    orgTree = createTree(trainDataSet, trainLabels, featTypes, CLF)
    # pruning
    treeList = genSubTreeList(orgTree)

    # choose the best one validating on testDataSet
    minErrTreeIndex = -1
    minErrRate = np.inf
    for index in range(len(treeList)):
        clfTree = treeList[index]
        errnum = 0.
        for i in range(testDataSet.shape[0]):
            testData = testDataSet[i]
            cls = forecastData(clfTree, testData)
            if cls != testLabels[i]:
                errnum += 1
        errRate = errnum / testDataSet.shape[0]
        if errRate < minErrRate:
            minErrRate = errRate
            minErrTreeIndex = index
    print ('minErrTreeIndex = %d' % minErrTreeIndex)
    print ('minErrRate is %f' % minErrRate)
    plotDTree(treeList[minErrTreeIndex], "bestTree")

    # compare with the decision tree built with sklearn.tree
    useSkDT(trainDataSet, trainLabels, testDataSet, testLabels)



def useSkDT(trainDataSet, trainLabels, testDataSet, testLabels):
    '''
    Using sklearn to build and test the decision tree with the same data
    '''
    skClf = SKTREE.DecisionTreeClassifier()
    skClf = skClf.fit(trainDataSet, trainLabels)

    dot_data = StringIO()
    SKTREE.export_graphviz(skClf, out_file=dot_data) 
    graph = graphviz.Source(dot_data.getvalue(), filename="./trees/skDTree", format="png")
    graph.view()

    predicts = skClf.predict(testDataSet)
    testCount = testDataSet.shape[0]
    skErrRate = float(np.count_nonzero(predicts != testLabels)) / testCount
    print ('sklearn error rate is %f' % skErrRate)


if __name__ == '__main__':
    pass
