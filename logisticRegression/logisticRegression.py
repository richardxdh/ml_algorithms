# -*- coding: utf8 -*-

# ref: https://www.coursera.org/learn/machine-learning/resources/Zi29t

import matplotlib.pyplot as plt
import numpy as np


# load testSet.txt
def loadDataSet():
    '''
    load data from testSet.txt
    '''
    dataMat = []
    labelMat = []
    with open('./testSet.txt', 'r') as df:
        for line in df:
            fields = line.strip().split()
            dataMat.append([1.0, float(fields[0]), float(fields[1])])
            labelMat.append(int(fields[2]))
    return np.array(dataMat), np.array(labelMat)


def plotDataSet(dataMat, labelMat, thetA=None):
    '''
    plot data loaded from testSet.txt
    '''
    data0 = dataMat[np.where(labelMat == 0)]
    data1 = dataMat[np.where(labelMat == 1)]

    plt.plot(data0[:, 1], data0[:, 2], 'g.')
    plt.plot(data1[:, 1], data1[:, 2], 'r.')

    if thetA is not None:
        # plot decision boundary
        x = np.arange(-3.0, 3.0, 0.1)
        y = -(thetA[0] + thetA[1] * x) / thetA[2]
        plt.plot(x, y, 'b-')

    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.show()


def loadHorseColicData():
    '''
    load data from horseColicTraining.txt
    '''
    dataMat = []
    labelMat = []
    with open('./horseColicTraining.txt', 'r') as df:
        for line in df:
            fields = line.strip().split()
            dataMat.append([float(field) for field in fields[0:-1]])
            labelMat.append(float(fields[-1]))
    return np.array(dataMat), np.array(labelMat)


# algorithm
def sigmoidFunc(z):
    return 1.0 / (1.0 + np.exp(-z))


def costFunction(thetA, dataMat, labelMat):
    H = sigmoidFunc(np.dot(dataMat, thetA))
    # to avoid issue "RuntimeWarning: divide by zero encountered in log"
    H[H == 0.0] += 10**-10
    H[H == 1.0] -= 10**-10
    m = dataMat.shape[0]
    cost = -(np.dot(labelMat.T,  np.log(H)) +
            np.dot((1 - labelMat).T, np.log(1 - H))) / m
    return cost


def gradientDescent(thetA, alpha, dataMat, labelMat):
    H = sigmoidFunc(np.dot(dataMat, thetA))
    m = dataMat.shape[0]
    thetA = thetA - alpha * np.dot(dataMat.T, H - labelMat) / m
    return thetA


def plotCostVsIter(costVsIter, alpha):
    plt.figure()
    plt.title('Cost Value VS Iteration Number, ALPHA = %s' % alpha)
    plt.plot(costVsIter[:, 0], costVsIter[:, 1], 'g-')
    plt.xlabel('Iteration Number')
    plt.ylabel('Cost Value')
    plt.show()


def trainLR0(dataMat, labelMat, alpha=0.03, costThreshold=0.18, cycles=50000, plot=False):
    '''
    Train logistic regression model using gradient descent.
    '''
    thetA = np.zeros(dataMat.shape[1])
    k = 0
    costVsIter = []
    while cycles <= 0 or k < cycles:
        k += 1
        cost = costFunction(thetA, dataMat, labelMat)
        costVsIter.append([k, cost])
        #print 'theta: %s, cost: %s' % (thetA, cost)
        if cost < costThreshold:
            print 'cost is less than costThreshold'
            break
        thetA = gradientDescent(thetA, alpha, dataMat, labelMat)
    print '%d cycles, thetA is %s, cost is %s' % (k, thetA, cost)

    if plot:
        # cost value versus iteration number
        plotCostVsIter(np.array(costVsIter), alpha)

    return thetA


# stochastic gradient descent
def plotSGDThetaVsIter(thetaIter):
    plt.figure()
    ax = plt.subplot(311)
    ax.plot(thetaIter[:, 0], thetaIter[:, 1], 'r-')
    ax = plt.subplot(312)
    ax.plot(thetaIter[:, 0],  thetaIter[:, 2], 'g-')
    ax = plt.subplot(313)
    ax.plot(thetaIter[:, 0], thetaIter[:, 3], 'b-')
    plt.show()


def trainLRSGD(dataMat, labelMat, cycles=150, plot=False):
    '''
    Train logistic regression model using stochastic gradient descent.
    '''
    m, n = dataMat.shape
    thetA = np.zeros(n)

    thetaIter = []
    for k in range(cycles):
        dataIdxList = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + k + i) + 0.01
            idx = int(np.random.uniform(0, len(dataIdxList)))
            dataIdx = dataIdxList[idx]
            h = sigmoidFunc(np.dot(thetA, dataMat[dataIdx]))
            thetA = thetA - alpha * (h - labelMat[dataIdx]) * dataMat[dataIdx]
            del dataIdxList[idx]

            thetaIter.append([k*200+i, thetA[0], thetA[1], thetA[2]])
    thetaIter = np.array(thetaIter)

    if plot:
        # THETA vesus ITERATION NUMBER for one pass through the dataset
        plotSGDThetaVsIter(thetaIter)

        # plot decision boundary
        plotDataSet(dataMat, labelMat, thetA)

    return thetA


def classifySample(thetA, sampleData):
    '''
    Classify sampleData using logistic regression model with coefficient thetA.
    '''
    h = sigmoidFunc(np.dot(thetA, sampleData))
    if h > 0.5:
        return 1.0
    else:
        return 0.0


def horseColicTest(thetA):
    '''
    Test model thetA with data loaded from horseColicTest.txt and calculate error rate.
    '''
    err = 0.0
    testCount = 0.0
    with open('./horseColicTest.txt', 'r') as df:
        for line in df:
            fields = line.strip().split()
            testMat = np.array([float(item) for item in fields[0:-1]])
            testCount += 1.0
            if classifySample(thetA, testMat) != float(fields[-1]):
                err += 1.0
    errorRate = err / testCount
    print 'the error rate for this test is %s' % (errorRate,)


def horseColicSample():
    '''
    load training data -> train model -> test model
    '''
    # load training data
    dataMat, labelMat = loadHorseColicData()

    # train model

    # GRADIENT DESCENT
    # thetA = trainLR0(dataMat, labelMat, alpha=0.001, costThreshold=0.53)

    # STOCHASTIC GRADIENT DESCENT
    thetA = trainLRSGD(dataMat, labelMat)

    # load test dataset and test
    horseColicTest(thetA)


# use optimization function
def gradCostFunction(thetA, dataMat, labelMat):
    H = sigmoidFunc(np.dot(dataMat, thetA))
    m = dataMat.shape[0]
    return np.dot(dataMat.T, H - labelMat) / m


def optimizeWithTools():
    from scipy import optimize
    dataMat, labelMat = loadHorseColicData()

    args = (dataMat, labelMat)
    thetA = np.zeros(dataMat.shape[1])
    # use fmin_cg
    #thetA = optimize.fmin_cg(costFunction, thetA, fprime=gradCostFunction, args=args)
    thetA = optimize.fmin_bfgs(costFunction, thetA, fprime=gradCostFunction, args=args)

    # test horse colic
    horseColicTest(thetA)


# regularization
def costFunctionReg(thetA, dataMat, labelMat, lamb):
    H = sigmoidFunc(np.dot(dataMat, thetA))
    # to avoid issue "RuntimeWarning: divide by zero encountered in log"
    H[H == 0.0] += 10**-10
    H[H == 1.0] -= 10**-10
    m = dataMat.shape[0]
    cost = -(np.dot(labelMat.T,  np.log(H)) + \
            np.dot((1 - labelMat).T, np.log(1 - H))) / m + \
                    lamb * np.sum(thetA[1:]**2) / (2*m) # for regularization
    return cost


def gradientDescentReg(thetA, alpha, dataMat, labelMat, lamb):
    H = sigmoidFunc(np.dot(dataMat, thetA))
    m = dataMat.shape[0]
    tempThetA = thetA - alpha * np.dot(dataMat.T, H - labelMat) / m
    # for regularization
    tempThetA[1:] -= alpha * lamb * thetA[1:] / m
    thetA = tempThetA
    return thetA


