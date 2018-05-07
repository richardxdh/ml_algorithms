# -*- coding: utf8 -*-

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(dataFilePath):
    dataFeatures = []
    dataLabels = []
    with open(dataFilePath) as df:
        for line in df:
            fields = line.strip().split()
            dataFeatures.append(fields[:-1])
            dataLabels.append(fields[-1])
    return np.array(dataFeatures, dtype='float'), np.array(dataLabels, dtype='float')


def costFunc(theta, dataFeatures, dataLabels, lambd=0.1):
    h = np.dot(dataFeatures, theta) - dataLabels
    m = dataFeatures.shape[0]
    reg = lambd * np.dot(theta[1:].T, theta[1:])
    return (np.dot(h.T, h) + reg) / (2 * m)


def gradCostFunc(theta, dataFeatures, dataLabels, lambd=0.1):
    h = np.dot(dataFeatures, theta) - dataLabels
    m = dataFeatures.shape[0]
    reg = np.copy(theta)
    reg[0] = 0.
    return (np.dot(dataFeatures.T, h) + lambd * reg) / m


def gradientDescent(alpha=0.1, maxIter=3000, lambd=0.1, plot=False):
    df, dl = loadDataSet('./ex0.txt')
    theta = np.array([0.] * df.shape[1])
    minCost = np.inf
    bestTheta = theta

    costIterTrace = []
    for i in range(maxIter):
        theta -= alpha * gradCostFunc(theta, df, dl, lambd)
        curCost = costFunc(theta, df, dl, lambd)
        costIterTrace.append([i, curCost])

        if curCost < minCost:
            minCost = curCost
            bestTheta = theta

    if plot:
        costIterTrace = np.array(costIterTrace)
        plt.plot(costIterTrace[:, 0], costIterTrace[:, 1], 'r--')
        plt.show()

    print 'estimate error on traing data is %s' % costFunc(bestTheta, df, dl)
    return bestTheta


def minimizeCostFunc(lambd=0.1):
    import scipy.optimize as opt
    df, dl = loadDataSet('./ex0.txt')
    m, n = df.shape
    theta = opt.fmin_cg(costFunc, np.array([0.] * n), gradCostFunc, args=(df, dl, lambd))
    print 'estimate error on traing data is %s' % costFunc(theta, df, dl)
    return theta


def normEquation(lambd=0.1):
    df, dl = loadDataSet('./ex0.txt')
    reg = np.eye(df.shape[1])
    reg[0][0] = 0.
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(df.T, df) + lambd * reg), df.T), dl)
    print 'estimate error on traing data is %s' % costFunc(theta, df, dl)
    return theta


# ==============================
# LWLR: locally weighted linear regression
def lwlr(predicted, dataFeatures, dataLabels, k=1.0):
    m, n = dataFeatures.shape
    w = np.eye(m)
    for i in range(m):
        diff = predicted - dataFeatures[i]
        w[i][i] = np.exp(np.dot(diff.T, diff) / (-2.0 * k**2))

    theta = np.dot(np.linalg.pinv(np.dot(np.dot(dataFeatures.T, w),
                dataFeatures)),
                np.dot(np.dot(dataFeatures.T, w), dataLabels))
    return np.dot(predicted.T, theta)


def lwlrBatch(preX, dataFeatures, dataLabels, k=1.0):
    m, n = preX.shape
    preY = np.zeros(m)
    for i in range(m):
        preY[i] = lwlr(preX[i], dataFeatures, dataLabels, k)
    return preY


def lwlrTest():
    df, dl = loadDataSet('./ex0.txt')

    srtIdx = df[:, 1].argsort(0)
    preX = df[srtIdx]
    preY1 = lwlrBatch(preX, df, dl, 1.0)
    preY001 = lwlrBatch(preX, df, dl, 0.01)
    preY0003 = lwlrBatch(preX, df, dl, 0.003)
    preY0001 = lwlrBatch(preX, df, dl, 0.001)

    fig = plt.figure()

    ax = fig.add_subplot(221)
    ax.set_title('k = 1.0')
    ax.plot(df[:, 1], dl, 'b.')
    ax.plot(preX[:, 1], preY1, 'g-')

    ax = fig.add_subplot(222)
    ax.set_title('k = 0.01')
    ax.plot(df[:, 1], dl, 'b.')
    ax.plot(preX[:, 1], preY001, 'r-')

    ax = fig.add_subplot(223)
    ax.set_title('k = 0.003')
    ax.plot(df[:, 1], dl, 'b.')
    ax.plot(preX[:, 1], preY0003, 'y-')

    ax = fig.add_subplot(224)
    ax.set_title('k = 0.001')
    ax.plot(df[:, 1], dl, 'b.')
    ax.plot(preX[:, 1], preY0001, 'k-')

    plt.show()


# ==============================
# ridge regression
def ridgeRegression(df, dl, lambd=0.1):
    reg = np.eye(df.shape[1])
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(df.T, df) + lambd * reg), df.T), dl)
    print 'estimate error on traing data is %s' % costFunc(theta, df, dl)
    return theta


def ridgeTest():
    df, dl = loadDataSet('./abalone.txt')
    xMean = np.mean(df, 0)
    xVar = np.var(df, 0, ddof=1)
    xNorm = (df - xMean) / xVar
    yMean = np.mean(dl, 0)
    yNorm = dl - yMean

    numTestTimes = 30
    ws = np.zeros((numTestTimes, df.shape[1]))
    for i in range(numTestTimes):
        ws[i, :] = ridgeRegression(xNorm, yNorm, np.exp(i - 10))

    plt.plot(ws)
    plt.show()

    return ws, xMean, xVar, yMean

# ==============================
# stagewise linear regression
def stageWise(df, dl, eps=0.01, numIter=100):
    xMean = np.mean(df, 0)
    xVar = np.var(df, 0, ddof=1)
    xNorm = (df - xMean) / xVar
    yMean = np.mean(dl, 0)
    yNorm = dl - yMean

    m, n = xNorm.shape
    ws = np.zeros(n)
    wsMax = np.copy(ws)
    for i in range(numIter):
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = np.copy(ws)
                wsTest[j] += eps * sign
                yTest = np.dot(xNorm, wsTest)
                rssErr = rssError(yTest, yNorm)
                if rssErr < lowestError:
                    lowestError = rssErr
                    wsMax = wsTest
        ws = np.copy(wsMax)
    return ws


# ------------------------------
def normalizeData(dataFeatures):
    featMean = np.mean(dataFeatures, 0)
    featVar = np.var(dataFeatures, 0, ddof=1)
    return (dataFeatures - featMean) / featVar


def plotData(dataFeatures, dataLabels, theta=None):
    plt.plot(dataFeatures[:, 1], dataLabels, 'g.')
    if theta is not None:
        feats = np.copy(dataFeatures)
        feats = np.sort(feats, 0)
        predicted = np.dot(feats, theta)
        plt.plot(feats[:, 1], predicted, 'r-')
    plt.show()


def rssError(y1, y2):
    error = y1 - y2
    return np.dot(error.T, error)
