# -*- coding: utf8 -*-

import numpy as np
import matplotlib.pyplot as plt


def loadDS(dsPath):
    ds = []
    with open(dsPath) as dsFile:
        for line in dsFile:
            ds.append(line.strip().split())
    return np.array(ds, dtype=float)


def calcEuclideanDist(d1, d2):
    diff = d1 - d2
    return np.sqrt(np.dot(diff, diff))


def chooseClosestCentroid(d, centroids):
    k = centroids.shape[0]
    minIdx = -1
    minDist = np.inf
    for i in range(k):
        dist = calcEuclideanDist(d, centroids[i])
        if dist < minDist:
            minIdx = i
            minDist = dist
    return minIdx, minDist


def randCentroids(ds, k):
    m, n = ds.shape
    minFeats = np.min(ds, axis=0)
    maxFeats = np.max(ds, axis=0)
    diffFeats = maxFeats - minFeats
    centroids = np.zeros((k, n))
    for i in range(k):
        centroids[i] = minFeats + diffFeats * np.random.random(n)
    return centroids


def plotClusters(ds, centroids, clusterInfo):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    marks = ['bs', 'g*', 'ro', 'cp', 'mD']
    for i in range(centroids.shape[0]):
        d = ds[clusterInfo[:, 0] == i]
        ax.plot(d[:, 0], d[:, 1], marks[i % len(marks)])
        ax.plot(centroids[i][0], centroids[i][1], 'k+', linewidth=3, markersize=15)
    plt.show()


def kMeans(ds, k):
    m, n = ds.shape

    # initialize centroids
    centroids = randCentroids(ds, k)

    # cluster information: [centroid index, distance to centroid]
    clusterInfo = np.zeros((m, 2))
    clusterInfo[:, 0] = -1
    while True:
        changed = False
        # update cluster information
        for i in range(m):
            minIdx, minDist = chooseClosestCentroid(ds[i], centroids)
            if minIdx != clusterInfo[i][0]:
                clusterInfo[i] = minIdx, minDist ** 2
                changed = True

        if changed is False:
            break

        # recalculate centroids
        for j in range(k):
            centroids[j] = np.mean(ds[clusterInfo[:, 0] == j], axis=0)

    return centroids, clusterInfo


def testKM():
    ds = loadDS('./testSet.txt')
    K = 4
    ct, ci = kMeans(ds, K)
    plotClusters(ds, ct, ci)


def biKMeans(ds, K):
    m, n = ds.shape
    centroids = [np.mean(ds, axis=0), ]
    clusterInfo = np.zeros((m, 2))
    clusterInfo[:, 1] = np.array([calcEuclideanDist(d, centroids[0]) ** 2 for d in ds])

    while len(centroids) < K:
        splitIdx = -1
        lowestSSE = np.inf

        for idx in range(len(centroids)):
            sseNotSplit = np.sum(clusterInfo[clusterInfo[:, 0] != idx][:, 1])

            splitDS = ds[clusterInfo[:, 0] == idx]
            splitCT, splitCI = kMeans(splitDS, 2)
            sseSplit = np.sum(splitCI[:, 1])

            if sseNotSplit + sseSplit < lowestSSE:
                lowestSSE = sseNotSplit + sseSplit
                splitIdx = idx
                bestCentroids = splitCT
                bestClusterInfo = splitCI

        if splitIdx != -1:
            bestClusterInfo[np.where(bestClusterInfo[:, 0] == 1), 0] = len(centroids)
            bestClusterInfo[np.where(bestClusterInfo[:, 0] == 0), 0] = splitIdx
            centroids[splitIdx] = bestCentroids[0]
            centroids.append(bestCentroids[1])
            clusterInfo[np.where(clusterInfo[:, 0] == splitIdx)] = bestClusterInfo

    return np.array(centroids), np.array(clusterInfo)


def testBKM():
    ds = loadDS('./testSet.txt')
    K = 4
    ct, ci = biKMeans(ds, K)
    plotClusters(ds, ct, ci)
