# -*- coding: utf8 -*-
import numpy as np


def createVocabList(dataSet):
    '''
    根据训练文档集生成词汇表
    '''
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)


def doc2Vec(vocabList, doc):
    '''
    根据词汇表vocabList为文档doc生成特征向量
    '''
    docVec = [0] * len(vocabList)
    for word in doc:
        if word in vocabList:
            docVec[vocabList.index(word)] = 1
    return np.array(docVec)


def trainNB(trainMat, trainClass):
    '''
    使用样本训练模型，用于文档分类；
    @param trainMat numpy.array 样本向量数组
    @param trainClass numpy.array 样本向量对应的样本类型；（1，0）
    '''
    docCount = trainMat.shape[0]
    featCount = trainMat.shape[1]
    probClass1 = float(np.sum(trainClass)) / docCount

    # 使用平滑处理避免了先验概率为0影响后验概率的问题
    featClass0 = np.ones(featCount)
    numClass0 = 2.0
    featClass1 = np.ones(featCount)
    numClass1 = 2.0

    for i in range(docCount):
        docVec = trainMat[i]
        docClass = trainClass[i]

        if docClass == 0:
            featClass0 += docVec
            numClass0 += np.sum(docVec)

        else:
            featClass1 += docVec
            numClass1 += np.sum(docVec)

    # 使用对数防止下溢
    cprobFeatClass0 = np.log2(featClass0 / numClass0)
    cprobFeatClass1 = np.log2(featClass1 / numClass1)
    return cprobFeatClass0, cprobFeatClass1, probClass1


def classifyNB(docVec, cprobFeatClass0, cprobFeatClass1, probClass1):
    p0 = np.sum(docVec * cprobFeatClass0) + np.log2(1.0 - probClass1)
    p1 = np.sum(docVec * cprobFeatClass1) + np.log2(probClass1)

    if p1 > p0:
        return 1
    else:
        return 0


# 测试数据来自《机器学习实践》第4章
def textParse(emailString):
    import re
    tokenList = re.split(r'\W*', emailString)
    return [token.lower() for token in tokenList if len(token) > 2]


def spamTest():
    import random
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # class ham
        wordList = textParse(open('./email/ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)
        fullText.extend(wordList)

        # class spam
        wordList = textParse(open('./email/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)
        fullText.extend(wordList)

    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIdx = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIdx])
        del trainingSet[randIdx]

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(doc2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB(np.array(trainMat), np.array(trainClasses))

    errorCount = 0.0
    for docIndex in testSet:
        docVec = doc2Vec(vocabList, docList[docIndex])
        if classifyNB(docVec, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]

    print 'the error rate is ', errorCount/len(testSet)
