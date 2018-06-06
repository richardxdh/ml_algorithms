# -*- coding: utf8 -*-


# get frequent item set
def createItemSetOne(ds):
    itemList = []
    for data in ds:
        for item in data:
            if [item] not in itemList:
                itemList.append([item])
    itemList.sort()
    return map(frozenset, itemList)


def createItemSetAddOne(itemSetList):
    newItemSetList = []
    if len(itemSetList) == 0:
        return newItemSetList

    isLL = len(itemSetList)
    for i in range(isLL):
        for j in range(i+1, isLL):
            beginningI = sorted(list(itemSetList[i])[:-1])
            beginningJ = sorted(list(itemSetList[j])[:-1])
            if beginningI == beginningJ:
                newItemSetList.append(itemSetList[i] | itemSetList[j])
    return newItemSetList


def scanDataSet(ds, itemSetList, minSupport):
    '''
    ds: data set, each data in ds is an isinstance of type set
    itemSetList: a list of item set to be checked on ds
    minSupport: threshold of minimum support
    '''
    itemSet2Total = {}
    for data in ds:
        for itemSet in itemSetList:
            if itemSet.issubset(data):
                itemSet2Total[itemSet] = itemSet2Total.get(itemSet, 0) + 1
 
    dataTotal = float(len(ds))
    reservedItemSetList = []
    itemSet2Support = {}
    for itemSet in itemSet2Total:
        support = itemSet2Total[itemSet] / dataTotal
        if support >= minSupport:
            reservedItemSetList.insert(0, itemSet)
            itemSet2Support[itemSet] = support
    return reservedItemSetList, itemSet2Support


def apriori(ds, minSupport=0.5):
    dss = map(set, ds)
    itemSetOne = createItemSetOne(ds)
    reservedItemSetList, itemSet2Support = scanDataSet(dss, itemSetOne, minSupport)
    frequentItemSetList = [reservedItemSetList] # list of reservedItemSetList
    while len(frequentItemSetList[-1]) > 0:
        itemSetKList = createItemSetAddOne(frequentItemSetList[-1])
        reservedList, reservedSupport = scanDataSet(dss, itemSetKList, minSupport)
        itemSet2Support.update(reservedSupport)
        frequentItemSetList.append(reservedList)
    return frequentItemSetList, itemSet2Support


# find the association rules from frequent item set

def calcConfidence(freqSet, conseqList, support, ruleList, minConf=0.7):
    reservedConseqList = []
    for conseq in conseqList:
        conf = support[freqSet] / support[freqSet - conseq]
        if conf > minConf:
            print (freqSet-conseq,"-->",conseq,"conf: ",conf)
            ruleList.append((freqSet - conseq, conseq, conf))
            reservedConseqList.append(conseq)
    return reservedConseqList


def rulesFromConseq(freqSet, conseqList, support, ruleList, minConf=0.7):
    m = len(conseqList[0])
    if len(freqSet) > (m + 1):
        conseqListP1 = createItemSetAddOne(conseqList)
        conseqListP1 = calcConfidence(freqSet, conseqListP1, support, ruleList, minConf)
        if len(conseqListP1) > 1:
            rulesFromConseq(freqSet, conseqListP1, support, ruleList, minConf)


def genRules(freqSetList, support, minConf=0.7):
    ruleList = []
    # skip frequent item set containing only one item
    for i in range(1, len(freqSetList)):
        for freqSet in freqSetList[i]:
            conseqList = [frozenset([item]) for item in freqSet]

            # my solution
#            while conseqList and len(freqSet) > len(conseqList[0]):
#                conseqList = calcConfidence(freqSet, conseqList, support, ruleList, minConf)
#                conseqList = createItemSetAddOne(conseqList)

            # solution from MLiA
            if i > 1:
                rulesFromConseq(freqSet, conseqList, support, ruleList, minConf)
            else:
                calcConfidence(freqSet, conseqList, support, ruleList, minConf)
    return ruleList


if __name__ == '__main__':
    ds = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    L, S = apriori(ds)
    print (L)
    print (S)
    rules = genRules(L, S)
    print (rules)
