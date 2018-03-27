# -*- coding: utf8 -*-
from knn import *
from plot import *
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
    bestK, k2Err = cvSelectBestK(orgDataset)
    print 'the best K for datingTestSet2.txt is %d' % bestK
    plotKVsErr(k2Err)


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

