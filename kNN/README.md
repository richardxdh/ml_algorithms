
#### kNN算法

k近邻算法根据样本数据中距离测试数据最近的k个实例，对测试数据进行分类或回归;
1. 分类可以通过多数票表决等方式对测试数据的类别做出预测;
2. 回归可以通过加权计算进行预测；（可以将样本与测试数据的距离作为权重）

关于距离度量的方法可以看[这里](https://my.oschina.net/hunglish/blog/787596)或[这里](https://blog.csdn.net/guoziqing506/article/details/51779536)

#### 为了提高kNN算法的性能，需要对数据进行结构化处理；比较常见的有kd树，它分为三个部分：
1. 构建KD树；（knn.py中的buildKDTree）
2. 搜索KD树，找出与测试数据最近的K个样本数据；（knn.py中的searchKDTree）
3. 根据K个邻居对测试数据进行分类或者回归；（多数票表决，加权计算）

#### 代码说明

knn.py用kd树实现kNN算法；包括构建kd树，搜索，数据标准化，交叉验证，多数票表决分类;

plot.py包含了所有的图示方法;

demo.py包含了测试方法：

1. demoRandomly

    随机生成样本数据和一个测试数据，使用kNN根据样本对测试数据进行分类

    ![demoRandomly](https://github.com/richardxdh/ml_algorithms/blob/master/kNN/imgs/demoRandomly.png)

2. demoDatingTestSet

    通过交叉验证的方式，找出最佳K；

    ![cross_valide_select_k](https://github.com/richardxdh/ml_algorithms/blob/master/kNN/imgs/cross_validate_select_k.png)

3. predictDatingPerson

    根据输入数据进行分类

    frequent flier miles earned per year?10000

    percentage of time spent playing video games?10

    liters of ice cream consumed per year?0.5

    predict label is in small doses

4. classifyDigit

    根据trainingdigits中的数据构建kd树，对testdigits下的数据进行分类

