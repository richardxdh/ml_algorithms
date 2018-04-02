### 决策树

叶节点表示数据类别，非叶节点表示数据特征；

从根节点到叶节点的路径表示一个完整的决策过程，最终的结果就是对应叶节点表示的类别；

### 信息

信息是用来消除随机不确定性地；如果待分类的事物存在多个类别，则某个类别（$x_i$）的信息定义如下；$p(x_i)$表示$x_i$的概率；

<center><font size=5 color=aaaaaa>$$l(x_i) = -\log_2p(x_i)$$</font></center>

### 熵

熵是表示随机变量不确定性的度量，取值为信息的数学期望；只依赖于随机变量的分布，与随机变量的取值无关；

<center><font size=5 color=aaaaaa>$$H = -\sum_{i=1}^np(x_i)\log_2p(x_i)$$</font></center>

### 信息增益

在划分数据前后信息的变化叫做信息增益；即经验熵与经验条件熵之差；

<center><font size=5 color=aaaaaa>$$g(D,A) = H(D) - H(D|A)$$</font></center> 

### 信息增益比

以信息增益作为划分训练数据集的特征，存在偏向于选择取值较多的特征的问题，使用信息增益比可以对这一问题进行校正；特征A对训练数据集D的信息增益比$g_R(D,A)$定义为其信息增益g(D,A)与训练数据集D关于特征A的值的熵$H_A(D)$之比，即

<center><font size=5 color=aaaaaa>$$g_R(D,A) = \cfrac{g(D,A)}{H_A(D)}$$</font></center> 

其中，$$H_A(D) = -\sum_{i=1}^n\cfrac{\left|D_i\right|}{\left|D\right|}\log_2\cfrac{\left|D_i\right|}{\left|D\right|}$$，n是特征A取值的个数。


### 创建决策树

- <font size=3>ID3</font>

<center><font size=5 color=aaaaaa>$$createDecisionTreeID3$$</font></center> 

- <font size=3>C4.5</font>

to be continue ...

- <font size=3>CART</font>

to be continue ...

- <font size=3>Prune</font>

to be continue ...


