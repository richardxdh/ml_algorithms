### 决策树

叶节点表示数据类别，非叶节点表示数据特征；

从根节点到叶节点的路径表示一个完整的决策过程，最终的结果就是对应叶节点表示的类别；

### 信息

信息是用来消除随机不确定性地；如果待分类的事物存在多个类别，则某个类别（<a href="https://www.codecogs.com/eqnedit.php?latex=x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i" title="x_i" /></a>）的信息定义如下；<a href="https://www.codecogs.com/eqnedit.php?latex=$p(x_i)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$p(x_i)$" title="$p(x_i)$" /></a>表示<a href="https://www.codecogs.com/eqnedit.php?latex=x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i" title="x_i" /></a>的概率；

<center><a href="https://www.codecogs.com/eqnedit.php?latex=\LARGE&space;$$l(x_i)&space;=&space;-\log_2p(x_i)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\LARGE&space;$$l(x_i)&space;=&space;-\log_2p(x_i)$$" title="\LARGE $$l(x_i) = -\log_2p(x_i)$$" /></a></center>

### 熵

熵是表示随机变量不确定性的度量，取值为信息的数学期望；只依赖于随机变量的分布，与随机变量的取值无关；

<center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;$$H&space;=&space;-\sum_{i=1}^np(x_i)\log_2p(x_i)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;$$H&space;=&space;-\sum_{i=1}^np(x_i)\log_2p(x_i)$$" title="\large $$H = -\sum_{i=1}^np(x_i)\log_2p(x_i)$$" /></a></center>

### 信息增益

在划分数据前后信息的变化叫做信息增益；即经验熵与经验条件熵之差；

<center><font size=5 color=aaaaaa>g(D,A) = H(D) - H(D|A)</font></center> 

### 信息增益比

以信息增益作为划分训练数据集的特征，存在偏向于选择取值较多的特征的问题，使用信息增益比可以对这一问题进行校正；特征A对训练数据集D的信息增益比$g_R(D,A)$定义为其信息增益g(D,A)与训练数据集D关于特征A的值的熵$H_A(D)$之比，即

<center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;$$g_R(D,A)&space;=&space;\cfrac{g(D,A)}{H_A(D)}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;$$g_R(D,A)&space;=&space;\cfrac{g(D,A)}{H_A(D)}$$" title="\large $$g_R(D,A) = \cfrac{g(D,A)}{H_A(D)}$$" /></a></center> 

其中，<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;$$H_A(D)&space;=&space;-\sum_{i=1}^n\cfrac{\left|D_i\right|}{\left|D\right|}\log_2\cfrac{\left|D_i\right|}{\left|D\right|}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;$$H_A(D)&space;=&space;-\sum_{i=1}^n\cfrac{\left|D_i\right|}{\left|D\right|}\log_2\cfrac{\left|D_i\right|}{\left|D\right|}$$" title="\small $$H_A(D) = -\sum_{i=1}^n\cfrac{\left|D_i\right|}{\left|D\right|}\log_2\cfrac{\left|D_i\right|}{\left|D\right|}$$" /></a>，n是特征A取值的个数。


### 创建决策树

- <font size=3>ID3</font>

#### createDecisionTreeID3

- <font size=3>C4.5</font>

to be continue ...

- <font size=3>CART</font>

to be continue ...

- <font size=3>Prune</font>

to be continue ...


