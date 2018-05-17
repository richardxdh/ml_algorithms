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

<center><a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;g(D,A)&space;=&space;H(D)&space;-&space;H(D|A)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\small&space;g(D,A)&space;=&space;H(D)&space;-&space;H(D|A)" title="\small g(D,A) = H(D) - H(D|A)" /></a></center>

### 信息增益比

以信息增益作为划分训练数据集的特征，存在偏向于选择取值较多的特征的问题，使用信息增益比可以对这一问题进行校正；特征A对训练数据集D的信息增益比<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;g_R(D,A)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\small&space;g_R(D,A)" title="\small g_R(D,A)" /></a>定义为其信息增益<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;g(D,A)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;g(D,A)" title="\small g(D,A)" /></a>与训练数据集D关于特征A的值的熵<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;H_A(D)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;H_A(D)" title="\small H_A(D)" /></a>之比，即

<center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;$$g_R(D,A)&space;=&space;\cfrac{g(D,A)}{H_A(D)}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;$$g_R(D,A)&space;=&space;\cfrac{g(D,A)}{H_A(D)}$$" title="\large $$g_R(D,A) = \cfrac{g(D,A)}{H_A(D)}$$" /></a></center> 

其中，<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;$$H_A(D)&space;=&space;-\sum_{i=1}^n\cfrac{\left|D_i\right|}{\left|D\right|}\log_2\cfrac{\left|D_i\right|}{\left|D\right|}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;$$H_A(D)&space;=&space;-\sum_{i=1}^n\cfrac{\left|D_i\right|}{\left|D\right|}\log_2\cfrac{\left|D_i\right|}{\left|D\right|}$$" title="\small $$H_A(D) = -\sum_{i=1}^n\cfrac{\left|D_i\right|}{\left|D\right|}\log_2\cfrac{\left|D_i\right|}{\left|D\right|}$$" /></a>，n是特征A取值的个数。


### 创建决策树

#### 1.)在特征空间上选择信息增益（ID3）或者信息增益比（C45）最大的特征作为当前分裂特征；
#### 2.)为分裂特征上的每个取值创建一个分支；将当前特征从特征空间移除后作为分支的特征空间，在每个分支上执行步骤1，直到遇到结束条件（特征空间已经为空，或者当前分支的样本都属于同一类别......）执行步骤3；
#### 3.)停止分裂，创建叶结点，将当前分支对应的数据集中占比最大的类别作为叶结点的类别；

### 决策树剪枝

#### 在给定参数Alpha的前提下，从叶结点开始，计算每个节点剪枝前后的损失；如果剪枝能够减少损失，则剪，否则不剪；

### CART

#### CART决策树是既可以分类又可以回归；它跟ID3和C45最大的区别是:
##### 1.)CART是二叉树；
##### 2.)生成回归树时分裂的条件是以分裂特征取值大于和小于等于分裂值分裂的；
##### 3.)生成分类树时分裂条件是以分裂特征取值等于和不等于分裂值分裂的；
##### 4.)剪枝时没有指定的ALPHA，需要根据数据自己找出最佳ALPHA值和对应的子树;具体做法就是从原始生成树开始，ALPHA值从0开始增长，找出所有子树的序列；然后通过交叉验证找出最佳子树；


