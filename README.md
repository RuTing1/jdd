##### This script is used to record my learning process of deep learning
# 1.神经网络调参技巧
-  **0.调参顺序** <br/>
   &nbsp;&nbsp;先调整学习率（一般开始设置0.1，然后除2或5），观察成本函数下降的速率，修正学习率，一方面快速下降，另一方面防止不收敛；<br/>
   &nbsp;&nbsp;mini-batch的大小；<br/>
   &nbsp;&nbsp;再调整隐藏层额数目；<br/>
   &nbsp;&nbsp;再调整隐藏层结点数目；逐渐增加，准确率理论上应该是先增大，后减小；<br/>
   &nbsp;&nbsp;固定参数，调整其他变量；<br/>
   &nbsp;&nbsp;一切稳定可以将激活函数换为ReLU函数再细调一下<br/>
- **1.神经网络的层数** <br/>
  &nbsp;&nbsp;首先使用最简单的网络结构：只包含输入层、输出层两层的神经网络;
  &nbsp;&nbsp;hidden_num一般设置为1就行，2层以上在简单的网络上只会到的适得其反的效果;<br/>
  &nbsp;&nbsp;训练集、验证集的个数要比较小，这样可以加快训练的速度，其他的参数先随便选择一个固定值; <br/>
- **2.每层神经元的个数** <br/>
  &nbsp;&nbsp;单隐藏层神经网络：视问题的复杂程度决定，简单线性可分，两个即可，复杂可以安排8个或者以上；<br/>
  
- **3.如何初始化Weights和biases** <br/>
  &nbsp;&nbsp;最简单的方法：让W和b服从N(0, 1 / sqrt(n_in) )，n_in：输入神经元的个数；<br/>
  &nbsp;&nbsp;设置合适的W和b可以加快学习的速率，在极个别的神经网络中，W和b甚至可以影响最后训练的准确度。<br/>
- **4.loss函数选择哪一个**  <br/>
  &nbsp;&nbsp;对于分类问题，我们最常见的损失函数依旧是SVM hinge loss和Softmax互熵损失<br/>
- **5.选择何种Regularization？L1,L2**  <br/>
  &nbsp;&nbsp;L1和L2是对cost函数后面增加一项; <br/> 
  &nbsp;&nbsp;L2即权值衰减; <br/> 
- **6.Regularization parameter lambda 选择多大合适**  <br/>
  &nbsp;&nbsp;实验lambda，从1.0,10,100…找到一个合适的; <br/>
  &nbsp;&nbsp;注意：lambda的大小和样本数成正比关系，如样本数、lambda分别是1000、0.4，当样本数为10000时，对应的lambda也要扩大到4。 <br/>
- **7.激励函数如何选择**  <br/>
  &nbsp;&nbsp;先使用tanh函数作为隐藏层的激活函数。虽然ReLU函数收敛快，但如果学习率或者隐藏层结点数目一开始设置不好，很容易产生震荡，无法收敛。相反tanh函数就稳定一点。 <br/>
  &nbsp;&nbsp;只能靠实验比较 <br/>
- **8.是否使用dropout** <br/>
  &nbsp;&nbsp;dropout（） 的参数可以从0.1开始调，以0.05为间隔，各个网络有不同的参数；所放的位置也有很大的影响，不一定要在全连接层后<br/>
- **9.训练集多大比较合适**  <br/>
- **10.mini-batch选择多大**  <br/>
  &nbsp;&nbsp;mini-batch选择太小：没有充分利用计算资源；<br/>
  &nbsp;&nbsp;太大：更新权重和偏向比较慢；<br/>
  &nbsp;&nbsp;mini-batch size和其他参数相对独立，一旦找到一个合适的以后，就不需要再改了。 <br/>
  &nbsp;&nbsp;iteration/step：每运行一个step,更新一次参数权重，进行一次学习，每次学习需要batch size个样本，假设有20000个样本，batch size为200，则 iteration=20000/200=100。 <br/>
- **11.学习率多少合适** <br/>
  &nbsp;&nbsp;优化器Adam的时候，lr=0.001; <br/>
  &nbsp;&nbsp;优化器SGD的时候，lr=0.01 <br/>
  &nbsp;&nbsp; 可以使用动态学习率，也可以先是大的学习率，然后慢慢减小，每次对半减小，寻找最优学习率
- **12.选择何种梯度下降算法**  <br/>
  &nbsp;&nbsp;对于初学者，选择SGD就可以
- **13.何时停止Epoch训练** <br/>
  &nbsp;&nbsp;当准确率不再上升，可以增加学习次数，让网络充分学习，当然要防止过拟合。迭代次数可以设置为1000或者更高，这个要结合样本量具体考量。在可以接受的时间范围内，将迭代次数设置的大一点。 <br/>
  
# 2.神经网络调参常遇问题汇总
-  **0. loss变化及其原因** <br/>
&nbsp;&nbsp;train loss 不断下降，test loss不断下降，说明网络仍在学习；<br/>
&nbsp;&nbsp;train loss 不断下降，test loss趋于不变，说明网络过拟合；<br/>
&nbsp;&nbsp;train loss 趋于不变，test loss不断下降，说明数据集100%有问题；<br/>
&nbsp;&nbsp;train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目；<br/>
&nbsp;&nbsp;train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。<br/>
-  **1. [梯度检验-相对误差](https://blog.csdn.net/han_xiaoyang/article/details/50521064)** <br/>
1. 相对误差>1e−2  意味着你的实现肯定是有问题的
2. 1e−2>相对误差>1e−4，你会有点担心
3. 1e−4>相对误差，基本是OK的，但是要注意极端情况(使用tanh或者softmax时候出现kinks)那还是太大
4. 1e−7>相对误差 ，放心大胆使用
5. 随着神经网络层数增多，相对误差是会增大的。这意味着，对于10层的神经网络，其实相对误差也许在1e-2级别就已经是可以正常使用的了。
6. 使用双精度浮点数。如果你使用单精度浮点数计算，那你的实现可能一点问题都没有，但是相对误差却很大。实际工程中出现过，从单精度切到双精度，相对误差立马从1e-2降到1e-8的情况。
7. 要留意浮点数的范围。一篇很好的文章是[What Every Computer Scientist Should Know About Floating-Point Arithmetic](http://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)我们得保证计算时，所有的数都在浮点数的可计算范围内，太小的值(比如h)会带来计算上的问题。

-  **2. 训练前的检查工作** <br/>
1. 在初始化之后看一眼loss。其实我们在用很小的随机数初始化神经网络后，第一遍计算loss可以做一次检查(当然要记得把正则化系数设为0)。以CIFAR-10为例，如果使用Softmax分类器，我们预测应该可以拿到值为2.302左右的初始loss(因为10个类别，初始概率应该都为0.1，Softmax损失是-log(正确类别的概率):-ln(0.1)=2.302)
2. 加回正则项，接着我们把正则化系数设为正常的小值，加回正则化项，这时候再算损失/loss，应该比刚才要大一些；
3. 试着去拟合一个小的数据集。最后一步，也是很重要的一步，在对大数据集做训练之前，我们可以先训练一个小的数据集(比如20张图片)，然后看看你的神经网络能够做到0损失/loss(当然，是指的正则化系数为0的情况下)，因为如果神经网络实现是正确的，在无正则化项的情况下，完全能够过拟合这一小部分的数据

-  **3. 训练过程中的监控** <br/>
1. 损失/loss随每轮完整迭代后的变化
 &nbsp;&nbsp;学习率设置使得loss稳定下降；<br/>
 &nbsp;&nbsp;loss上下波动剧烈，调高batch size；<br/>
2. 训练集/验证集上的准确度
 &nbsp;&nbsp;随着时间额推进，训练集和验证集上的准确度都会上升，如果训练集上的准确度到达一定的程度后，两者之间的插值较大，可能存在过拟合现象，如果插值不大，说明模型状况良好；<br/>
3. 

# 3.相关blog
-  **1. [数据预处理](https://blog.csdn.net/han_xiaoyang/article/details/50451460)** <br/>
1. 在很多神经网络的问题中，我们都建议对数据特征做预处理，去均值，然后归一化到[-1,1]之间;
2. 从一个标准差为√2/n的高斯分布中初始化权重，其中n为输入的个数;
3. 使用L2正则化(或者最大范数约束)和dropout来减少神经网络的过拟合;
4. 对于分类问题，我们最常见的损失函数依旧是SVM hinge loss和Softmax互熵损失.


# 4.专题调参
- cnn [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-2/#init)
- RBM [A Practical Guide to Training Restricted Boltzmann Machines](http://www.trade2win.com/boards/attachments/trading-software/96700d1290910624-3rd-generation-nn-deep-learning-deep-belief-nets-restricted-boltzmann-machines-practical-guide-training-rbm.pdf)


# 5.keras结构
![avatar](/images/keras1.jpg)
