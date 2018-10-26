##### This script is used to record my learning process of deep learning
# 1.神经网络调参技巧
-  **0.调参顺序** <br/>
   &nbsp;&nbsp;先调整学习率，观察成本函数下降的速率，修正学习率，一方面快速下降，另一方面防止不收敛；<br/>
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
- **5.选择何种Regularization？L1,L2**  <br/>
   &nbsp;&nbsp;L1和L2是对cost函数后面增加一项; <br/> 
- **6.Regularization parameter lambda 选择多大合适**  <br/>
  &nbsp;&nbsp;实验lambda，从1.0,10,100…找到一个合适的; <br/>
  &nbsp;&nbsp;注意：lambda的大小和样本数成正比关系，如样本数、lambda分别是1000、0.4，当样本数为10000时，对应的lambda也要扩大到4。 <br/>
- **7.激励函数如何选择**  <br/>
  &nbsp;&nbsp;先使用tanh函数作为隐藏层的激活函数。虽然ReLU函数收敛快，但如果学习率或者隐藏层结点数目一开始设置不好，很容易产生震荡，无法收敛。相反tanh函数就稳定一点。 <br/>
  &nbsp;&nbsp;只能靠实验比较 <br/>
- **8.是否使用dropout** <br/>
- **9.训练集多大比较合适**  <br/>
- **10.mini-batch选择多大**  <br/>
  &nbsp;&nbsp;mini-batch选择太小：没有充分利用计算资源；<br/>
  &nbsp;&nbsp;太大：更新权重和偏向比较慢；<br/>
  &nbsp;&nbsp;mini-batch size和其他参数相对独立，一旦找到一个合适的以后，就不需要再改了。 <br/>
- **11.学习率多少合适** <br/>
  &nbsp;&nbsp;优化器Adam的时候，lr=0.001; <br/>
  &nbsp;&nbsp;优化器SGD的时候，lr=0.01 <br/>
- **12.选择何种梯度下降算法**  <br/>
  &nbsp;&nbsp;对于初学者，选择SGD就可以
- **13.何时停止Epoch训练** <br/>
  &nbsp;&nbsp;迭代次数可以设置为1000或者更高，这个要结合样本量具体考量。在可以接受的时间范围内，将迭代次数设置的大一点。 <br/>
  

