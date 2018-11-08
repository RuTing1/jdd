# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf 
import tensorlayer as tl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


#数据集切分
train = pd.read_csv('data/yhj_bank_desire_train_v2.csv',encoding='gbk')
test = pd.read_csv('data/yhj_bank_desire_validation_v2.csv',encoding='gbk')
train = train.fillna(0)
test = test.fillna(0)
cols = [col for col in train.columns if col not in ['jdpin','label','inv_desire', 'current_fin', 'fun_desire']]

x_train = train.loc[:,cols]
x_train = preprocessing.scale(x_train)
y_train = train[['label']]
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
x_test = test.loc[:,cols]
x_test = preprocessing.scale(x_test)
y_test = test[['label']]

#X_train = X_train.values
y_train = y_train.values
#X_val = X_val.values
y_val = y_val.values
#x_test = x_test.values
y_test = y_test.values



sess = tf.InteractiveSession()
# 定义 placeholder
x = tf.placeholder(tf.float32, shape=[None, 72], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')
# 定义模型
network = tl.layers.InputLayer(x, name='input_layer')
#network = tl.layers.DropoutLayer(network, keep=0.9, name='drop1')
network = tl.layers.DenseLayer(network, n_units=10,act = tf.nn.relu, name='relu1')
#network = tl.layers.DropoutLayer(network, keep=0.8, name='drop2')
network = tl.layers.DenseLayer(network, n_units=6,act = tf.nn.relu, name='relu2')
#network = tl.layers.DropoutLayer(network, keep=0.8, name='drop3')
network = tl.layers.DenseLayer(network, n_units=1,act = tf.identity,name='output_layer')

# 定义损失函数和衡量指标(分类问题的损失函数一般使用交叉熵代价函数，递归问题最小二乘误差函数)
# tl.cost.cross_entropy 在内部使用 tf.nn.sparse_softmax_cross_entropy_with_logits() 实现 softmax
#cost = tl.cost.cross_entropy(y, y_, name = 'cost')
#####构建NetworkStructure.loss
y = network.outputs
loss = tl.cost.mean_squared_error( y,y_)
#loss = tl.cost.binary_cross_entropy(y,y_,name='entropy')
#####构建NetworkStructure.acc
#correct_prediction = tf.equal(tf.arg_max(y,1),y_)
#acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#auc = compute_auc(y,y_,500)

# 定义 optimizer
train_params = network.all_params
train_op = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

# 初始化 session 中的所有参数
tl.layers.initialize_global_variables(sess)



# 列出模型信息
network.print_params()
network.print_layers()


# 训练模型,tensorboard=True可以生成EVENT，也可用tensorboard看调参过程
tl.utils.fit(sess, network, train_op, loss, X_train, y_train, x, y_,
            batch_size=500, n_epoch=1000, print_freq=100,
            X_val=X_val, y_val=y_val, eval_train=False)





# 评估模型
tl.utils.test(sess, network, x_test, y_test, x, y_, batch_size=None, cost=loss)

#模型结果预测
#y_op = tf.argmax(tf.nn.sigmoid(y), 1)
y_pred = tl.utils.predict(sess, network, x_test, x, y)

dev_pred = pd.DataFrame()

dev_pred['score'] = y_pred.ravel().tolist()
dev_pred['dep'] = test['label']
dev_pred['dep'] = dev_pred['dep'].replace(0,2)
dev_pred['dep'] = dev_pred['dep'].replace(1,0)
dev_pred['dep'] = dev_pred['dep'].replace(2,1)

import pandas as pa
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def predictive_Accu(df, dep, score):
    
    fpr_dev = dict()
    tpr_dev = dict()
    roc_auc_dev = dict() 
    fpr_dev, tpr_dev, _ = roc_curve(df[dep], df[score])
    roc_auc_dev = auc(fpr_dev, tpr_dev)

    dev_roc = {"fpr_dev":fpr_dev,"tpr_dev":tpr_dev}
    Dev_Roc = pa.DataFrame(dev_roc, columns=["fpr_dev", "tpr_dev"])
 
    return Dev_Roc, roc_auc_dev


dev_roc, dev_auc = predictive_Accu(dev_pred, "dep", "score")
plt.figure()
lw = 2
plt.plot(dev_roc['fpr_dev'], dev_roc['tpr_dev'], color='darkorange', lw=lw,  label='ROC curve(Dev)(area = %0.3f)' % dev_auc)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()
