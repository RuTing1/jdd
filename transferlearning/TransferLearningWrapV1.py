# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 10:11:37 2018

@author: dingru1
"""




#调用接口
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class TransferLearningMethod(object):
    """Specification for a data processing task.

        
    Parameters
    ----------
    data: DataFrame
    IDcol: string
    label: string
    encoder_cols: list
        list of features which trained by encoder
    new_features: boole
        defult False, do not need to care about it 
        
 
    Notes
    ----------
    1.应用时需要加载预训练的encoder结构，并利用训练好的模型初始化相关参数
    2.实现同构异构型迁移学习模型：异构的特征只能加在encoder末层结构，且末层结构目前紧邻最后一层
    3.可实现二分类迁移学习，多分类尚未遇到场景测试
    4.当数据量较少时，可以获得encoder特顶层的输出最为特征训练模型，即调用get_layer_outputs；当数据量还可以，可调用retrain_classifier_nets进行微调
    
    
    Examples
    --------
    A simple example for TransferLearningWrap where all the workers use the same dataset:
    >>> from TransferLearningWrapV1 import TransferLearningMethod #调用
    >>> tlm = TransferLearningMethod(data, None, 'card_desire', encoder_cols)  #实例化
    >>> layer_out = tlm.get_layer_outputs(network, sess, x, layer_name='output') #获取对应网络层的输出
    >>> dev_pred = tlm.retrain_classifier_nets(network, sess, x ) #微调同构或者异构迁移学习模型
    
    """
     
    
    __all__ = ['TransferLearningMethod', 'split_data', 'get_layer_outputs', 'retrain_classifier_nets']
    __version__ = '0.1'
    __author__ = 'dingru1'

        
    def __init__(self, data, IDcol, label, encoder_cols, new_features=False):
        
        
        self.data = data
        self.IDcol = IDcol
        self.label = label
        self.encoder_cols = encoder_cols
        self.new_features = new_features
    

    def split_data(self, test_size=0.1, val_size=0.1):
        """根据数据的特征将数据划分可以用于训练模型的数据
        
        Parameters
        ----------
        test_size：原始数据中用于划分训练集测试集时测试集占比
        val_size：训练集中用于划分训练集验证集时验证集占比
        
        
        Return 
        ----------
        data_split : dict
          同构型特征包含index：[X_train, X_val, x_test, y_train, y_val, y_test]
          异构型特征包含index：[X_train_old, X_train_new, X_val_old, X_val_new, x_test_old, x_test_new, y_train, y_val, y_test]
    
        """
        
        # 变量
        data = self.data
        IDcol = self.IDcol
        label = self.label
        encoder_cols = self.encoder_cols
        
        
        data_split = {}
        #目标数据包含的特征
        cols_sub_label = [col for col in data.columns if col not in [IDcol,label]]
        cols_new = [col for col in cols_sub_label if col not in encoder_cols]
        cols_omit = [col for col in encoder_cols if col not in cols_sub_label]
        
    #     assert is_new_cols, '所传数据不囊括编码器全部特征...'
        
        if len(cols_omit) > 0:
            print('所传数据遗漏{}个编码器特征...'.format(len(cols_omit)))
        elif len(cols_omit) ==0 and len(cols_new) ==0:
            print('所传数据特征与编码器特征匹配...')
            self.new_features = False
        else:
            print('所传数据包含{}个编码器外新特征...'.format(len(cols_new)))
            self.new_features = True
            
        train = data[:round(len(data)*(1 - test_size))]
        test = data[round(len(data)*(1 - test_size)):]
        
        x_train = train.loc[:,cols_sub_label]
        y_train = train[[label]]
        X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=42)
        x_test = test.loc[:,cols_sub_label]
        y_test = test[[label]]
        
        
        if self.new_features == False:
            print('--> 迁移学习类型：同构型迁移学习')
            data_split['X_train'] = X_train.values
            data_split['X_val'] = X_val.values
            data_split['y_train'] = y_train.values
            data_split['y_val'] = y_val.values
            data_split['x_test'] = x_test.values
            data_split['y_test'] = y_test.values
        elif self.new_features == True:
            print('--> 迁移学习类型：异构型迁移学习')
            X_train_old = X_train.loc[:,encoder_cols]
            X_train_new = X_train.loc[:,cols_new]
            X_val_old = X_val.loc[:,encoder_cols]
            X_val_new = X_val.loc[:,cols_new]
            x_test_old = x_test.loc[:,encoder_cols]
            x_test_new = x_test.loc[:,cols_new]
            
            data_split['X_train'] = X_train_old.values
            data_split['X_train_new'] = X_train_new.values
            data_split['X_val'] = X_val_old.values
            data_split['X_val_new'] = X_val_new.values
            data_split['x_test'] = x_test_old.values
            data_split['x_test_new'] = x_test_new.values
            data_split['y_train'] = y_train.values
            data_split['y_val'] = y_val.values
            data_split['y_test'] = y_test.values
        else:
            print('Please check the data features...')
        return data_split
    
    
    def get_layer_outputs(self, network, sess, x=None, layer_name='output'):
        """根据训练的encoder加上分类层训练分类器，并根据模型训练结果选择对应的编码器层进行输出
        
        Parameters
        ----------
        network: encoder built by tensorlayer network
        sess: Session which restore encoder model 
        x: network's inputlayer
        layer_name: name of layer which should output
        
        
        Return 
        ----------
        layer_out : output of corresponding network of the encoder   
        
        """     
        
        
        data = self.data
        IDcol = self.IDcol
        label = self.label
        encoder_cols = self.encoder_cols
        
        X = data.loc[:,encoder_cols].values
        Y = data.loc[:,label].values
        Y = Y.reshape(Y.shape[0],1)
        #assert data_split['X_train'].shape[1] == len(encoder_cols),'columns not much, data should has the same features with encoder features...'
        #x = tf.placeholder(tf.float32, shape=[None, len(encoder_cols)], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')
        network = tl.layers.DenseLayer(network, n_units=1, act=tf.identity, name='output')
        y = network.outputs
        loss = tl.cost.sigmoid_cross_entropy(y, y_)
        #train_op = tf.train.AdamOptimizer(0.005).minimize(loss) 
        sess.run(tf.global_variables_initializer())
        #读取所需层的输出
        dp_dict = tl.utils.dict_to_one(network.all_drop)
        feed_dict = {x: X, y_: Y}
        feed_dict.update(dp_dict)
        sess.run([loss], feed_dict=feed_dict)
        x_recon3 = tl.layers.get_layers_with_name(network,layer_name,True)
        layer_out = sess.run(x_recon3,feed_dict=feed_dict)
        layer_out = pd.DataFrame(layer_out[0])
        #ID_frame = data.loc[:,IDcol]
        #layer_out = pd.concat([ID_frame, layer_out], axis=1)
        layer_out.to_csv('{}_results.csv'.format(layer_name))
        return layer_out   
    
     
    def retrain_classifier_nets(self, network, sess, x=None, n_epoch=100, batch_size=100, learning_rate=0.001):
        """

        Parameters
        ----------
        network: encoder built by tensorlayer network
        sess: Session which restore encoder model 
        x: network's inputlayer
        n_epoch: int
        batch_size: int
        learning_rate: float    
        
        Notes
        ----------
        在同构或异构型迁移学习模型network的最末层加上一层分类器
         
         
        Return 
        ----------
        dev_pred : the prediction through the trained network
        Loss image: saved in results folder
        AUC image: saved in results folder
        
        """
        
        data = self.data
        IDcol = self.IDcol
        label = self.label
        encoder_cols = self.encoder_cols 
        new_features = self.new_features
        
        feas_num = len(encoder_cols)
        
        #根据data数据量，判断迁移学习的类型
        print('匹配数据量：',len(data))
        
        if len(data) < 3000:
            model_type = 'SmallParity'
        elif len(data) < 100000:
            model_type = 'MediumParity'
        else:
            model_type = 'LargeParity'
         
            
        #获取数据
        data_split = self.split_data(test_size=0.1, val_size=0.1)
        X_train, X_val, x_test = data_split['X_train'], data_split['X_val'] , data_split['x_test']
        y_train, y_val, y_test = data_split['y_train'], data_split['y_val'], data_split['y_test']
        #为保证minibatch的一致性，当不存在新特征时基于一列虚拟全1特征
        X_train_add, X_val_add, x_test_add = np.ones((X_train.shape[0],1)), np.ones((X_val.shape[0],1)), np.ones((x_test.shape[0],1))
        act = tf.nn.relu
        
        
        if (new_features == True) and (model_type != 'SmallParity'):
            X_train_add, X_val_add, x_test_add = data_split['X_train_new'], data_split['X_val_new'] , data_split['x_test_new']
            new_num = X_train_add.shape[1]
            x_new = tf.placeholder(tf.float32, shape=[None, new_num], name='x_new')
            inputs = tl.layers.InputLayer(x_new, name='input_layer_new')
            net_new = tl.layers.DenseLayer(inputs, n_units=new_num, act=act, name='relu3_1')
            network = tl.layers.ConcatLayer([network, net_new], 1, name ='concat_layer')
            """
            根据新增加的特征的多少，可以考虑前序网络与输出网络之间是否增加中间层
            """
        
        y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')
        network = tl.layers.DenseLayer(network,n_units=1, act=tf.identity, name='output')
        y = network.outputs
        #设置模型训练的参数
#        train_params = network.all_params
        #更新全部参数
        loss = tl.cost.sigmoid_cross_entropy(y,y_)
        train_op = tf.train.AdamOptimizer(0.005).minimize(loss) 
        # # 只更新部分参数
        # train_vars = tl.layers.get_variables_with_name('output',True, True)  #根据name获取特定的参数
        # train_op = tf.train.AdamOptimizer(0.005).minimize(loss,var_list=train_vars) 

        print_freq = 10
        sess.run(tf.global_variables_initializer())
        train_loss_list=[]
        val_loss_list=[]
        #train_val_dict = {}
    
        #训练模型
        for epoch in range(n_epoch):
            start_time = time.time()
            for X_train_ab, y_train_a in tl.iterate.minibatches(np.hstack((X_train, X_train_add)), y_train, batch_size, shuffle=True):
        #         print(X_train_a)
                X_train_a = X_train_ab[:,:feas_num]
                X_train_b = X_train_ab[:,feas_num:]
                if new_features == False:
                    feed_dict = {x: X_train_a, y_: y_train_a}
                elif new_features == True:
                    feed_dict = {x: X_train_a, y_: y_train_a, x_new:X_train_b}
                #微调阶段开启各降噪编码器内部Dropout层
                feed_dict.update(network.all_drop)  # enable noise layers
                #而denoising1只在预训练过程中开启，微调时则关闭
                feed_dict[tl.layers.LayersConfig.set_keep['denoising1']] = 1  # disable denoising layer
                sess.run(train_op, feed_dict=feed_dict)
            #每个epoch完结后，在训练集和测试集上做测试
            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                train_loss, n_batch = 0, 0
                #在训练集上测试
                for X_train_ab, y_train_a in tl.iterate.minibatches(np.hstack((X_train, X_train_add)), y_train, batch_size, shuffle=True):
                    X_train_a = X_train_ab[:,:feas_num]
                    X_train_b = X_train_ab[:,feas_num:]
                    if new_features == False:
                        feed_dict = {x: X_train_a, y_: y_train_a}
                    elif new_features == True:
                        feed_dict = {x: X_train_a, y_: y_train_a, x_new:X_train_b}
                    #关闭所有dropout层
                    dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict.update(dp_dict)
                    err  = sess.run([loss], feed_dict=feed_dict)[0]
    #                 if n_batch ==0:
    #                     x_recon3 = tl.layers.get_layers_with_name(network,'relu',True)
    #                     layer_out = sess.run(x_recon3,feed_dict=feed_dict)  #获得第三层的输出
    #                     print(sess.run([loss], feed_dict=layer_out))
                    train_loss += err
        #             train_acc += ac
                    n_batch += 1
                print("   train loss: %f" % (train_loss / n_batch))
        #         print("   train acc: %f" % (train_acc / n_batch))
                train_loss_list.append(train_loss/ n_batch)
                val_loss, n_batch = 0, 0
                #在验证集上测试
                for X_val_ab, y_val_a in tl.iterate.minibatches(np.hstack((X_val, X_val_add)), y_val, batch_size, shuffle=True):
                    X_val_a = X_val_ab[:,:feas_num]
                    X_val_b = X_val_ab[:,feas_num:]
                    if new_features == False:
                        feed_dict = {x: X_val_a, y_: y_val_a}
                    elif new_features == True:
                        feed_dict = {x: X_val_a, y_: y_val_a, x_new:X_val_b}
                    #关闭所有dropout层
                    dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict.update(dp_dict)
                    err  = sess.run([loss], feed_dict=feed_dict)[0]
                    val_loss += err
        #             val_acc += ac
                    n_batch += 1
                print("   val loss: %f" % (val_loss / n_batch))
        #         print("   val acc: %f" % (val_acc / n_batch))
                val_loss_list.append(val_loss/ n_batch)
        print('Evaluation')
        test_loss, n_batch = 0, 0
        for X_test_ab, y_test_a in tl.iterate.minibatches(np.hstack((x_test, x_test_add)), y_test, batch_size, shuffle=True):
            X_test_a = X_test_ab[:,:feas_num]
            X_test_b = X_test_ab[:,feas_num:]
            if new_features == False:
                feed_dict = {x: X_test_a, y_: y_test_a}
            elif new_features == True:
                feed_dict = {x: X_test_a, y_: y_test_a, x_new:X_test_b}
            dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
            feed_dict.update(dp_dict)
            err  = sess.run([loss], feed_dict=feed_dict)[0]
            test_loss += err
        #     test_acc += ac
            n_batch += 1
        print("   test loss: %f" % (test_loss / n_batch))
        # print("   test acc: %f" % (test_acc / n_batch))
        # print("   test acc: %f" % np.mean(y_test == sess.run(y_op, feed_dict=feed_dict)))
    
        y_pred = tl.utils.predict(sess, network, x_test, x, y)
        saver = tf.train.Saver()
        # you may want to save the model
        save_path = saver.save(sess, './model2/transfer_learning_lab_{}.ckpt'.format(label))
        print("Model saved in file: %s" % save_path)
        sess.close()
        
        #绘制训练集和测试集的训练过程
        x = range(round(n_epoch/print_freq +1))
        x = [i*10 for i in x]
        assert len(x) == len(train_loss_list) and len(x)== len(val_loss_list), 'not in the same length'
        plt.plot(x, train_loss_list, 'r', label = 'train')
        plt.plot(x, train_loss_list, 'ro')
        plt.plot(x, val_loss_list, 'b', label = 'validate')
        plt.plot(x, val_loss_list, 'bo')
        plt.title('change of loss during training and validation')
        plt.xlabel('number of epoch')
        plt.ylabel('loss of classification')
        plt.legend(loc="lower right")
    #     plt.show()
        plt.savefig('./result/trainingprocess_{}.jpg'.format(label))
        plt.close()
        
        #保存测试集的训练结果，二分类
        dev_pred = pd.DataFrame()
        dev_pred['score'] = y_pred.ravel().tolist()
        dev_pred['dep'] = y_test
        
        return dev_pred



        