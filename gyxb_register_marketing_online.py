#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:18:21 2018

@author: dingru1
"""

import numpy as np
import pandas as pd
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve,auc
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
IDcol = 'jd_pin'
target = 'is_register'
file_path = '/exportfs/home/cuitingting/server/gyxb/'
#file_path = 'C://Users//dingru1//03marketing//'
random_state = 2018


#####################函数库1##############################
### ks曲线
def ks_calc(data,score_col,class_col):
    from scipy import stats 
    ks_dict = {}
    Bad = data.ix[data[class_col]==1,[score_col]]
    Good = data.ix[data[class_col]==0, [score_col]]
    ks,pvalue = stats.ks_2samp(Bad.values.flatten(),Good.values.flatten())

    crossfreq = pd.crosstab(data[score_col],data[class_col])
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['gap'] = abs(crossdens[0] - crossdens[1])
    score_split = crossdens[crossdens['gap'] == crossdens['gap'].max()].index[0]
    ks_dict['ks'] = ks
    ks_dict['split'] = score_split

    return ks_dict 

###计算AUC
def auc_calc(data,score_col,class_col):
    '''
    功能: 计算AUC值，并输出ROC曲线
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    score_col: 一维数组或series，代表模型得分（一般为预测正类的概率）
    class_col: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值: 
    字典，键值关系为{'auc': AUC值，'auc_fig': ROC曲线}
    '''
    auc_dict = {}
    fpr,tpr,threshold = roc_curve(data[class_col],data[score_col])
    roc_auc = auc(fpr,tpr)
    auc_dict['auc'] = roc_auc

    return auc_dict


#####################函数库2##############################
def mis_high_std_low(data_stat,data_len,missing_ratio=0.5,std_threshold=0.2):
    """
    input： data_stat 变量的描述统计，可传入data.describe()
            data_len  原始数据的样本数
            missing_ratio 缺失值删除的占比阈值
            std_threshold 方差过小的删除阈值
    output: info_cols 剔除缺失值过多和方差过小变量后变量组
    
    """
    drop_cols = set([IDcol,target])&set(data_stat.columns)
    data_stat = data_stat.drop(drop_cols,axis=1)
    std_info = set(data_stat.loc['std',data_stat.loc['std',]>std_threshold].index)
    missing_info = set(data_stat.loc['count',data_stat.loc['count',]/data_len>missing_ratio].index)
    info_cols = list(std_info&missing_info)
    
    return info_cols

def delete_coonline(data,target,threshold=0.8):
    """
    input：data 变量相关性矩阵
           target 目标变量
           threshold 变量相关性阈值
    output:共线性较低的变量组
    
    """
    drop_cols = []
    cor_cols = [col for col in data.columns if col not in [target]]
    data_cor = data.loc[:,cor_cols]
    
    for col in cor_cols:
        drops = list(data_cor[(data_cor[col]> threshold)|(data_cor[col]< -threshold)].index)
        traget_col = data.loc[drops,[target]]
        cor_target = max(traget_col[target].max(),-traget_col[target].min())
        if cor_target != traget_col[target].max():
            cor_target = traget_col[target].min()
        retain_col_index = list(traget_col[target]).index(cor_target)  #最大值选一个就好
        retain_col = traget_col[target].index[retain_col_index]
        if retain_col is  col:
            if len(traget_col)==1:
                drops = []
            drops = [col for col in drops if col not in [retain_col]]
        else:
            drops = [col]

        drop_cols.extend(drops)
    retain_cols = list((set(cor_cols) - set(drop_cols)))
    
    return retain_cols

##########################模型训练####################

from sklearn.linear_model import LogisticRegression

def train_model(X,y,title):
    """
    input：X 特征
          y 目标变量
          title 训练的模型类型
    output:最优模型
    """
    params_lr = [{'penalty':['l2'],'C':[0.001,0.01,0.1,1]}]
    clf_lr =  LogisticRegression(random_state=2018)
    cv = GridSearchCV(clf_lr,params_lr,cv=5,scoring='roc_auc')
    cv.fit(X.as_matrix(columns=None),list(y[target]))
    best_params = cv.best_params_
    print('the best params of the classifier in train data is:',best_params)
    print('the best score of the classifier in train data is:',cv.best_score_)
    best_clf = cv.best_estimator_ 

    return best_clf


##########################模型评估####################
def eva_model(X,y,clf,title):
    """
    input：X 特征
          y 目标变量
          clf 预测模型
          title 训练的模型类型
    output:y_score 模型整体表现评估
           y 每个id的预测结果
    
    """
    y['prob'] = clf.predict_proba(X)[:,1]
    y['pred'] = clf.predict(X)
    fpr,tpr,threshold = roc_curve(y.loc[:,[target]],y.loc[:,['prob']])
    auc_dic = auc_calc(y,'prob',target)
    ks_dic = ks_calc(y,'prob',target)
    print('the auc of the classifier in test data is:',auc_dic['auc'])
    print('the ks of the classifier in test data is:',ks_dic['ks'])
    y.loc[y['prob'] < 0.001,['prob']] = 0.001
    y.loc[y['prob'] > 0.999,['prob']] = 0.999
    y['odds'] = y['prob']/ (1-y['prob'])
    y['score'] = (np.log(y['odds'])*(20/np.log(2))+600).astype(int)
    score_bins = list(range(400,800,20))
    y['scorebin'] = pd.cut(y['score'],bins=score_bins)
    y_score = pd.crosstab(y['scorebin'],y[target])
    y_score.columns = ['Bad','Good']
    y_score['reg_rate'] = y_score['Good']/y_score.sum(axis=1)
    y_score['noreg_rate'] = y_score['Bad']/y_score.sum(axis=1)
    y_score['reg_cumrate'] = y_score['Good'].cumsum()/y_score['Good'].sum()
    y_score['noreg_cumrate'] = y_score['Bad'].cumsum()/y_score['Bad'].sum()
    
    return y_score,y


### train model
from sklearn.preprocessing import MinMaxScaler
def train_model_iid(train_data,title):
    train_data = train_data.replace(['N',-1,'-1',-9999,'-9999'],np.nan).astype(float).fillna(-1)
    info_cols = mis_high_std_low(train_data.describe(),len(train_data),missing_ratio=0.5,std_threshold=0.3)
    train_data = train_data.loc[:,info_cols+[target]]
    use_cols = delete_coonline(train_data.corr(),target,threshold=0.8)
    X,y = train_data.loc[:,use_cols] ,train_data.loc[:,[target]] 
    mm_scaler = MinMaxScaler()
    X = mm_scaler.fit_transform(X)
    X = pd.DataFrame(X,index=train_data.index)
    X.columns = use_cols
    
    clf = train_model(X,y,title)
    
    return use_cols,clf,mm_scaler
    
    

###########################inv_propensity_score##############################
# train model
propensity_train =  pd.read_csv('{}data/inv_properity_train.csv'.format(file_path),index_col='jd_pin',converters={'jrid':str},delimiter=',')
use_cols1,clf_1,mm_scaler1 = train_model_iid(propensity_train,'inv_propensity')
# test model
propensity_tmp = pd.read_csv('{}data/inv_properity_test.csv'.format(file_path),index_col='jd_pin',converters={'jrid':str},delimiter='\t')
propensity_tmp = propensity_tmp.replace(['N',-1,'-1',-9999,'-9999'],np.nan).astype(float).fillna(-1)
propensity_test = propensity_tmp.loc[:,use_cols1]
propensity_test = pd.DataFrame(mm_scaler1.transform(propensity_test),index=propensity_tmp.index)
propensity_test.columns = use_cols1
X1,y1 = propensity_test,propensity_tmp.loc[:,[target]].astype(int)
inv_propensity_score,inv_propensity = eva_model(X1,y1,clf_1,'inv_propensity')
inv_propensity_score.to_csv('{}result/inv_propensity_score.csv'.format(file_path),sep='\t')
inv_propensity = pd.concat([inv_propensity,propensity_tmp.loc[:,'is_inv']],axis=1)
inv_propensity.to_csv('{}result/inv_propensity.csv'.format(file_path),sep='\t')



###########################assert_value_score##############################
# train model
assert_train =  pd.read_csv('{}data/assert_value_train.csv'.format(file_path),index_col='jd_pin',converters={'jrid':str},delimiter=',')
use_cols2,clf_2,mm_scaler2 = train_model_iid(assert_train,'assert_value')
# test model
assert_tmp = pd.read_csv('{}data/assert_value_test.csv'.format(file_path),index_col='jd_pin',converters={'jrid':str},delimiter='\t')
assert_tmp = assert_tmp.replace(['N',-2,'-2',-9999,'-9999'],np.nan).astype(float).fillna(-1)
assert_test = assert_tmp.loc[:,use_cols2]
assert_test = pd.DataFrame(mm_scaler2.transform(assert_test),index=assert_tmp.index)
assert_test.columns = use_cols2
X2,y2 = assert_test,assert_tmp.loc[:,[target]].astype(int)
assert_value_score,assert_value = eva_model(X2,y2,clf_2,'assert_value')
assert_value_score.to_csv('{}result/assert_value_score.csv'.format(file_path),sep='\t')
assert_value = pd.concat([assert_value,assert_tmp.loc[:,'is_inv']],axis=1)
assert_value.to_csv('{}result/assert_value.csv'.format(file_path),sep='\t')


###########################marketing_response_score##############################
response_train =  pd.read_csv('{}data/marketing_response_train.csv'.format(file_path),index_col='jd_pin',converters={'jrid':str},delimiter=',')
use_cols3,clf_3,mm_scaler3 = train_model_iid(response_train,'marketing_response')
# test model
response_tmp = pd.read_csv('{}data/marketing_response_test.csv'.format(file_path),index_col='jd_pin',converters={'jrid':str},delimiter='\t')
response_tmp = response_tmp.replace(['N',-3,'-3',-9999,'-9999'],np.nan).astype(float).fillna(-1)
response_test = response_tmp.loc[:,use_cols3]
response_test = pd.DataFrame(mm_scaler3.transform(response_test),index=response_tmp.index)
response_test.columns = use_cols3
X3,y3 = response_test,response_tmp.loc[:,[target]].astype(int)
marketing_response_score,marketing_response = eva_model(X3,y3,clf_3,'marketing_response')
marketing_response_score.to_csv('{}result/marketing_response_score.csv'.format(file_path),sep='\t')
marketing_response = pd.concat([marketing_response,response_tmp.loc[:,'is_inv']],axis=1)
marketing_response.to_csv('{}result/marketing_response.csv'.format(file_path),sep='\t')








