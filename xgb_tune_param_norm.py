# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:12:41 2017

@author: dingru1
"""

import numpy as np
import pandas as pd
#import warnings
#import itertools
import time
from time import strftime
#import numpy.random as random
from numpy import hstack, vstack
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
#from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
#from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
file_dir = "C:\\Users\\dingru1\\Desktop\\"
today = strftime("%Y%m%d")
rcParams['figure.figsize'] = 12, 4
target = 'is_passed'
IDcol = 'lend_id'
lend_ids = ['CRPL_MSJR','CRPL_LHP', 'CRPL_NJCB','CRPL_BSB','CRPL_ZYXF','CRPL_WEIXIN','CRPL_ZLXF']



#读取数据

def read_data(lend_id,data_name):
    """
    输入：
    lend_id:不同建模对象
    data_name: 数据在file_dir下的名称
    输出：
    训练模型的X,y值
    """
    
    data = pd.read_csv(file_dir + data_name)
    train_data = data[data.lend_id==lend_id] 
    negative = train_data[train_data['is_passed'] == 0]
    positive = train_data[train_data['is_passed'] == 1]
    print("This is a model trained for {}".format(lend_id))
    print("\nThe num of positive samples is",len(positive))
    print("\nThe num of negative samples is",len(negative))
    X = train_data[predictors]
    y = train_data['is_passed']

    return X,y
    
def under_sample(X,y):
    # data_set:数据集
    # label:抽样标签
    # percent:抽样占比
    # q:每次抽取是否随机
    # 抽样根据目标列分层，自动将样本数较多的样本按percent抽样，得到目标列样本较多特征的欠抽样数据
    X['is_passed'] = y
    diff_case = pd.DataFrame(pd.Series(X['is_passed']).value_counts()).reset_index()
    max_cnt = diff_case['is_passed'].max()
    k3 = int(diff_case[diff_case['is_passed'] == max_cnt]['index'])
    percent = round(float(diff_case[diff_case.index!=k3].is_passed)/float(diff_case[diff_case.index==k3].is_passed),3)
    new_data = X[X['is_passed'] == k3].sample(frac=percent, random_state=10, axis=0)
    less_data = X[X['is_passed'] != k3]
    fina_data = new_data.append(less_data).sample(frac=1,random_state=10, axis=0)
    
    print('数据处理之欠采样处理完成！，正样本数量{},负样本数量{}'.format(len(new_data),len(less_data)))
    
    return fina_data

#predictors = [x for x in train.columns if x not in [target, IDcol]]

#初始xgboost学习器
pipe_xgb=Pipeline([('fillna',Imputer(strategy="mean",axis=0)),
                   ('select_feature',SelectFromModel(RandomForestClassifier(random_state=10,),threshold=0.005)),
                   ('clf', xgb.XGBClassifier(
                            learning_rate=0.1,n_estimators=100,max_depth=5,min_child_weight=1,
                            gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',
                            nthread=4,scale_pos_weight=1,seed=27))
                  ])

##模型调参过程

#lend_id = 'CRPL_NJCB'
X,y = read_data('lend_id',train)
#train = under_sample(X,y)
##X,y = train[predictors],train[target]

def step_0():
    """
    对进入模型的变量进行筛选
    """
    start_time = time.time()
    param_test0 = {
                     'select_feature__threshold':[0.004,0.005,0.006,0.007,0.008]
                    }
    gsearch0 = GridSearchCV(pipe_xgb,param_grid = param_test0, scoring='roc_auc',iid=False, cv=5)
    gsearch0.fit(X,y)

    end_time = time.time()

    print("\nThis is a tuning params of {}".format(lend_id))
    print("\nStep0: The process of tuning features threshold:")
    print(gsearch0.grid_scores_)
    print(gsearch0.best_params_)
    print(gsearch0.best_score_)
    print("\nThe time step0 used is:",end_time - start_time)
#         print ("\nThe Baseline Accuracy : %.4g" % accuracy_score(y_test.values, dtrain_predictions))
#         print ("\nThe Baseline AUC Score (Train): %f" % roc_auc_score(y_test, dtrain_predprob))
    
    return gsearch0.best_params_
    
prior_params0 = step_0()

def step_1(prior_params):
    """
    max_depth：list(range(4,12,2))
    min_child_weight:list(range(1,8,2))
    """
    
    start_time = time.time()
    pipe_xgb.set_params(**prior_params)
    param_test1 = {
                     'clf__max_depth':list(range(4,12,2)),
                     'clf__min_child_weight':list(range(1,8,2))
                    }
    gsearch1 = GridSearchCV(pipe_xgb,param_grid = param_test1, scoring='roc_auc',iid=False, cv=5)
    gsearch1.fit(X,y)
    print("\nStep1: The process of tuning max_depth & min_child_weigh:")
    print(gsearch1.grid_scores_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)
    params = gsearch1.best_params_
#    prior_params_new = dict(prior_params,**params)
    end_time = time.time()
    print("\nThe time step1 used is:",end_time - start_time)

    return params

prior_params1 = step_1(prior_params0)

def step_2(prior_params):
    """
    gamma：[0,0.1,0.2,0.3,0.4,0.5]
    
    """

    start_time = time.time()
    pipe_xgb.set_params(**prior_params)
    param_test2 = {
#                          'clf__gamma':[[i/10.0 for i in range(0,5)]]
                    'clf__gamma': [0,0.1,0.2,0.3,0.4,0.5]
                    }
    gsearch2 = GridSearchCV(pipe_xgb,param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
    gsearch2.fit(X,y)
    print("\nStep2:The process of tuning gamma:")
    print(gsearch2.grid_scores_)
    print(gsearch2.best_params_)
    print(gsearch2.best_score_)
    params = gsearch2.best_params_
    prior_params_new = dict(prior_params,**params)
    end_time = time.time()
    print("\nThe time step2 used is:",end_time - start_time)
    
    return prior_params_new

prior_params2 = step_2(prior_params1)

def step_3(prior_params):

    """
    subsample：[i/10.0 for i in range(6,10)]
    colsample_bytree：[i/10.0 for i in range(6,10)]
    
    """
    start_time = time.time()
    pipe_xgb.set_params(**prior_params)
    param_test3 = {
                     'clf__subsample':[i/10.0 for i in range(6,10)],
                     'clf__colsample_bytree':[i/10.0 for i in range(6,10)]
                    }

    gsearch3 = GridSearchCV(pipe_xgb,param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
    gsearch3.fit(X,y)
    print("\nStep3:The process of tuning subsample&colsample_bytree:")
    print(gsearch3.grid_scores_)
    print(gsearch3.best_params_)
    print(gsearch3.best_score_)
    params = gsearch3.best_params_
    prior_params_new = dict(prior_params,**params)
    end_time = time.time()
    print("\nThe time step3 used is:",end_time - start_time)
    
    return prior_params_new

prior_params3 = step_3(prior_params2)

def step_4(prior_params):
    """
    reg_alpha:[1e-5, 1e-2, 0.1, 1, 100]
        
    """
    
    start_time = time.time()
    pipe_xgb.set_params(**prior_params)
    param_test4 = {
                     'clf__reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
                    }
    gsearch4 = GridSearchCV(pipe_xgb,param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
    gsearch4.fit(X,y)
    print("\nStep4:The process of tuning reg_alpha:")
    print(gsearch4.grid_scores_)
    print(gsearch4.best_params_)
    print(gsearch4.best_score_)
    params = gsearch4.best_params_
    prior_params_new = dict(prior_params,**params)
    joblib.dump(gsearch4.best_estimator_, '{}{}_XGB_{}.model'.format(file_dir,lend_id,today))
    end_time = time.time()
    print("\nThe time step4 used is:",end_time - start_time)
    
    return prior_params_new,gsearch4.best_score_

prior_params4,best_score = step_4(prior_params3)

def step_5(prior_params,best_score):
    
    """
    learning_rate:
    n_estimators:
    
    """
    
    start_time = time.time()
    pipe_xgb.set_params(**prior_params)
    param_test5 = {
                     'clf__learning_rate':[0.01],
                     'clf__n_estimators':[1000]
                    }
    gsearch5 = GridSearchCV(pipe_xgb,param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
    gsearch5.fit(X,y)
    print("\nStep5:The process of tuning learning_rate&n_estimators:")
    print(gsearch5.grid_scores_)
    print(gsearch5.best_params_)
    print(gsearch5.best_score_)
    
    prior_params = gsearch5.best_params_
    pipe_xgb.set_params(**prior_params)
    pipe_xgb.fit(X,y)
    
    print ("\nThe Tuned AUC Score (Train): %f" % gsearch5.best_score_)
    
    if gsearch5.best_score_ > best_score:
        joblib.dump(pipe_xgb, '{}{}_XGB_{}.model'.format(file_dir,lend_id,today))
    else:
        print('Tuning clf__learning_rate&clf__n_estimators again didnt get any improve')
        
    end_time = time.time()
    print("\nThe time step5 used is:",end_time - start_time)

step_5(prior_params4,best_score)

end_time = time.time()
print('\nThe time of tuning params is:',end_time - start_time)
print('-----------------------------------------------------------------')



################################################
#报告分隔线
################################################

#####test

test  = pd.read_csv(file_dir+'test.csv')
test = test.loc[:,[x for x in train.columns if x not in [target, IDcol]]]

#
#load model
def model_evaluation(lend_id,data):
    
    #获取验证数据
    test_data = data[data.lend_id==lend_id]
    print('正样本数量%s'%len(test_data[test_data.is_passed==1]),'负样本数量%s'%len(test_data[test_data.is_passed==0]))
#     online_test = test_data[:1]
    y = test_data[target]
    test_data = test_data[predictors]
    
    #获取验证模型
    model = joblib.load('{}{}_XGB_{}.model'.format(file_dir,lend_id,'20171109'))

    #获取评价结果
    probas_y = model.predict_proba(test_data)
    
    
    test_y = model.predict(test_data)
    test_yprob = model.predict_proba(test_data)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, test_yprob [:,1] )
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("the auc of XGB model is {}".format(roc_auc))
    print("the Confusion report of {}:".format(lend_id), classification_report(y, test_y))
    
    
#     online_test['predict_y'] = probas_y
#     online_test.to_csv('{}{}_onlinetest_{}.csv'.format(file_dir,lend_id,today),index=True) 
    
    #线上测试数据：
    online_test = test_data[10:11]
    online_y = y[10:11]
    online_y_pro = model.predict_proba(online_test)
#     print(pd.DataFrame(online_y_pro))
    online_test['is_passed'] = list(online_y)
    online_test['is_passed_pro'] = list(online_y_pro)
    online_test.to_csv('{}{}_testdata_{}.csv'.format(file_dir,lend_id,today),index=True)
    
    score,score_level = [],[]
    for i in range(0,len(probas_y)):
        p_score = probas_y[i][1]*100
        score.append(int(p_score))
        score_level.append(int(p_score/2)+1)
    scores = pd.DataFrame({'score':score,
                           'is_passed':y,
                           'score_level':score_level
            }
            )
#     scores = scores.dropna(axis=0, how='any')
    def score_ks(_max_score, true_score, false_score):
        true_score_len = len(true_score)
        false_score_len = len(false_score)
        _score = []
        _true_ratio = []
        _false_ratio = []
        _ks = 0
        _ks_index = 0
        for i in range(1, _max_score):
            true_len = len(true_score[scores.score <= i])
            false_len = len(false_score[scores.score <= i])
            _true_ratio.append(true_len/true_score_len)
            _false_ratio.append(false_len/false_score_len)
            _score.append(i)
            ks = np.abs(true_len/true_score_len - false_len/false_score_len)
            if(ks > _ks):
                _ks = ks
                _ks_index = i
            true_ks = pd.DataFrame({'score': _score,
                              'true_ratio': _true_ratio})
            false_ks = pd.DataFrame({'score': _score,
                              'false_ratio': _false_ratio})
        return true_ks, false_ks, _ks, _ks_index


    def get_ks(original_scores):
#            _min_score = np.min(scores['score'])
        _max_score = np.max(scores['score'])
        true_score = scores[scores.is_passed == 1]
        false_score = scores[scores.is_passed == 0]
        true_ks,false_ks, _ks, _ks_index = score_ks(_max_score, true_score, false_score)
        return _ks

    def score_distribution(scores,roc_auc):
        scores_group = scores.groupby(['score_level'])
        kss = list(map(lambda x: get_ks(x),scores_group.groups))
        true_scores = scores[scores.is_passed == 1]
        false_scores = scores[scores.is_passed == 0]
        count_true_score_level = pd.value_counts(true_scores['score_level'], sort = True).sort_index()
        count_false_score_level = pd.value_counts(false_scores['score_level'], sort = True).sort_index()
        count_score_level = pd.value_counts(scores['score_level'], sort = True).sort_index()
        table = pd.DataFrame({
                              'auc':roc_auc,
                              'passed':count_true_score_level,
                              'not_passed':count_false_score_level,
                              'total':count_score_level,
                              'ks':kss})
        table = table.fillna(0)
        total_true_num = table['passed'].sum()
        total_false_num = table['not_passed'].sum()
        total_num = table['total'].sum()

        table['passed_rate'] = table.apply(lambda x:x['passed']/x['total'],axis=1)
        table['passed_prop'] = table.apply(lambda x:x['passed']/total_true_num,axis=1)
        table['not_passed_prop'] = table.apply(lambda x:x['not_passed']/total_false_num,axis=1)
        table.to_csv('{}{}_score_{}.csv'.format(file_dir,lend_id,today),index=True) 
        
    score_distribution(scores,roc_auc)

model_evaluation(lend_id,test)
#