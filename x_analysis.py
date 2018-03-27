#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:53:46 2017

@author: Jovial
"""

import pandas as pd
from sklearn.utils import resample

###调整好坏人比例
def BadResample(data,overdue_col,multiple=5,replace=True,random_state=None):
    '''
    功能: 调整好坏样本比例，输出调整后的样本集 
    输入值:
    data: 二维数组或dataframe，包括违约标记
    overdue_col: 一维数组或series，代表违约标记
    multiple: int，调整后好坏样本比例，默认5:1
    replace：boolean，是否有放回的采样，默认True
    random_state: 随机种子，默认None
    输出值: 
    二维数组或dataframe，调整后的样本集
    '''
    data_good = data[data[overdue_col[0]] == 0]
    data_bad = data[data[overdue_col[0]] == 1]
    n_resample = int(len(data_good)/multiple)
    data_bad = resample(data_bad,n_samples=n_resample,replace=replace,random_state=random_state)
    data = pd.concat([data_good,data_bad],axis=0)
    data.index = list(range(len(data)))
    return data

################################################################################################################
###小贷
#读取银河标签和还款表现
y_loan = pd.read_csv('y_analysis/20171227/y_loan.csv',encoding='iso-8859-1')
y_loan[['apply_no','jrid']] = y_loan[['apply_no','jrid']].astype(str)

data_loan_old = pd.read_csv('data/20171015/data_loan.csv',converters={'jrid':str})
yinhe_index = pd.read_excel('银河系统标签列表V2.xlsx')
index_select = ['jrid']+yinhe_index['标签编码'].tolist()
data_loan_old = data_loan_old.ix[:,data_loan_old.columns.isin(index_select)]
index_mapping = yinhe_index[['变量名','标签编码']].set_index('标签编码').T.to_dict('records')[0]
data_loan_old.rename(columns=index_mapping,inplace=True)

#合并银河标签和还款表现，排除灰色人群
data_loan = pd.merge(y_loan,data_loan_old,on='jrid',how='inner') #36662*251
data_loan = data_loan[~((data_loan['overdue_grey']==1)&(data_loan['overdue_m1']==0))] #31419 #排除1-3天逾期人群
data_loan = data_loan[~((data_loan['overdue_m0']==1)&(data_loan['overdue_m1']==0))] #29573 #排除4-30天逾期人数
data_loan.to_csv('data/20171227/data_loan.csv',index=None,header=True)

#划分训练集和测试集
data_loan_train = data_loan[data_loan['apply_month']<=201706] #9024
data_loan_test = data_loan[data_loan['apply_month']==201707] #4354
len(data_loan_train[data_loan_train['overdue_m1'] == 1]) #154
len(data_loan_test[data_loan_test['overdue_m1'] == 1]) #53

#调整训练集和测试集好坏人比例
data_loan_train = BadResample(data_loan_train,['overdue_m1'],multiple=5,replace=True,random_state=0) #10644
data_loan_test = BadResample(data_loan_test,['overdue_m1'],multiple=5,replace=True,random_state=0) #5161
len(data_loan_train[data_loan_train['overdue_m1'] == 1]) #1774
len(data_loan_test[data_loan_test['overdue_m1'] == 1]) #860

data_loan_train.to_csv('data/20171227/data_loan_train.csv',index=None,header=True)
data_loan_test.to_csv('data/20171227/data_loan_test.csv',index=None,header=True)

###银行
#读取银河标签和还款表现
y_bank = pd.read_csv('y_analysis/20171227/y_bank.csv',encoding='iso-8859-1')
y_bank[['apply_no','jrid']] = y_bank[['apply_no','jrid']].astype(str)

data_bank_old = pd.read_csv('data/20171015/data_bank.csv',converters={'jrid':str})
yinhe_index = pd.read_excel('银河系统标签列表V2.xlsx')
index_select = ['jrid']+yinhe_index['标签编码'].tolist()
data_bank_old = data_bank_old.ix[:,data_bank_old.columns.isin(index_select)]
index_mapping = yinhe_index[['变量名','标签编码']].set_index('标签编码').T.to_dict('records')[0]
data_bank_old.rename(columns=index_mapping,inplace=True)

#合并银河标签和还款表现，排除灰色人群
data_bank = pd.merge(y_bank,data_bank_old,on='jrid',how='inner') #76995*251
#(0,1]:40453,(1,2]:7766,(2,3]:2678
#data_bank = data_bank[~((data_bank['overdue_grey']==1)&(data_bank['overdue_m1']==0))] #34517 #排除1-3天逾期人群
data_bank = data_bank[~((data_bank['overdue_m0']==1)&(data_bank['overdue_m1']==0))] #73093 #排除4-30天逾期人数
data_bank.to_csv('data/20171227/data_bank.csv',index=None,header=True)

#划分训练集和测试集
data_bank_train = data_bank[data_bank['apply_month']<=201706] #15773
data_bank_test = data_bank[data_bank['apply_month']==201707] #12027
len(data_bank_train[data_bank_train['overdue_m1'] == 1]) #259
len(data_bank_test[data_bank_test['overdue_m1'] == 1]) #100

#调整训练集和测试集好坏人比例
data_bank_train = BadResample(data_bank_train,['overdue_m1'],multiple=5,replace=True,random_state=0) #18616
data_bank_test = BadResample(data_bank_test,['overdue_m1'],multiple=5,replace=True,random_state=0) #14312
len(data_bank_train[data_bank_train['overdue_m1'] == 1]) #3102
len(data_bank_test[data_bank_test['overdue_m1'] == 1]) #2385

data_bank_train.to_csv('data/20171227/data_bank_train.csv',index=None,header=True)
data_bank_test.to_csv('data/20171227/data_bank_test.csv',index=None,header=True)


