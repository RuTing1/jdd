#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:15:47 2017

@author: Jovial
"""

import pandas as pd
import numpy as np
import src.DataAnalysis as DataAnalysis
import src.discretize as discretize
import src.continuous as continuous
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_curve,auc
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

###################################定义函数库#########################################
###计算KS
def ks_calc(data,score_col,class_col):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    score_col: 一维数组或series，代表模型得分（一般为预测正类的概率）
    class_col: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值: 
    字典，键值关系为{'ks': KS值，'split': KS值对应节点，'fig': 累计分布函数曲线图}
    '''
    ks_dict = {}
    Bad = data.ix[data[class_col[0]]==1,score_col[0]]
    Good = data.ix[data[class_col[0]]==0, score_col[0]]
    ks,pvalue = stats.ks_2samp(Bad.values,Good.values)
    crossfreq = pd.crosstab(data[score_col[0]],data[class_col[0]])
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['gap'] = abs(crossdens[0] - crossdens[1])
    score_split = crossdens[crossdens['gap'] == crossdens['gap'].max()].index[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    crossdens[[0,1]].plot(kind='line',ax=ax)
    ax.set_xlabel('%s' % score_col[0])
    ax.set_ylabel('Density')
    ax.set_title('CDF Curve of Classified %s' % score_col[0])
    plt.close()
    ks_dict['ks'] = ks
    ks_dict['split'] = score_split
    ks_dict['ks_fig'] = fig
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
    fpr,tpr,threshold = roc_curve((1-data[class_col[0]]).ravel(),data[score_col[0]].ravel())
    roc_auc = auc(fpr,tpr)
    fig = plt.figure()
    plt.plot(fpr,tpr,color='b',label='ROC Curve (area=%0.3f)'%roc_auc,alpha=0.3)
    plt.plot([0,1],[0,1],color='r',linestyle='--',alpha=0.3)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of %s'%score_col[0])
    plt.legend(loc='lower right')
    plt.close()
    auc_dict['auc'] = roc_auc
    auc_dict['auc_fig'] = fig
    return auc_dict

###交叉验证计算模型测试集上平均AUC和KS
def fit_cv(X_matrix,Y_matrix,clf,cv,random):
    '''
    功能: 交叉验证计算模型测试集上平均AUC和KS
    输入值:
    X_matrix: 多维数组或dataframe，入模变量大宽表
    Y_matrix: 一维数组或series，真实的标签
    clf: 训练好的分类器
    cv: int，交叉验证中测试集的个数
    random: 随机排列的随机种子
    输出值: 
    字典，键值关系为{'auc': 测试集上平均AUC值，'ks': 测试集上平均KS值}
    '''
    skf = StratifiedKFold(y=Y_matrix,n_folds=cv,random_state=random)
    auc_list = []
    ks_list = []
    cv_dict = {}
    for train_index,test_index in skf:
        X_train,X_test = X_matrix[train_index],X_matrix[test_index]
        Y_train,Y_test = Y_matrix[train_index],Y_matrix[test_index]
        fit_data = pd.DataFrame(index=range(len(test_index)),columns=['Y','Prob'])
        clf = clf.fit(X_train,Y_train)
        fit_data['Y'] = Y_test
        fit_data['Prob'] = clf.predict_proba(X_test)[:,0]
        score_col = ['Prob']
        class_col = ['Y']
        auc_ = auc_calc(fit_data,score_col,class_col)['auc']
        auc_list.append(auc_)
        ks_ = ks_calc(fit_data,score_col,class_col)['ks']
        ks_list.append(ks_)
        cv_dict['ks'] = np.mean(ks_list)
        cv_dict['auc'] = np.mean(auc_list)
    return cv_dict

###画好坏人分数分布对比直方图
def plot_hist_score(y_true,y_score,close=True):
    '''
    功能: 画好坏人分数分布对比直方图
    y_true: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    y_score: 一维数组或series，代表模型得分
    close: 是否关闭图片
    返回图片对象
    '''
    bad = y_score[y_true==1]
    good = y_score[y_true==0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(bad,bins=100,alpha=0.6,color='r',label='Bad')
    ax.hist(good,bins=100,alpha=0.6,color='b',label='Good')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.legend(loc='best')
    ax.set_title('Histogram of Score in Good vs Bad')
    if close:
        plt.close('all')
    return fig

####读取表现数据
#y_loan = pd.read_csv('y_analysis/y_loan.csv',converters={'jrid':str}) 
#y_bank = pd.read_csv('y_analysis/y_bank.csv',converters={'jrid':str})
#y_loan.shape #63682*6
#y_bank.shape #142293*6
#
####读取银河数据第一部分
#x_yinhe1 = pd.read_csv('x_analysis/tag_x1.csv',converters={'jdpin':str})
#x_yinhe1['dt'] = pd.to_datetime(x_yinhe1['dt'].astype(str))
#x_yinhe1 = x_yinhe1.sort_values(by='dt')
#x_yinhe1 = x_yinhe1.groupby(by='jdpin',sort=False).last().reset_index() 
#x_yinhe1 = x_yinhe1.drop(['unikey','dt','row_index'],axis=1)
#x_yinhe1.shape #3261904*158
#
####读取银河数据第二部分
#x_yinhe2 = pd.read_csv('x_analysis/tag_x2.csv',converters={'jdpin':str})
#x_yinhe2['dt'] = pd.to_datetime(x_yinhe2['dt'].astype(str))
#x_yinhe2 = x_yinhe2.sort_values(by='dt')
#x_yinhe2 = x_yinhe2.groupby(by='jdpin',sort=False).last().reset_index() 
#x_yinhe2 = x_yinhe2.drop(['unikey','dt','row_index'],axis=1)
#x_yinhe2.shape #3181305*196
#
####合并银河数据
#x_yinhe = pd.merge(x_yinhe1,x_yinhe2,on='jdpin',how='inner')
#x_yinhe.shape #2294391*353
#
####合并表现和银河数据
#data_loan = pd.merge(y_loan[['jrid','overdue_m0']],x_yinhe,left_on='jrid',right_on='jdpin',how='inner')
#data_bank = pd.merge(y_bank[['jrid','overdue_m0']],x_yinhe,left_on='jrid',right_on='jdpin',how='inner')
#data_loan = data_loan.drop(['jdpin'],axis=1)
#data_bank = data_bank.drop(['jdpin'],axis=1)
#data_loan.shape #32070
#data_bank.shape #71749
#len(data_loan[data_loan['overdue_m0'] == 1]) #3422
#len(data_bank[data_bank['overdue_m0'] == 1]) #3770
#data_loan.to_csv('result/data_loan.csv',index=None,header=True)
#data_bank.to_csv('result/data_bank.csv',index=None,header=True)

###################################读入数据#########################################
###筛选标签列表并重命名
data_loan = pd.read_csv('data/data_loan.csv',converters={'jrid':str})
data_bank = pd.read_csv('data/data_bank.csv',converters={'jrid':str})
yinhe_index = pd.read_excel('银河系统标签列表V2.xlsx')
index_select = ['jrid','overdue_m0']+yinhe_index['标签编码'].tolist()
data_loan = data_loan.ix[:,data_loan.columns.isin(index_select)]
data_bank = data_bank.ix[:,data_bank.columns.isin(index_select)]
index_mapping = yinhe_index[['变量名','标签编码']].set_index('标签编码').T.to_dict('records')[0]
data_loan.rename(columns=index_mapping,inplace=True)
data_bank.rename(columns=index_mapping,inplace=True)

###################################互联网小贷建模#########################################
#####逻辑回归
###数据预处理
data_loan = data_loan.replace(['N',-1,'-1',-9999,'-9999'],np.nan)
data_loan['xjkacctopenday'] = (datetime.now() - pd.to_datetime(data_loan['xjkacctopentime'].astype(str))).astype('timedelta64[D]')
data_loan['xjkopenday'] = (datetime.now() - pd.to_datetime(data_loan['xjkopentime'].astype(str))).astype('timedelta64[D]')
data_loan['sytfirstpayday'] = (datetime.now() - pd.to_datetime(data_loan['sytfirstpaytime'].astype(str))).astype('timedelta64[D]')
data_loan['applastday'] = (datetime.now() - pd.to_datetime(data_loan['applasttime'].astype(str))).astype('timedelta64[D]')
data_loan['userregday'] = (datetime.now() - pd.to_datetime(data_loan['userregtime'].astype(str))).astype('timedelta64[D]')
data_loan['sytfirstregday'] = (datetime.now() - pd.to_datetime(data_loan['sytfirstregtime'].astype(str))).astype('timedelta64[D]')
data_loan['sytfirstusedday'] = (datetime.now() - pd.to_datetime(data_loan['sytfirstusedtime'].astype(str))).astype('timedelta64[D]')
data_loan = data_loan.drop(['xjkacctopentime','xjkopentime','sytfirstpaytime','applasttime','userregtime','sytfirstregtime','sytfirstusedtime'],axis=1)
data_loan['appuserstatus'].replace('loanded',1,inplace=True)
data_loan['memdevice'].replace(['idfa','imei'],[1,2],inplace=True)
data_loan.shape #32070*244
data_loan.ix[:,2:] = data_loan.ix[:,2:].astype(float)

btscore_loan = data_loan['btcreditscore']
xbscore_loan = data_loan['memxbscore']

stat_describe_loan = DataAnalysis.stat_df(data_loan)
stat_describe_loan = stat_describe_loan.sort_values(by='missing_ratio')
columns_select = stat_describe_loan[stat_describe_loan['missing_ratio']<0.5].index.tolist()
data_loan = data_loan[columns_select] 
data_loan.shape #32070*150

###计算WOE和IV
stat_describe_loan = DataAnalysis.stat_df(data_loan.ix[:,2:])
stat_describe_loan = stat_describe_loan.sort_values(by='count_unique')
columns_continous = stat_describe_loan[stat_describe_loan['count_unique']>13].index.tolist()
quantiles = [20*i  for i in range(1,5)]
discretize_clf = discretize.QuantileDiscretizer(feature_names=columns_continous,quantiles=quantiles)
discretize_clf.fit(data_loan)
discretize_clf.cuts
data_loan[columns_continous] = discretize_clf.transform(data_loan[columns_continous])
data_loan = data_loan.replace(np.nan,-1)

woe_clf = continuous.WoeContinuous(feature_names=data_loan.columns[2:])
y = data_loan['overdue_m0']
data_loan_woe = woe_clf.fit_transform(data_loan,y)
iv = woe_clf.cal_iv()
sorted_iv = sorted(iv.items(), key=lambda x: x[1], reverse=True)
maps = woe_clf.maps

###筛选重要变量
#columns_imp = ['jrid','overdue_m0']+list({key for key,value in iv.items() if value>0.05})
#figs_woe = woe_clf.plot(close=True,show_last=False)
#figs_woe = {key:figs_woe[key] for key in columns_imp[2:]}
#outfile='./result/loan/'
#for feature in figs_woe:
#    figs_woe[feature].savefig(outfile+'%s.png'%feature)
    
columns_imp = ['jrid','overdue_m0','appfreq1m','applastday','appuserstatus','btcreditscore',
               'dlcactive90dflg','dlclastday','dlcprofit','dlctransamt',
               'finconsumeliab','memjrlasttime','memusertype','memxbscore',
               'sytpaynum1m','xbtacctamt','xbtacctlevel','xjkcurmoneyamt','xjkmoneyamtmax'
               ]
data_loan_adj = data_loan[columns_imp].copy()
data_loan_adj.shape #32070*19

###调整WOE和IV
data_loan_adj['xbtacctlevel'].replace(5,4,inplace=True)
data_loan_adj['memusertype'].replace(5,4,inplace=True)
data_loan_adj['sytpaynum1m'].replace(6,5,inplace=True)
data_loan_adj['xjkcurmoneyamt'].replace(5,4,inplace=True)
data_loan_adj['dlclastday'].replace(8,7,inplace=True)

woe_clf = continuous.WoeContinuous(feature_names=data_loan_adj.columns[2:])
y = data_loan_adj['overdue_m0']
data_loan_woe = woe_clf.fit_transform(data_loan_adj,y)
iv = woe_clf.cal_iv()
sorted_iv = sorted(iv.items(), key=lambda x: x[1], reverse=True)
maps = woe_clf.maps
fig_IV = woe_clf.plot_iv(top=0,rot=45,close=True)[1]
fig_IV.savefig('./result/loan/woe&iv/IVs.png')

figs_woe = woe_clf.plot(close=True,show_last=False)
figs_woe = {key:figs_woe[key] for key in columns_imp[2:]}
outfile='./result/loan/woe&iv/'
for feature in figs_woe:
    figs_woe[feature].savefig(outfile+'%s.png'%feature)

###训练逻辑回归
X_matrix = data_loan_woe[data_loan_woe.columns[2:]].as_matrix(columns=None)
Y_matrix = data_loan_woe['overdue_m0'].as_matrix(columns=None)

tuned_parameters = [{'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1]}]
clf = GridSearchCV(LogisticRegression(),tuned_parameters,cv=5,scoring='roc_auc')
clf.fit(X_matrix,Y_matrix)
clf.best_params_
lr = LogisticRegression(penalty='l2',C=0.01)
dict_Con = fit_cv(X_matrix=X_matrix,Y_matrix=Y_matrix,clf=lr,cv=5,random=1)
#{'auc': 0.73350877275377813, 'ks': 0.34804402148708086}

lr.intercept_
lr.coef_

###模型结果与逾期表现对比
data_loan_woe['Score'] = lr.predict_proba(X_matrix)[:,0]
fig = plot_hist_score(data_loan_woe['overdue_m0'],data_loan_woe['Score'])
fig.savefig('result/loan/score/score_lr_loan.png')

auc_fig = auc_calc(data_loan_woe,['Score'],['overdue_m0'])['auc_fig']
auc_fig.savefig('result/loan/auc/auc_lr_loan.png')

ks_fig = ks_calc(data_loan_woe,['Score'],['overdue_m0'])['ks_fig']
ks_fig.savefig('result/loan/ks/ks_lr_loan.png')

###小白分与逾期表现对比
data_loan_woe['xbscore'] = xbscore_loan
data_loan_woe['xbscore'] = data_loan_woe['xbscore'].fillna(data_loan_woe['xbscore'].mean())
fig = plot_hist_score(data_loan_woe['overdue_m0'],data_loan_woe['xbscore'])
fig.savefig('result/loan/score/score_xb_loan.png')

auc_fig = auc_calc(data_loan_woe,['xbscore'],['overdue_m0'])['auc_fig']
auc_fig.savefig('result/loan/auc/auc_xb_loan.png')

ks_fig = ks_calc(data_loan_woe,['xbscore'],['overdue_m0'])['ks_fig']
ks = ks_calc(data_loan_woe,['xbscore'],['overdue_m0'])['ks'] #0.10124113139498003
ks_fig.savefig('result/loan/ks/ks_xb_loan.png')

###白条分与逾期表现对比
data_loan_woe['btscore'] = btscore_loan
data_loan_woe['btscore'] = data_loan_woe['btscore'].fillna(data_loan_woe['btscore'].mean())
fig = plot_hist_score(data_loan_woe['overdue_m0'],data_loan_woe['btscore'])
fig.savefig('result/loan/score/score_bt_loan.png')

auc_fig = auc_calc(data_loan_woe,['btscore'],['overdue_m0'])['auc_fig']
auc_fig.savefig('result/loan/auc/auc_bt_loan.png')

ks_fig = ks_calc(data_loan_woe,['btscore'],['overdue_m0'])['ks_fig']
ks = ks_calc(data_loan_woe,['btscore'],['overdue_m0'])['ks'] #0.11012036543932513
ks_fig.savefig('result/loan/ks/ks_bt_loan.png')

###模型结果分数段好坏人占比
score_bins = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]
data_loan_woe['scorebin'] = pd.cut(data_loan_woe['Score'],bins=score_bins)
data_loan_score = pd.crosstab(data_loan_woe['scorebin'],data_loan_woe['overdue_m0'])
data_loan_score.columns = ['Good','Bad']
data_loan_score['BadRate'] = data_loan_score['Bad']/data_loan_score.sum(axis=1)
data_loan_score.to_csv('result/loan/score/badrate_lr_loan.csv',index=True)

#####随机森林
###数据预处理
data_loan = data_loan.replace(['N',-1,'-1',-9999,'-9999'],np.nan)
data_loan['xjkacctopenday'] = (datetime.now() - pd.to_datetime(data_loan['xjkacctopentime'].astype(str))).astype('timedelta64[D]')
data_loan['xjkopenday'] = (datetime.now() - pd.to_datetime(data_loan['xjkopentime'].astype(str))).astype('timedelta64[D]')
data_loan['sytfirstpayday'] = (datetime.now() - pd.to_datetime(data_loan['sytfirstpaytime'].astype(str))).astype('timedelta64[D]')
data_loan['applastday'] = (datetime.now() - pd.to_datetime(data_loan['applasttime'].astype(str))).astype('timedelta64[D]')
data_loan['userregday'] = (datetime.now() - pd.to_datetime(data_loan['userregtime'].astype(str))).astype('timedelta64[D]')
data_loan['sytfirstregday'] = (datetime.now() - pd.to_datetime(data_loan['sytfirstregtime'].astype(str))).astype('timedelta64[D]')
data_loan['sytfirstusedday'] = (datetime.now() - pd.to_datetime(data_loan['sytfirstusedtime'].astype(str))).astype('timedelta64[D]')
data_loan = data_loan.drop(['xjkacctopentime','xjkopentime','sytfirstpaytime','applasttime','userregtime','sytfirstregtime','sytfirstusedtime'],axis=1)
data_loan['appuserstatus'].replace('loanded',1,inplace=True)
data_loan['memdevice'].replace(['idfa','imei'],[1,2],inplace=True)
data_loan.shape #32070*244
data_loan.ix[:,2:] = data_loan.ix[:,2:].astype(float)

stat_describe_loan = DataAnalysis.stat_df(data_loan)
stat_describe_loan = stat_describe_loan.sort_values(by='missing_ratio')
#columns_select = stat_describe_loan[stat_describe_loan['missing_ratio']<0.5].index.tolist()
#data_loan = data_loan[columns_select] 
#以上两步可以合并成以下的一步
data_loan = data_loan.dropna(thresh=len(data_loan)*0.5,axis=1)
data_loan.shape #32070*150

data_loan = data_loan.replace(np.nan,-1)

###筛选重要变量
X_matrix = data_loan[data_loan.columns[2:]].as_matrix(columns=None)
Y_matrix = data_loan['overdue_m0'].as_matrix(columns=None)

tuned_parameters = [{'n_estimators':[100,150,200],'criterion':['gini','entropy'],'max_features':['sqrt','log2'],'max_depth':[10,15,20],'random_state':[0]}]
rf = GridSearchCV(RandomForestClassifier(),tuned_parameters,cv=5,scoring='roc_auc')
rf.fit(X_matrix,Y_matrix)
rf.best_params_
rf = RandomForestClassifier(n_estimators=200,criterion='entropy',max_features='sqrt',max_depth=10,random_state=0)
dict_Con = fit_cv(X_matrix=X_matrix,Y_matrix=Y_matrix,clf=rf,cv=5,random=1)
#{'auc': 0.74931440549053951, 'ks': 0.37086434222025333}

feature_importances = pd.DataFrame(rf.feature_importances_,columns=['importance'])
feature_importances['feature'] = data_loan.columns[2:].tolist()
feature_importances = feature_importances.sort_values(by='importance',ascending=False)

columns_imp = ['jrid','overdue_m0']+feature_importances.ix[feature_importances['importance']>0.01,'feature'].tolist()
data_loan_adj = data_loan[columns_imp].copy()
data_loan_adj.shape #32070*37

###训练随机森林
X_matrix = data_loan_adj[data_loan_adj.columns[2:]].as_matrix(columns=None)
Y_matrix = data_loan_adj['overdue_m0'].as_matrix(columns=None)

tuned_parameters = [{'n_estimators':[100,150,200],'criterion':['gini','entropy'],'max_features':['sqrt','log2'],'max_depth':[10,15,20],'random_state':[0]}]
rf = GridSearchCV(RandomForestClassifier(),tuned_parameters,cv=5,scoring='roc_auc')
rf.fit(X_matrix,Y_matrix)
rf.best_params_
rf = RandomForestClassifier(n_estimators=200,criterion='entropy',max_features='sqrt',max_depth=10,random_state=0)
dict_Con = fit_cv(X_matrix=X_matrix,Y_matrix=Y_matrix,clf=rf,cv=5,random=1)
#{'auc': 0.74957765704824864, 'ks': 0.37061962502705448}

###模型结果与逾期表现对比
data_loan_adj['Score'] = rf.predict_proba(X_matrix)[:,0]
fig = plot_hist_score(data_loan_adj['overdue_m0'],data_loan_adj['Score'])
fig.savefig('result/loan/score/score_rf_loan.png')

auc_fig = auc_calc(data_loan_adj,['Score'],['overdue_m0'])['auc_fig']
auc_fig.savefig('result/loan/auc/auc_rf_loan.png')

ks_fig = ks_calc(data_loan_adj,['Score'],['overdue_m0'])['ks_fig']
ks = ks_calc(data_loan_adj,['Score'],['overdue_m0'])['ks'] #0.63208815162040188
ks_fig.savefig('result/loan/ks/ks_rf_loan.png')

###模型结果分数段好坏人占比
score_bins = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
data_loan_adj['scorebin'] = pd.cut(data_loan_adj['Score'],bins=score_bins)
data_loan_score = pd.crosstab(data_loan_adj['scorebin'],data_loan_adj['overdue_m0'])
data_loan_score.columns = ['Good','Bad']
data_loan_score['BadRate'] = data_loan_score['Bad']/data_loan_score.sum(axis=1)
data_loan_score.to_csv('result/loan/score/badrate_rf_loan.csv',index=True)

###################################银行建模#########################################
#####逻辑回归
###数据预处理
data_bank = data_bank.replace(['N',-1,'-1',-9999,'-9999'],np.nan)
data_bank['xjkacctopenday'] = (datetime.now() - pd.to_datetime(data_bank['xjkacctopentime'].astype(str))).astype('timedelta64[D]')
data_bank['xjkopenday'] = (datetime.now() - pd.to_datetime(data_bank['xjkopentime'].astype(str))).astype('timedelta64[D]')
data_bank['sytfirstpayday'] = (datetime.now() - pd.to_datetime(data_bank['sytfirstpaytime'].astype(str))).astype('timedelta64[D]')
data_bank['applastday'] = (datetime.now() - pd.to_datetime(data_bank['applasttime'].astype(str))).astype('timedelta64[D]')
data_bank['userregday'] = (datetime.now() - pd.to_datetime(data_bank['userregtime'].astype(str))).astype('timedelta64[D]')
data_bank['sytfirstregday'] = (datetime.now() - pd.to_datetime(data_bank['sytfirstregtime'].astype(str))).astype('timedelta64[D]')
data_bank['sytfirstusedday'] = (datetime.now() - pd.to_datetime(data_bank['sytfirstusedtime'].astype(str))).astype('timedelta64[D]')
data_bank = data_bank.drop(['xjkacctopentime','xjkopentime','sytfirstpaytime','applasttime','userregtime','sytfirstregtime','sytfirstusedtime'],axis=1)
data_bank['appuserstatus'].replace('loanded',1,inplace=True)
data_bank['memdevice'].replace(['idfa','imei'],[1,2],inplace=True)
data_bank.shape #71749*244
data_bank.ix[:,2:] = data_bank.ix[:,2:].astype(float)

btscore_bank = data_bank['btcreditscore']
xbscore_bank = data_bank['memxbscore']

stat_describe_bank = DataAnalysis.stat_df(data_bank)
stat_describe_bank = stat_describe_bank.sort_values(by='missing_ratio')
columns_select = stat_describe_bank[stat_describe_bank['missing_ratio']<0.5].index.tolist()
data_bank = data_bank[columns_select] 
data_bank.shape #71749*150

###计算WOE和IV
stat_describe_bank = DataAnalysis.stat_df(data_bank.ix[:,2:])
stat_describe_bank = stat_describe_bank.sort_values(by='count_unique')
columns_continous = stat_describe_bank[stat_describe_bank['count_unique']>13].index.tolist()
quantiles = [20*i  for i in range(1,5)]
discretize_clf = discretize.QuantileDiscretizer(feature_names=columns_continous,quantiles=quantiles)
discretize_clf.fit(data_bank)
discretize_clf.cuts
data_bank[columns_continous] = discretize_clf.transform(data_bank[columns_continous])
data_bank = data_bank.replace(np.nan,-1)

woe_clf = continuous.WoeContinuous(feature_names=data_bank.columns[2:])
y = data_bank['overdue_m0']
data_bank_woe = woe_clf.fit_transform(data_bank,y)
iv = woe_clf.cal_iv()
sorted_iv = sorted(iv.items(), key=lambda x: x[1], reverse=True)
maps = woe_clf.maps

###筛选重要变量
#columns_imp = ['jrid','overdue_m0']+list({key for key,value in iv.items() if value>0.05})
#figs_woe = woe_clf.plot(close=True,show_last=False)
#figs_woe = {key:figs_woe[key] for key in columns_imp[2:]}
#outfile='./result/bank/'
#for feature in figs_woe:
#    figs_woe[feature].savefig(outfile+'%s.png'%feature)
    
columns_imp = ['jrid','overdue_m0','appfreq1m','applastday','btcreditscore',
               'dlclastday','dlctransnum12m','finconsumeliab','lmksplitflg',
               'memjrlasttime','memusertype','sytpaynum2m',
               'xbtacctamt','xbtacctlevel','xjkcurmoneyamt','xjkmoneyamtmax'
               ]
data_bank_adj = data_bank[columns_imp].copy()
data_bank_adj.shape #71749*16

###调整WOE和IV
data_bank_adj['dlclastday'].replace(8,7,inplace=True)
data_bank_adj['memusertype'].replace(5,4,inplace=True)
data_bank_adj['xbtacctlevel'].replace(5,4,inplace=True)

woe_clf = continuous.WoeContinuous(feature_names=data_bank_adj.columns[2:])
y = data_bank_adj['overdue_m0']
data_bank_woe = woe_clf.fit_transform(data_bank_adj,y)
iv = woe_clf.cal_iv()
sorted_iv = sorted(iv.items(), key=lambda x: x[1], reverse=True)
maps = woe_clf.maps
fig_IV = woe_clf.plot_iv(top=0,rot=45,close=True)[1]
fig_IV.savefig('./result/bank/woe&iv/IVs.png')

figs_woe = woe_clf.plot(close=True,show_last=False)
figs_woe = {key:figs_woe[key] for key in columns_imp[2:]}
outfile='./result/bank/woe&iv/'
for feature in figs_woe:
    figs_woe[feature].savefig(outfile+'%s.png'%feature)

###训练逻辑回归
X_matrix = data_bank_woe[data_bank_woe.columns[2:]].as_matrix(columns=None)
Y_matrix = data_bank_woe['overdue_m0'].as_matrix(columns=None)

tuned_parameters = [{'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1]}]
clf = GridSearchCV(LogisticRegression(),tuned_parameters,cv=5,scoring='roc_auc')
clf.fit(X_matrix,Y_matrix)
clf.best_params_
lr = LogisticRegression(penalty='l2',C=0.01)
dict_Con = fit_cv(X_matrix=X_matrix,Y_matrix=Y_matrix,clf=lr,cv=5,random=1)
#{'auc': 0.7019068430239469, 'ks': 0.30449456415373699}

lr.intercept_
lr.coef_

###模型结果与逾期表现对比
data_bank_woe['Score'] = lr.predict_proba(X_matrix)[:,0]
fig = plot_hist_score(data_bank_woe['overdue_m0'],data_bank_woe['Score'])
fig.savefig('result/bank/score/score_lr_bank.png')

auc_fig = auc_calc(data_bank_woe,['Score'],['overdue_m0'])['auc_fig']
auc_fig.savefig('result/bank/auc/auc_lr_bank.png')

ks_fig = ks_calc(data_bank_woe,['Score'],['overdue_m0'])['ks_fig']
ks_fig.savefig('result/bank/ks/ks_lr_bank.png')

###小白分与逾期表现对比
data_bank_woe['xbscore'] = xbscore_bank
data_bank_woe['xbscore'] = data_bank_woe['xbscore'].fillna(data_bank_woe['xbscore'].mean())
fig = plot_hist_score(data_bank_woe['overdue_m0'],data_bank_woe['xbscore'])
fig.savefig('result/bank/score/score_xb_bank.png')

auc_fig = auc_calc(data_bank_woe,['xbscore'],['overdue_m0'])['auc_fig']
auc_fig.savefig('result/bank/auc/auc_xb_bank.png')

ks_fig = ks_calc(data_bank_woe,['xbscore'],['overdue_m0'])['ks_fig']
ks = ks_calc(data_bank_woe,['xbscore'],['overdue_m0'])['ks'] #0.075969544815349677
ks_fig.savefig('result/bank/ks/ks_xb_bank.png')

###白条分与逾期表现对比
data_bank_woe['btscore'] = btscore_bank
data_bank_woe['btscore'] = data_bank_woe['btscore'].fillna(data_bank_woe['btscore'].mean())
fig = plot_hist_score(data_bank_woe['overdue_m0'],data_bank_woe['btscore'])
fig.savefig('result/bank/score/score_bt_bank.png')

auc_fig = auc_calc(data_bank_woe,['btscore'],['overdue_m0'])['auc_fig']
auc_fig.savefig('result/bank/auc/auc_bt_bank.png')

ks_fig = ks_calc(data_bank_woe,['btscore'],['overdue_m0'])['ks_fig']
ks = ks_calc(data_bank_woe,['btscore'],['overdue_m0'])['ks'] #0.071581772230096152
ks_fig.savefig('result/bank/ks/ks_bt_bank.png')

###模型结果分数段好坏人占比
score_bins = [0.7,0.8,0.9,1.0]
data_bank_woe['scorebin'] = pd.cut(data_bank_woe['Score'],bins=score_bins)
data_bank_score = pd.crosstab(data_bank_woe['scorebin'],data_bank_woe['overdue_m0'])
data_bank_score.columns = ['Good','Bad']
data_bank_score['BadRate'] = data_bank_score['Bad']/data_bank_score.sum(axis=1)
data_bank_score.to_csv('result/bank/score/badrate_lr_bank.csv',index=True)

#####随机森林
###数据预处理
data_bank = data_bank.replace(['N',-1,'-1',-9999,'-9999'],np.nan)
data_bank['xjkacctopenday'] = (datetime.now() - pd.to_datetime(data_bank['xjkacctopentime'].astype(str))).astype('timedelta64[D]')
data_bank['xjkopenday'] = (datetime.now() - pd.to_datetime(data_bank['xjkopentime'].astype(str))).astype('timedelta64[D]')
data_bank['sytfirstpayday'] = (datetime.now() - pd.to_datetime(data_bank['sytfirstpaytime'].astype(str))).astype('timedelta64[D]')
data_bank['applastday'] = (datetime.now() - pd.to_datetime(data_bank['applasttime'].astype(str))).astype('timedelta64[D]')
data_bank['userregday'] = (datetime.now() - pd.to_datetime(data_bank['userregtime'].astype(str))).astype('timedelta64[D]')
data_bank['sytfirstregday'] = (datetime.now() - pd.to_datetime(data_bank['sytfirstregtime'].astype(str))).astype('timedelta64[D]')
data_bank['sytfirstusedday'] = (datetime.now() - pd.to_datetime(data_bank['sytfirstusedtime'].astype(str))).astype('timedelta64[D]')
data_bank = data_bank.drop(['xjkacctopentime','xjkopentime','sytfirstpaytime','applasttime','userregtime','sytfirstregtime','sytfirstusedtime'],axis=1)
data_bank['appuserstatus'].replace('loanded',1,inplace=True)
data_bank['memdevice'].replace(['idfa','imei'],[1,2],inplace=True)
data_bank.shape #32070*244
data_bank.ix[:,2:] = data_bank.ix[:,2:].astype(float)

stat_describe_bank = DataAnalysis.stat_df(data_bank)
stat_describe_bank = stat_describe_bank.sort_values(by='missing_ratio')
columns_select = stat_describe_bank[stat_describe_bank['missing_ratio']<0.5].index.tolist()
data_bank = data_bank[columns_select] 
data_bank.shape #32070*150

data_bank = data_bank.replace(np.nan,-1)

###筛选重要变量
X_matrix = data_bank[data_bank.columns[2:]].as_matrix(columns=None)
Y_matrix = data_bank['overdue_m0'].as_matrix(columns=None)

tuned_parameters = [{'n_estimators':[100,150,200],'criterion':['gini','entropy'],'max_features':['sqrt','log2'],'max_depth':[10,15,20],'random_state':[0]}]
rf = GridSearchCV(RandomForestClassifier(),tuned_parameters,cv=5,scoring='roc_auc')
rf.fit(X_matrix,Y_matrix)
rf.best_params_
rf = RandomForestClassifier(n_estimators=200,criterion='entropy',max_features='sqrt',max_depth=10,random_state=0)
dict_Con = fit_cv(X_matrix=X_matrix,Y_matrix=Y_matrix,clf=rf,cv=5,random=1)
#{'auc': 0.73971588175007841, 'ks': 0.35104906048330864}

feature_importances = pd.DataFrame(rf.feature_importances_,columns=['importance'])
feature_importances['feature'] = data_bank.columns[2:].tolist()
feature_importances = feature_importances.sort_values(by='importance',ascending=False)

columns_imp = ['jrid','overdue_m0']+feature_importances.ix[feature_importances['importance']>0.01,'feature'].tolist()
data_bank_adj = data_bank[columns_imp].copy()
data_bank_adj.shape #32070*38

###训练随机森林
X_matrix = data_bank_adj[data_bank_adj.columns[2:]].as_matrix(columns=None)
Y_matrix = data_bank_adj['overdue_m0'].as_matrix(columns=None)

tuned_parameters = [{'n_estimators':[100,150,200],'criterion':['gini','entropy'],'max_features':['sqrt','log2'],'max_depth':[10,15,20],'random_state':[0]}]
rf = GridSearchCV(RandomForestClassifier(),tuned_parameters,cv=5,scoring='roc_auc')
rf.fit(X_matrix,Y_matrix)
rf.best_params_
rf = RandomForestClassifier(n_estimators=200,criterion='entropy',max_features='sqrt',max_depth=10,random_state=0)
dict_Con = fit_cv(X_matrix=X_matrix,Y_matrix=Y_matrix,clf=rf,cv=5,random=1)
#{'auc': 0.73942190356183291, 'ks': 0.35598968157278882}

#skf = StratifiedKFold(y=Y_matrix,n_folds=5,random_state=1)
#auc_list_train = []
#ks_list_train = []
#cv_dict_train = {}
#auc_list_test = []
#ks_list_test = []
#cv_dict_test = {}
#for train_index,test_index in skf:
#    X_train,X_test = X_matrix[train_index],X_matrix[test_index]
#    Y_train,Y_test = Y_matrix[train_index],Y_matrix[test_index]
#    fit_data_test = pd.DataFrame(index=range(len(test_index)),columns=['Y','Prob'])
#    fit_data_train = pd.DataFrame(index=range(len(train_index)),columns=['Y','Prob'])
#    rf = rf.fit(X_train,Y_train)
#    fit_data_test['Y'] = Y_test
#    fit_data_test['Prob'] = rf.predict_proba(X_test)[:,0]
#    fit_data_train['Y'] = Y_train
#    fit_data_train['Prob'] = rf.predict_proba(X_train)[:,0]    
#    score_col = ['Prob']
#    class_col = ['Y']
#    auc_ = auc_calc(fit_data_test,score_col,class_col)['auc']
#    auc_list_test.append(auc_)
#    auc_ = auc_calc(fit_data_train,score_col,class_col)['auc']
#    auc_list_train.append(auc_)
#    ks_ = ks_calc(fit_data_test,score_col,class_col)['ks']
#    ks_list_test.append(ks_)
#    ks_ = ks_calc(fit_data_train,score_col,class_col)['ks']
#    ks_list_train.append(ks_)
#    cv_dict_test['ks'] = np.mean(ks_list_test)
#    cv_dict_test['auc'] = np.mean(auc_list_test)
#    cv_dict_train['ks'] = np.mean(ks_list_train)
#    cv_dict_train['auc'] = np.mean(auc_list_train)

###模型结果与逾期表现对比
data_bank_adj['Score'] = rf.predict_proba(X_matrix)[:,0]
fig = plot_hist_score(data_bank_adj['overdue_m0'],data_bank_adj['Score'])
fig.savefig('result/bank/score/score_rf_bank.png')

auc_fig = auc_calc(data_bank_adj,['Score'],['overdue_m0'])['auc_fig']
auc_fig.savefig('result/bank/auc/auc_rf_bank.png')

ks_fig = ks_calc(data_bank_adj,['Score'],['overdue_m0'])['ks_fig']
ks = ks_calc(data_bank_adj,['Score'],['overdue_m0'])['ks'] #0.66491258437082479
ks_fig.savefig('result/bank/ks/ks_rf_bank.png')

###模型结果分数段好坏人占比
score_bins = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]
data_bank_adj['scorebin'] = pd.cut(data_bank_adj['Score'],bins=score_bins)
data_bank_score = pd.crosstab(data_bank_adj['scorebin'],data_bank_adj['overdue_m0'])
data_bank_score.columns = ['Good','Bad']
data_bank_score['BadRate'] = data_bank_score['Bad']/data_bank_score.sum(axis=1)
data_bank_score.to_csv('result/bank/score/badrate_rf_bank.csv',index=True)


