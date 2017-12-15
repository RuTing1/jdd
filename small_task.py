# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:42:12 2017

@author: dingru1
"""
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold



######################################选用模型,定义模型函数#####################################################

### naive_bayes
def naive_bayes_classifier(x_train, y_train,x_test,y_test): 
    
    from sklearn.naive_bayes import GaussianNB
    model_name = 'naive_bayes'
    nb = GaussianNB()
    dict_Con = fit_cv(X_matrix=x_train,Y_matrix=y_train,clf=nb,cv=5,random=random)
    print('the avg ks of {} is:'.format(model_name),dict_Con['ks'])
    print('the avg auc of {} is:'.format(model_name),dict_Con['auc'])
    test_y = nb.predict(x_test)
    y_test = pd.DataFrame(y_test).rename(columns={0:target})
    y_test['pred'] = nb.predict_proba(x_test)[:,1]
    model_eva(model_name,nb,x_train,y_train,y_test,test_y)
    
    return nb

### KNN
def knn_classifier(x_train, y_train,x_test,y_test):
    
    from sklearn.neighbors import KNeighborsClassifier 
    model_name = 'knn'
    knn = KNeighborsClassifier(p=2) 
    params = [{'n_neighbors':[pow(2,i) for i in range(1,5)],'p':[2,3]}]
    knn_best = modelfit(model_name,knn,params,x_train,y_train,x_test,y_test)
    return knn_best 

# LR
def logistic_regression_classifier(x_train, y_train,x_test,y_test):  
    
    from sklearn.linear_model import LogisticRegression  
    model_name = 'lr'
    lr = LogisticRegression(random_state=random)  
    params = [{'penalty':['l1','l2'],'C':[0.001,0.01,0.1,10,100]}]
    lr_best = modelfit(model_name,lr,params,x_train,y_train,x_test,y_test)
    return lr_best  

# Decision Tree Classifier  
def decision_tree_classifier(x_train, y_train,x_test,y_test):  
    
    from sklearn import tree  
    model_name = 'dt'
    dt = tree.DecisionTreeClassifier(max_features=None,random_state=random)
    params = [{'criterion':['gini','entropy'],'max_depth':list(range(4,8,1)),'min_samples_leaf':list(range(1,8,1))}]
    dt_best = modelfit(model_name,dt,params,x_train,y_train,x_test,y_test) 
    return dt_best  


# SVM Classifier  
def svm_classifier(x_train, y_train,x_test,y_test):  
    
    from sklearn.svm import SVC  
    model_name = 'svm'
    svm = SVC(kernel='rbf',random_state=random,probability =True)
    params = [{'C':[0.001,0.01,10,100],'class_weight':['auto',None]}]
    svm_best = modelfit(model_name,svm,params,x_train,y_train,x_test,y_test)
    return svm_best 

# Random Forest Classifier  
def random_forest_classifier(x_train, y_train,x_test,y_test):  
    
    from sklearn.ensemble import RandomForestClassifier  
    model_name = 'rf'
    rf = RandomForestClassifier(n_estimators=100,random_state=random)
    params = [{'n_estimators':[120,300],'max_depth':[5,8,15,None],'max_features':['sqrt','log2',None]}]
#    params = [{'n_estimators':[120,300,500,800,1200],'max_depth':[5,8,15,25,30,None],'max_features':['sqrt','log2',None],
#              'min_samples_split':[1,2,5,10,15,100],'min_samples_leaf':[1,2,5,10]}]
    rf_best = modelfit(model_name,rf,params,x_train,y_train,x_test,y_test)
    return rf_best  

def AdaBoost_classifier(x_train, y_train,x_test,y_test): 
    
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    model_name = 'ab'
    ab = AdaBoostClassifier(n_estimators=50,learning_rate=1.0,random_state=random)
    params = [{'base_estimator':[DecisionTreeClassifier(),LogisticRegression()],'n_estimators':list(range(50,100,10))}]
    ad_best = modelfit(model_name,ab,params,x_train,y_train,x_test,y_test)
    return ad_best 

def GBDT_classifier(x_train, y_train,x_test,y_test):  
    
    from sklearn.ensemble import GradientBoostingClassifier 
    model_name = 'GBDT'
    GBDT = GradientBoostingClassifier(loss='deviance',random_state=random) 
    params = [{'n_estimators':[50,100],'learning_rate':[0.01,0.05,0.1],'subsample':[0.6,0.7,0.8,0.9,1.0],'max_depth':[5,8,15,25,30,None],
               'min_samples_leaf':[1,2,5,10],'max_features':['log2','sqrt',None]}]
    GBDT_best = modelfit(model_name,GBDT,params,x_train,y_train,x_test,y_test)  
    return GBDT_best  

def XGBoost_classifier(x_train, y_train,x_test,y_test):  
    
    import xgboost as xgb  
    model_name = 'XGBoost'
    xgbc = xgb.XGBClassifier(objective= 'binary:logistic',scale_pos_weight=1,seed=random) 
    params = [{'gamma':[0.05,0.1,0.3,0.5,0.7,0.9,1.0],'max_depth':[5,8,15,25,30,None],
               'min_samples_split':[2,5,10,15,100],'eta':[0.01,0.015,0.025,0.05,0.1],
               'min_samples_leaf':[1,2,5,10],'max_features':['log2','sqrt',None]}]
    xgb_best = modelfit(model_name,xgbc,params,x_train,y_train,x_test,y_test)  
    return xgb_best


#############################################评价模型，模型效果输出#################################################
###输出训练模型的学习曲线、ROC曲线和KS曲线
def model_eva(model_name,alg,x_train,y_train,y_test,test_y):
    """
    输入值：
    y_test：样本原本正负标志+模型预测的样本为正的概率
    test_y：模型预测的样本正负标志
    输出值：无
    """
#    print('===================the learning curve of the estimator is:===================')
    cv = ShuffleSplit(n_splits=10, test_size=test_size, random_state=0)
    plot_learning_curve(alg, 'the learning curve of {}'.format(model_name), x_train,y_train, (0.4, 1.01), cv=cv, n_jobs=2)
#    plt.show()
    
#    print('===================the ROC curve of the estimator is:===================')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test[target], y_test['pred'])
    roc_auc = auc(false_positive_rate, true_positive_rate)
#    plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc)  
    auc_dict = auc_calc(y_test,['pred'],[target])
    auc_dict['auc_fig'].savefig(file_dir+'auc_{}.png'.format(model_name))
    
    print("the auc of {} model is {}".format(model_name,roc_auc))
    print("the Confusion report is: ", classification_report(y_test[target], test_y))
    
#    print('===================the ks curve of the estimator is:=========================')
    ks_dict = ks_calc(y_test,['pred'],[target])
    ks_dict['ks_fig'].savefig(file_dir+'ks_{}.png'.format(model_name))
    
#    print('===================the distribution of good and bad is:=========================')
    fig = plot_hist_score(y_test[target],y_test['pred'])
    fig.savefig(file_dir + 'score_distribution_{}.png'.format(model_name))
    

####1.选最优参数；2.交叉验证；3调用模型评估函数
def modelfit(model_name,alg,parameters,x_train,y_train,x_test,y_test):   
    #模型调参
    clf = GridSearchCV(alg,parameters,cv=5,scoring='roc_auc')
#     x_train_0 = x_train.as_matrix(columns=None)
    clf.fit(x_train,y_train)
#    best_params = clf.best_params_
#     alg.set_params(**best_params)
    best_alg = clf.best_estimator_
    #模型5倍交叉验证训练
    dict_Con = fit_cv(X_matrix=x_train,Y_matrix=y_train,clf=best_alg,cv=5,random=1)
    print('the avg ks of {} is:'.format(model_name),dict_Con['ks'])
    print('the avg auc of {} is:'.format(model_name),dict_Con['auc'])
    test_y = best_alg.predict(x_test)
    y_test = pd.DataFrame(y_test).rename(columns={0:target})
    y_test['pred'] = best_alg.predict_proba(x_test)[:,1]
    model_eva(model_name,best_alg,x_train,y_train,y_test,test_y)
    
    return best_alg

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
#    fpr,tpr,threshold = roc_curve((1-data[class_col[0]]).ravel(),data[score_col[0]].ravel())   ###!!!!###
    fpr,tpr,threshold = roc_curve((data[class_col[0]]).ravel(),data[score_col[0]].ravel())
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
        fit_data['Prob'] = clf.predict_proba(X_test)[:,1]
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

###ROC曲线
#def plot_roc_curve(false_positive_rate, true_positive_rate,roc_auc):
#    
#    plt.figure()
#    lw = 2
#    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic example')
#    plt.legend(loc="lower right")
#    plt.show()
#    pass

###学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    fig = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")
    plt.close()
    fig.savefig(file_dir+'{}.png'.format(title))
    
    return plt


if __name__ == '__main__':
    
    # 全局变量
    random = 0 
    target = 'status'
    IDcol = 'null'
    test_size = 0.2
    version = 'v1.0/'
    file_dir = 'C:/Users/dingru1/Wed/small_task/'+version
    model_save = {} 
    # 获取数据
    data = pd.read_csv(file_dir+'rrd_data.csv')
    all_vars = [var for var in list(data.columns) if var not in [target,IDcol]]
    bin_vars = list(data.describe()['max':].T.reset_index()[data.describe()['max':].T.reset_index()['max']==1]['index'])
    cntu_vars = [var for var in data.columns if var not in bin_vars]
    
    y = data.loc[:,target].as_matrix(columns=None)
    X = data.loc[:,all_vars]
    X = X.drop(['hasChild'],axis=1).as_matrix(columns=None)
    X = StandardScaler().fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    #初步选取的模型
    test_classifiers = ['NB', 'KNN', 'LR','DT','SVM', 'RF'] 
    classifiers = {'NB':naive_bayes_classifier,   
                   'KNN':knn_classifier,  
                   'LR':logistic_regression_classifier,
                   'DT':decision_tree_classifier,
                   'SVM':svm_classifier, 
                   'RF':random_forest_classifier,
                   'AB':AdaBoost_classifier,
#                   'GBDT':GBDT_classifier,
#                   'XGB':XGBoost_classifier
        }
    
    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)  
        start_time = time.time()
        model = classifiers[classifier](x_train, y_train,x_test,y_test)
        model_save[classifier] = model
        end_time = time.time()
        print('the time used for {} model training is:'.format(classifier),round(end_time-start_time,4))

        
        
        
    
    

    
        