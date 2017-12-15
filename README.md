# 常用分类模型统一建模标准
1.设置常用的分类模型，目前有以下几类模型，可通过加函数的方式添加想加入的分类模型进行训练
 #初步选取的模型
    test_classifiers = ['NB', 'KNN', 'LR','DT','SVM', 'RF'] 
    classifiers = {'NB':naive_bayes_classifier,   
                   'KNN':knn_classifier,  
                   'LR':logistic_regression_classifier,
                   'DT':decision_tree_classifier,
                   'SVM':svm_classifier, 
                   'RF':random_forest_classifier,
                   'AB':AdaBoost_classifier,
                   'GBDT':GBDT_classifier,
                   'XGB':XGBoost_classifier
        }


2.对对应的模型进行调参，调参通常是经验参数范畴

3.对对应的模型的效果：学习曲线、ROC曲线、KS值以及好坏样本的分布进行输出保存
