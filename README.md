### 1. small_task:常用分类模型统一建模标准
1.设置常用的分类模型，目前有以下几类模型，可通过加函数的方式添加想加入的分类模型进行训练
 #初步选取的模型
    test_classifiers = ['NB', 'KNN', 'LR','DT','SVM', 'RF']  </br>
    classifiers = {'NB':naive_bayes_classifier,   </br>
                   'KNN':knn_classifier,  </br>
                    'LR':logistic_regression_classifier,  </br>
                    'DT':decision_tree_classifier,    </br>
                   'SVM':svm_classifier,    </br>
                   'RF':random_forest_classifier,   </br>
                   'AB':AdaBoost_classifier,    </br>
                    'GBDT':GBDT_classifier,    </br>
                 'XGB':XGBoost_classifier   </br>
        }


2.对对应的模型进行调参，调参通常是经验参数范畴

3.对对应的模型的效果：学习曲线、ROC曲线、KS值以及好坏样本的分布进行输出保存



### 2.xgb_tune_param_norm : xgboosting调参全流程标准化
涉及模型： pipline + gridsearch + xgboost
+ step1: 设置变量进入模型的最低标准
+ step2: clf__max_depth-list(range(4,12,2));clf__min_child_weight-list(range(1,8,2))
+ step3: clf__gamma-[[i/10.0 for i in range(0,5)]]
+ step4: clf__subsample-[i/10.0 for i in range(6,10)];clf__colsample_bytree-[i/10.0 for i in range(6,10)]
+ step5: clf__reg_alpha-[1e-5, 1e-2, 0.1, 1, 100]
+ step6: 调低学习率，调高学习器的个数
