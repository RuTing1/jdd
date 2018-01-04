# jdd
### 概率分和征信分分数映射
+ y_test['odds'] = y_test['prob']/(1-y_test['prob'])
+ y_test['score'] = (np.log(y_test['odds'])*(20/np.log(2))+600).astype(int)

+ y_test['score'].describe() #649-769

+ score_bins = list(range(640,800,20))
+ y_test['scorebin'] = pd.cut(y_test['score'],bins=score_bins)
+ y_test = pd.crosstab(y_test['scorebin'],y_test['overdue_m1'])
+ y_test.columns = ['Good','Bad']
+ y_test['BadRate'] = y_test['Bad']/y_test.sum(axis=1)

### 群体稳定性指标(population stability index)

+ 公式： psi = sum(（实际占比-预期占比）/ln(实际占比/预期占比))
+ 解释：如训练一个logistic回归模型，预测时候会有个概率输出p。你测试集上的输出设定为p1吧，将它从小到大排序后10等分，如0-0.1,0.1-0.2,......。
用这个模型对新的样本进行预测，预测结果叫p2,按p1的区间划分为10等分。实际占比就是p2上在各区间的用户占比，预期占比就是p1上各区间的用户占比。
+ 评估标准：一般认为psi小于0.1时候模型稳定性很高，0.1-0.25一般，大于0.25模型稳定性差，建议重做。

