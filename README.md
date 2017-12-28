# jdd
###分数映射
"""
y_test['odds'] = y_test['prob']/(1-y_test['prob'])

"""
+ y_test['odds'] = y_test['prob']/(1-y_test['prob'])
+ y_test['score'] = (np.log(y_test['odds'])*(20/np.log(2))+600).astype(int)

+ y_test['score'].describe() #649-769

+ score_bins = list(range(640,800,20))
+ y_test['scorebin'] = pd.cut(y_test['score'],bins=score_bins)
+ y_test = pd.crosstab(y_test['scorebin'],y_test['overdue_m1'])
+ y_test.columns = ['Good','Bad']
+ y_test['BadRate'] = y_test['Bad']/y_test.sum(axis=1)

