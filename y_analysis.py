# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd    
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

###推算每笔贷款申请时间
def ApplyTimeCalc(data):
    '''
    功能: 推算每笔贷款的申请时间
    输入值:
    data: 二维数组或dataframe，包括每笔提现的还款日期
    输出值: 
    二维数组或dataframe，每笔提现的贷款申请日期和月份
    '''
    data['apply_time'] = min(data['repayment_time'])-pd.DateOffset(months=1)
    data['apply_month'] = data['apply_time'].apply(lambda x:datetime.strftime(x,'%Y%m'))
    return data

###计算滚动率
def RollRateAnalysis(data):
    '''
    功能: 计算滚动率，输出各个逾期占比以及转化率 
    输入值:
    data: 二维数组或dataframe，包括各个逾期标记
    输出值: 
    二维数组或dataframe，各个逾期占比以及转化率
    '''
    data['num_total'] = len(data)
    data['num_grey'] = data['overdue_grey'].sum()
    data['percent_grey'] = round(data['num_grey']/len(data),6)
    data['num_m0'] = data['overdue_m0'].sum()
    data['percent_m0'] = round(data['num_m0']/len(data),4)
    data['num_m1'] = data['overdue_m1'].sum()
    data['percent_m1'] = round(data['num_m1']/len(data),4)
    data['num_m2'] = data['overdue_m2'].sum()
    data['percent_m2'] = round(data['num_m2']/len(data),4)  
    data['num_m3'] = data['overdue_m3'].sum()
    data['percent_m3'] = round(data['num_m3']/len(data),4)
    data['rate_m0_m1'] = round(data['num_m1']/data['num_m0'],4)
    data['rate_m1_m2'] = round(data['num_m2']/data['num_m1'],4)
    data['rate_m2_m3'] = round(data['num_m3']/data['num_m2'],4)
    return data

###计算坏人捕获率
def BadCaptureRate(data,applyno_col,repayment_col,overdue_col):
    '''
    功能: 计算坏人捕获率，输出坏人捕获率矩阵和曲线
    输入值:
    data: 二维数组或dataframe，包括申请编号、还款期数、违约标记
    applyno_col: 一维数组或series，代表申请编号
    repayment_col: 一维数组或series，代表还款期数
    overdue_col: 一维数组或series，代表违约标记
    输出值: 
    字典，键值关系为{'bad_cumsum': 坏人捕获率矩阵，'fig': 坏人捕获率曲线}
    '''
    badcapturerate_dict = {}
    bad_cumsum = data[data[overdue_col[0]] == 1] 
    bad_cumsum = bad_cumsum.groupby(by=applyno_col[0],sort=False).first().reset_index()
    bad_cumsum = pd.DataFrame(bad_cumsum.groupby(by=repayment_col[0],sort=False)[overdue_col[0]].sum(),index=np.sort(data[repayment_col[0]].unique())).fillna(0)
    bad_cumsum['overdue_cumsum'] = bad_cumsum[overdue_col[0]].cumsum()
    bad_cumsum = bad_cumsum.drop([overdue_col[0]],axis=1)
    bad_cumsum.index.name = repayment_col[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bad_cumsum['overdue_cumsum'].plot(kind='line',ax=ax)
    ax.set_xlabel('%s' % repayment_col[0])
    ax.set_ylabel('Bad Number')
    ax.set_title('Bad Capture Rate')
    plt.close()
    badcapturerate_dict['bad_cumsum'] = bad_cumsum
    badcapturerate_dict['fig'] = fig
    return badcapturerate_dict

################################################################################################################
###读取表现数据
y_data = pd.read_csv('y_analysis/20171227/y_data.csv',encoding='iso-8859-1')
y_data = y_data[y_data['status']!='INIT']

###表现特征选择
columns_select = ['jrid','apply_no','product_id','period_count','repayment_index','repayment_time','overdue_m3','overdue_m2','overdue_m1','overdue_m0','overdue_grey']
data_select = y_data[columns_select]
columns_str = ['jrid','apply_no']
data_select[columns_str] = data_select[columns_str].astype(str)
data_select['repayment_time'] = pd.to_datetime(data_select['repayment_time'].astype(str))
data_select = data_select.sort_values(by=['apply_no','repayment_index']).drop_duplicates()
data_select = data_select.groupby(by='apply_no',sort=False).apply(ApplyTimeCalc).reset_index()
data_select = data_select.drop(['index'],axis=1)

###根据金融机构类型划分
product_loan = ['1','weixin_001']
#product_credit = ['msjr_001']
product_bank = ['zyxf_001','zlxf_001','njcb_001','bsb_001']
loan_select = data_select[data_select['product_id'].isin(product_loan)]
#credit_select = data_select[data_select['product_id'].isin(product_credit)]
bank_select = data_select[data_select['product_id'].isin(product_bank)]

###滚动率分析（Roll Rate Analysis）
columns_imp = ['num_total','num_grey','percent_grey','num_m0','percent_m0','rate_m0_m1','num_m1','percent_m1','rate_m1_m2','num_m2','percent_m2','rate_m2_m3','num_m3','percent_m3']

#小贷滚动率分析
groupby = loan_select.groupby(by='apply_no',sort=False).agg({'overdue_grey':'max','overdue_m0':'max','overdue_m1':'max','overdue_m2':'max','overdue_m3':'max','apply_month':'min'}).reset_index()
groupby = groupby.groupby(by='apply_month',sort=False).apply(RollRateAnalysis).reset_index()
roll_rate = groupby.groupby(by=['apply_month'],sort=False)[columns_imp].mean().reset_index().sort_values(by='apply_month')
roll_rate.to_excel('y_analysis/20171227/result/roll_rate_loan.xlsx',index=None,header=True)

#银行滚动率分析
groupby = bank_select.groupby(by='apply_no',sort=False).agg({'overdue_grey':'max','overdue_m0':'max','overdue_m1':'max','overdue_m2':'max','overdue_m3':'max','apply_month':'min'}).reset_index()
groupby = groupby.groupby(by='apply_month',sort=False).apply(RollRateAnalysis).reset_index()
roll_rate = groupby.groupby(by=['apply_month'],sort=False)[columns_imp].mean().reset_index().sort_values(by='apply_month')
roll_rate.to_excel('y_analysis/20171227/result/roll_rate_bank.xlsx',index=None,header=True)

###坏人捕获率分析（Bad Capture Rate）
#小贷坏人捕获率分析
loan_select.ix[loan_select['repayment_index'] == 9999,'repayment_index'] = loan_select.ix[loan_select['repayment_index'] == 9999,'period_count']
bad_cumsum_loan = BadCaptureRate(loan_select,['apply_no'],['repayment_index'],['overdue_m1'])['bad_cumsum']
fig_loan = BadCaptureRate(loan_select,['apply_no'],['repayment_index'],['overdue_m1'])['fig']
bad_cumsum_loan.to_excel('y_analysis/20171227/result/bad_capture_loan.xlsx',index=True,header=True)
fig_loan.savefig('y_analysis/20171227/result/bad_capture_loan.png') #3

#银行坏人捕获率分析
bank_select.ix[bank_select['repayment_index'] == 9999,'repayment_index'] = bank_select.ix[bank_select['repayment_index'] == 9999,'period_count']
bad_cumsum_bank = BadCaptureRate(bank_select,['apply_no'],['repayment_index'],['overdue_m1'])['bad_cumsum']
fig_bank = BadCaptureRate(bank_select,['apply_no'],['repayment_index'],['overdue_m1'])['fig']
bad_cumsum_bank.to_excel('y_analysis/20171227/result/bad_capture_bank.xlsx',index=True,header=True)
fig_bank.savefig('y_analysis/20171227/result/bad_capture_bank.png') #3

###保存小贷和银行的账户维度逾期信息
y_loan = loan_select.groupby(by=['apply_no','jrid'],sort=False)['overdue_grey','overdue_m0','overdue_m1','overdue_m2','overdue_m3','apply_month'].max().reset_index()
y_bank = bank_select.groupby(by=['apply_no','jrid'],sort=False)['overdue_grey','overdue_m0','overdue_m1','overdue_m2','overdue_m3','apply_month'].max().reset_index()

y_loan.to_csv('y_analysis/20171227/y_loan.csv',index=None,header=True)
y_bank.to_csv('y_analysis/20171227/y_bank.csv',index=None,header=True)


