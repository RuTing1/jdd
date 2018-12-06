# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:05:52 2018

@author: dingru1
"""
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing

#__all__ = ['TaskSpecDef', 'TaskSpec', 'DistributedSession', 'StopAtTimeHook', 'LoadCheckpoint']


class DataProcessing(object):
    """Specification for a data processing task.
    It contains many methods to help you process your data. you can use it by import preprocessing import DataProcessing first,
    this script includes methods as follows:
        delete_high_missing_ratio_variables #缺失率变量剔除
        deal_with_time_variables  #时间类变量处理
        get_variable_type #分类变量连续变量区分
        unify_variables #变量的归一化处理
        delete_low_variance_variables #低方差变量剔除，变量的方差低，表明变量的信息含量低
        reduce_vars_corr # 降低变量之间的相关性，根据可设定的阈值对相关性强的变量，只保留其中之一
        
    
    Parameters
    ----------
    no_act_vars: str or list
        such as ['IDcol','target'], mark for variables which you want to exclude from dataprocessing.

 
    Notes
    ----------
    master might not be included in TF_CONFIG and can be None. The shard_index is adjusted
    in any case to assign 0 to master and >= 1 to workers.
    This implementation doesn't support sparse arrays in the `TF_CONFIG` variable as the
    official TensorFlow documentation shows, as it is not a supported by the json
    definition.
    
    
    Examples
    --------
    A simple example for distributed training where all the workers use the same dataset:
    >>> from preprocessing import DataProcessing
    >>> data_process = DataProcessing(data,no_act_vars)  #实例化
    >>> data_low_missing_ratio = data_process.delete_high_missing_ratio_variables(data,threshold=0.5) #高缺失率变量剔除
    >>> data_time_processsed = data_process.deal_with_time_variables(data,time_vars, end_date='2018-10-31') #时间变量处理
    >>> categorical_cols,continue_cols = data_process.get_variable_type(data,threshold=100)
    >>> data_unified = data_process.unify_variables(data,method='MinMaxScaler')
    >>> data_high_variance = data_process.delete_low_variance_variables(data,threshold=0.01)
    >>> data_reduce_relation = data_process.reduce_vars_corr(data,categorical_cols,cor_type='pearson', threshold=0.9)
    
    """
    
        
    def __init__(self, no_act_vars=None):
        self.no_act_vars = no_act_vars
        
    
    #函数运行时间装饰器
    def time_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            end_time = time.time()
            print('--> RUN TIME: <%s> : %s' % (func.__name__, end_time - start_time))
            return res
        return wrapper
            
    
    #变量缺失率分析
    @time_decorator
    def delete_high_missing_ratio_variables(self, data, threshold=0.5):
        """Return data which low missing ratio columns.
         Parameters
         ----------
         data : dataframe
         fix_cols: list
             default None,variables that should exclude from this task.
         threshold: int
             default 0.5,retain variables should with missing ration less than threshold.
    
         Examples
         --------
         >>> data_low_missing_ratio = delete_high_missing_ratio_variables(0.5)
    
        """   
        fix_cols = self.no_act_vars
        assert len(data) >0, 'data is empty.'
        
        retain_cols = []
        data_cnt = len(data)
        
        if fix_cols is None:
            cols = [col for col in data.columns ]
        else:  
            assert type(fix_cols) == list or type(fix_cols)== str, 'fix_cols should be a list or str.'
            cols = [col for col in data.columns if col not in fix_cols]
            
        for col in cols:
            count = data[col].count()
            if round(count*1.0/data_cnt,3)>threshold:
                retain_cols.append(col)
                
        if fix_cols is not None:
            col = retain_cols.extend(fix_cols)
            
        data = data.loc[:,retain_cols]
        return data
    
    
    #时间类变量处理
    @time_decorator
    def deal_with_time_variables(self, data, time_vars, end_date='2018-10-31'):
        """Return the data with datetime data processed.
         Parameters
         ----------
         data : dataframe
         time_vars: str or list
             datetime variables that should deal .
         end_date: str
             default '2018-10-31',the date used to caculate diff between it and target date variables.
    
         Examples
         --------
         >>> data_time_processsed = deal_with_time_variables(time_vars, '2018-10-31')
         
        """
        assert time_vars is not None, 'There is no datetime variables to process.'
        
        
        if type(time_vars == str):
            data[time_vars] = (pd.to_datetime(end_date)- pd.to_datetime(data[time_vars]))/np.timedelta64(1,'D')
        elif type(time_vars == list):
            for col in time_vars:
                data[col] = (pd.to_datetime(end_date)- pd.to_datetime(data[col]))/np.timedelta64(1,'D')
        else:
            print('Please check the type of time_vars, it should be a str or a list.')
        return data
    
    
    #变量类型区分
    @time_decorator
    def get_variable_type(self, data, threshold=200):
        """Return the list of continue and catigorical variables for the data.
         Parameters
         ----------
         data : dataframe
         fix_cols: list
             default None,variables that should exclude from this task.
         threshold: int
             default 200,catigorical variables should contains values less than threshold.
    
         Examples
         --------
         >>> categorical_cols,continue_cols = get_variable_type(threshold=100)
         
        """
        fix_cols = self.no_act_vars        
        assert len(data) >0, 'Data is empty.'
        
        
        if fix_cols is None:
            cols = [col for col in data.columns ]
        else:  
            cols = [col for col in data.columns if col not in fix_cols]
            
        continue_cols, categorical_cols = [], []
        
        
        for col in cols:
            if len(data[col].value_counts()) < threshold:
                categorical_cols.append(col)
            else:
                continue_cols.append(col)    
        return categorical_cols,continue_cols
    
    
    #变量归一化
    @time_decorator
    def unify_variables (self, data, method='MinMaxScaler'):
        """Return data which low missing ratio columns.
         Parameters
         ----------
         data: dataframe
         method: [scale, RobustScaler, Normalization, MinMaxScaler]
             scale: Gaussian with zero mean and unit variance
             RobustScaler: Scaling data with outliers
             Normalization: scaling individual samples to have unit norm
             MinMaxScaler: Scaling features to a range
                 
         fix_cols: list
             default None,variables that should exclude from this task.
    
         Examples
         --------
         >>> data_unified = unify_variables(method='MinMaxScaler')
         
        """
        fix_cols = self.no_act_vars
        methods = ['scale', 'RobustScaler', 'Normalization', 'MinMaxScaler']
        
        assert len(data) > 0, 'Data is empty.'
        assert method in methods,'Please choose method from  scale|RobustScaler|Normalization|MinMaxScaler .'
        
        
        if fix_cols is None:
            cols = [col for col in data.columns ]
        else:  
            assert type(fix_cols) == list ,'fix_cols should be a list.'
            cols = [col for col in data.columns if col not in fix_cols]
        data_x = data.loc[:,cols]
        
        
        if method == 'scale':
            data_scaled = preprocessing.scale(data_x)
        elif method == 'RobustScaler':
            transformer = preprocessing.RobustScaler().fit(data_x)
            data_scaled = transformer.transform(data_x)
        elif method == 'Normalization':  
            data_scaled = preprocessing.normalize(data_x, norm='l2') # norm 可以选l1或者l2
        else:
            transformer = preprocessing.MinMaxScaler().fit(data_x)
            data_scaled = transformer.transform(data_x)
            
            
        data_scaled = pd.DataFrame(data_scaled)
        data_scaled.columns = cols
        
        
        if fix_cols is not None:
            for col in fix_cols:
                data_scaled[col] = data[col].values
        return data_scaled
    

    #低方差变量剔除
    @time_decorator
    def delete_low_variance_variables(self, data, threshold=0.1):
        """Return data which low missing ratio columns.
         Parameters
         ----------
         data : dataframe
             advice scaling data first.
         fix_cols: list
             default None,variables that should exclude from this task.
         threshold: int
             default 0.5,retain variables should with high variance over than threshold.
    
         Examples
         --------
         >>> data_high_variance = delete_low_variance_variables(threshold=0.01)
         
         """
        fix_cols = self.no_act_vars
        retain_cols = []
        
        assert len(data) >0, 'Data is empty.'
        
        #方差计算上首先进行变量的归一化，方便选取统一的方差阈值进行阈值筛选
#        data = self.unify_variables(method=)
        
        if fix_cols is None:
            cols = [col for col in data.columns ]
        else:  
            cols = [col for col in data.columns if col not in fix_cols]
            
        for col in cols:
            var = np.var(data[col])
    #         print(var)
            if var > threshold:
                retain_cols.append(col)
                
        if fix_cols is not None:
            retain_cols.extend(fix_cols)
            
        data = data.loc[:,retain_cols]
        return data
    
    
    #变量相关性
    @time_decorator
    def reduce_vars_corr(self, data, variables, cor_type='pearson', threshold=0.9):
        
        """Return data with high relevance variables delete.
    
    
    Parameters
         ----------
         data : dataframe
             advice scaling data first.
         variables: list
             variable list for processing.
         cor_type: str
             defult pearson, the correlation type of variables, should choose from ['pearson', 'kendall', 'spearman']
             normally,continue columns for pearson type, categrical columns for kendall type.
         threshold: positive int
             default 0.9,should delete one of the high correlation variable.
    
         Examples
         --------
         >>> data_reduce_relation = reduce_vars_corr(variables,cor_type='pearson', threshold=0.9)
         
        """
        drop_cols = []
        cor_types = ['pearson', 'kendall', 'spearman']
        data_cor = data.loc[:,variables].corr(cor_type)
        
        assert cor_type in cor_types, 'Please choose cor_type from pearson|kendall|spearman'
        assert len(variables) >=2, 'The num of data columns should over 2.'
    
        for col in variables:
            if col in drop_cols:
                continue
            drop_col = list(data_cor.loc[data_cor[col]>threshold,:].index)
            drop_col = [newcol for newcol in drop_col if newcol not in [col]]
            data_cor.drop(drop_col,axis=1, inplace=True)
            data_cor.drop(drop_col,axis=0, inplace=True)
            drop_cols.extend(drop_col)
         
        print('Delete columns as follows:')
        print(drop_cols)
        new_data = data.drop(drop_cols, axis=1)
        return new_data
    
    

    
    
    
    
    
        
        
