# -*- coding: utf-8 -*-
import time
from pyspark.ml import Transformer
from pyspark.sql import functions as fn
from pyspark import since, keyword_only
from datetime import datetime, date, timedelta
from pyspark.sql.types import FloatType,IntegerType
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaTransformer, _jvm
from pyspark.sql.functions import lit,unix_timestamp,datediff,to_date,col,create_map

"""
@author: DR
@date: 2018-08-20
@desc: This script is used to provide some commom data processing modules in our modeling time.
@question: 
@v1版本使用说明：
 1.修改文件内的project_name = 'your_project_name' 
 2.每个模块的单独使用：
    from DataProcessSpark import *
    #重复数据处理
    from DataProcessSpark import DropDuplicate
    dd = DropDuplicate()
    df = dd.transform(df)
    #异常缺失值替换
    from DataProcessSpark import DataReplaceNa
    dr = DataReplaceNa()
    df = dr.transform(df)
    # 时间类特征处理
    from DataProcessSpark import DatetimePro
    dt = DatetimePro(inputcol= ['your time type features']) #inputcol 为时间类特征列表
    df = dt(df)
    # 高缺失率低方差数据剔除
    from DataProcessSpark import FilterLowMisHighStdFeatures
    flmh = FilterLowMisHighStdFeatures(IDcol, target, features, missing_ratio=0.5, std_threshold=1)
        # IDcol key字段，如jdpin
        # target 分类模型目标字段，如 is_buy
        # features 需要根据信息量进行筛选的特征list，默认为不包含IDcol和target的df.columns
        # missing_ratio:默认为0.5,如设置为a,则缺失率>=a的特征都会被过滤掉
        # std_threshold:默认为1,如设置为b,则方差小于b的特征都会被过滤掉
    df = flmh(df)
    #高共线性特征剔除
    from DataProcessSpark import DeleteCollineFeatures
    dcf = DeleteCollineFeatures(IDcol, target, colline_threshold=0.8)
        # IDcol key字段，如jdpin
        # target 分类模型目标字段，如 is_buy
        # colline_threshold:默认值0.8，即两个特征的皮尔逊相关性高于0.8时其中一个会被剔除，如果是分类问题，与target相关性较弱的会被剔除，如果是聚类问题，会随机剔除一个
    df = dcf(df)
    # 异常值处理
    from DataProcessSpark import DealOutliers
    do = DealOutliers(IDcol, target, continue_feas)
        # IDcol key字段，如jdpin
        # target 分类模型目标字段，如 is_buy
        # continue_feas:通常异常值处理都是指连续变量
    df = do(df)
 3. 结合Pipeline使用
    from DataProcessSpark import *
    from pyspark.ml import Pipeline
    continue_feas = ['amt_ziti_order', 'amt_intl_order']
    time_feas = ['fin_dlc_f0055']
    dd = DropDuplicate()
    dr = DataReplaceNa()
    dt = DatetimePro(inputcol= time_feas)
    flmh = FilterLowMisHighStdFeatures(IDcol, target, features=None, missing_ratio=0.5, std_threshold=1)
    dcf = DeleteCollineFeatures(IDcol, target, colline_threshold=0.8)
    do = DealOutliers(IDcol, target, continue_feas)
    pipeline = Pipeline(stages = [dd,dr,dt,flmh,dcf,do])  # 挑选自己需要的数据处理过程
    model = pipeline.fit(df)
    df = model.transform(df)

@数据存储：
由于以下几步耗时较久，会将处理完的数据进行存储
ft_tmp.{}_filter_lowMis_highstd_features:存储剔除低方差高缺失率特征的数据
ft_tmp.{}_delete_colline_features：存储共线性较高的且与目标变量相关性较低特征的数据
ft_tmp.{}_deal_outliers:存储异常值平滑处理的特征，将数值范围平滑至[0.25p-1.5*IQR,0.75p+1.5*IQR]
"""

project_name = 'your_project_name'

class DropDuplicate(JavaTransformer, JavaMLReadable, JavaMLWritable):
    """Test the dataframe to drop duplicate rows."""
    def __init__(self):
        super(DropDuplicate, self).__init__()
    def _transform(self, dataset):
        cnt_df = dataset.count()
        cnt_d_df = dataset.distinct().count()
        print('Count of rows: {0}'.format(cnt_df))
        print('Count of distinct rows: {0}'.format(cnt_d_df))

        if cnt_df==cnt_d_df:
            return dataset
        else:
            return dataset.dropDuplicates()


class DataReplaceNa(JavaTransformer, JavaMLReadable, JavaMLWritable):
    """replace values in to_replace to None in spark 2.1.1 using sql.functions.when"""
    def __init__(self):
        super(DataReplaceNa, self).__init__()
    #异常缺失值处理
    def _transform(self, dataset, to_replace=None, feature_list=None):
        start_time = time.time()
        if not feature_list:
            feature_list = dataset.columns
        if not to_replace:
            to_replace = ['N','\\N','-1',-1,'9999',9999,'-9999',-9999]
        for feature in feature_list:
            dataset = dataset.withColumn(feature,fn.when(~fn.col(feature).isin(to_replace),fn.col(feature)))
        end_time = time.time()
        print('time cost for DataReplaceNa transform process is:', end_time-start_time)
        return dataset


class DatetimePro(JavaTransformer, JavaMLReadable, JavaMLWritable):
    """caculate date diff,using apply date"""
    def __init__(self, inputcol=None):
        super(DatetimePro, self).__init__()
        self.inputcol = inputcol
    def _transform(self, dataset):
        start_time = time.time()
        in_col = self.inputcol
        yesterday = (date.today() + timedelta(days = -1)).strftime("%Y-%m-%d")    # 昨天日期
        dataset = dataset.withColumn('apply_date',fn.lit(yesterday))
        for col in in_col:
            dataset = dataset.withColumn(col,datediff(to_date(dataset["apply_date"]),to_date(dataset[col])))
            dataset = dataset.withColumn(col,fn.when(dataset[col]<0,0).otherwise(dataset[col]))
        dataset = dataset.drop('apply_date')
        end_time = time.time()
        print('time cost for DatetimePro transform process is:', end_time-start_time)
        return dataset


class FilterLowMisHighStdFeatures(JavaTransformer, JavaMLReadable, JavaMLWritable):
    """deleting features which missing ratio is higher and std is lower"""
    def __init__(self, IDcol, target, features, missing_ratio=0.5, std_threshold=1):
        super(FilterLowMisHighStdFeatures, self).__init__()
        self.IDcol = IDcol
        self.target = target
        self.features = features
        self.missing_ratio = missing_ratio
        self.std_threshold = std_threshold
    def _transform(self, dataset):
        start_time = time.time()
        IDcol = self.IDcol
        target = self.target
        features = self.features
        missing_ratio = self.missing_ratio
        std_threshold = self.std_threshold
        def datatypecast(df,features=None,dtype=IntegerType()):
            if not features:
                features = [col for col in df.columns if col not in [IDcol,target]]
            for feature in features:
                df = df.withColumn(feature,df[feature].cast(dtype))
            return df
        info_cols = []
        df_ = dataset.drop(IDcol).drop(target)
        cnt = df_.count()
        if not features:
            features = df_.columns
        df_desb = datatypecast(df_.describe())
        for col in features:
            if df_desb.select(col).collect()[0][0] > missing_ratio*cnt: #获得缺失率不超过阈值的特征
                #normlizer = Normalizer()  #改进方案：先归一化再求方差
                if df_desb.select(col).collect()[2][0] > std_threshold: #获得方差超过阈值的特征
                    info_cols.append(col)
        info_cols.append(IDcol)
        info_cols.append(target)
        dataset = dataset.select(info_cols)
        dataset.write.mode("overwrite").saveAsTable('ft_tmp.{}_filter_lowMis_highstd_features'.format(project_name))
        end_time = time.time()
        print('time cost for FilterLowMisHighStdFeatures transform process is:', end_time-start_time)
        return dataset


class DeleteCollineFeatures(JavaTransformer, JavaMLReadable, JavaMLWritable):
    """deleting features which missing ratio is higher and std is lower"""
    def __init__(self, IDcol, target, colline_threshold=0.8):
        super(DeleteCollineFeatures, self).__init__()
        self.IDcol = IDcol
        self.target = target
        self.colline_threshold = colline_threshold
    def _transform(self, dataset):
        start_time = time.time()
        IDcol = self.IDcol
        target = self.target
        colline_threshold = self.colline_threshold
        drop_cols,corr = [],[]
        cor_cols = [col for col in dataset.columns if col not in [IDcol,target]]
        len_cor = len(cor_cols)
        for i in range(0,len_cor):
            if cor_cols[i] in drop_cols:
                continue
            for j in range(i+1,len(cor_cols)):
                if cor_cols[j] in drop_cols:
                    continue
                corr_value = abs(dataset.corr(cor_cols[i],cor_cols[j],'pearson'))
                if target:
                    if corr_value > colline_threshold:
                        corr_i = abs(dataset.corr(cor_cols[i],target,'pearson'))
                        corr_j = abs(dataset.corr(cor_cols[j],target,'pearson'))
                        if corr_i>= corr_j:
                            drop_cols.append(cor_cols[j])
                        else:
                            drop_cols.append(cor_cols[i])  
                else:
                    drop_cols.append(cor_cols[j])
        info_cols = [col for col in dataset.columns if col not in drop_cols]  
        dataset = dataset.select(info_cols)
        dataset.write.mode("overwrite").saveAsTable('ft_tmp.{}_delete_colline_features'.format(project_name))
        end_time = time.time()
        print('time cost for DeleteCollineFeatures transform process is:', end_time-start_time)
        return dataset


class DealOutliers(JavaTransformer, JavaMLReadable, JavaMLWritable):
    """deleting features which missing ratio is higher and std is lower"""
    def __init__(self, IDcol, target, continue_feas=None):
        super(DealOutliers, self).__init__()
        self.IDcol = IDcol
        self.target = target
        self.continue_feas = continue_feas
    def _transform(self, dataset):
        start_time = time.time()
        IDcol = self.IDcol
        target = self.target
        continue_feas = self.continue_feas
        if not continue_feas:
            continue_feas = [col for col in dataset.columns if col not in [IDcol, target]]
        for col in continue_feas:
            quantiles = dataset.approxQuantile(col, [0.25, 0.75], 0.05)
            IQR = quantiles[1] - quantiles[0]
            bounds = [quantiles[0] - 1.5*IQR, quantiles[1] + 1.5*IQR]
            dataset.withColumn(col, fn.when(dataset[col] > bounds[1], bounds[1]).when(dataset[col] < bounds[0], bounds[0]).otherwise(dataset[col]))
        dataset.write.mode("overwrite").saveAsTable('ft_tmp.{}_deal_outliers'.format(project_name))
        end_time = time.time()
        print('time cost for DealOutliers transform process is:', end_time-start_time)
        return dataset











