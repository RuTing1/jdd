# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as fn
from datetime import datetime, date, timedelta
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import lit,unix_timestamp,datediff,to_date,col,create_map

"""
@author: DR
@date: 2018-08-13
@desc: This script is used to get data for the project of wakeing up sleeping customers
@question: 
@v1版本说明：
@example:spark-submit dataProcess.py --executor-memory 6g --executor-num 20  --master yarn
@测试数据存储：
ft_tmp.yhj_sleepuser_v0:最原始可用的特征数据,异常缺失值已经处理
ft_tmp.yhj_sleepuser_v1：针对v0版数据，对数据进行处理，全部变成数值型数据
ft_tmp.yhj_sleepuser_v2:针对v1版数据，对数据进行处理，剔除高缺失率低方差（<0.1）数据
ft_tmp.yhj_sleepuser_v3:针对v2版数据，对数据进行处理，剔除共线性较高（>0.8）的数据
ft_tmp.yhj_sleepuser_v4:针对v3版数据，对数据进行处理，对IQR异常值进行平滑处理
"""

reload(sys)
sys.setdefaultencoding('utf-8')
print('active SparkSession ...')
spark = SparkSession.builder.appName("dataprocess").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel('WARN')
yesterday = (date.today() + timedelta(days = -1)).strftime("%Y-%m-%d")    # 昨天日期
IDcol = 'jdpin'
target = 'is_buy'

#获取数据
def get_data():
    #准备数据源,[表名，dt，字段（第一个字段为主key）]
    target_df = spark.sql("SELECT all_users.jdpin jdpin") #取得目标用户
    tbls_feas = [] 
    for i in range(0,len(tbls)):
    #以存量用户和潜在用户作为左连接，将各表连接起来
    df_comb_all = target_df
    for df in tbls_feas:
        df_comb_all = df_comb_all.join(df,'jdpin','left')
    return df_comb_all

#剔除重复数据
def drop_duplicate(df):
    """
    Test the dataframe to drop duplicate rows.
    @param df: dataframe to drop duplicate rows
    @return:    
    """
    cnt_df = df.count()
    cnt_d_df = df.distinct().count()
    print('Count of rows: {0}'.format(cnt_df))
    print('Count of distinct rows: {0}'.format(cnt_d_df))
    if cnt_df==cnt_d_df:
        return df
    else:
        return df.dropDuplicates()

#异常缺失值处理
def datareplacena(df,to_replace=None,feature_list=None):
    '''
    replace values in to_replace to None in spark 2.1.1 using sql.functions.when
    @param df:
    @param to_replace:
    @param feature_list:
    return:
    '''
    if not feature_list:
        feature_list = df.columns
    if not to_replace:
        to_replace = ['N','\\N','-1',-1,'9999',9999,'-9999',-9999]
    for feature in feature_list:
        df = df.withColumn(feature,fn.when(~fn.col(feature).isin(to_replace),fn.col(feature)))
    return df

#时间类数据处理
def datatimepro(df,feature_time=None,apply_date="apply_date"):
    '''
    caculate date diff,using apply date
    :param df:
    :param feature_time:
    :param apply_date:
    :return:
    '''
    if not feature_time:
        feature_time = df.columns
    for feature in feature_time:
        df = df.withColumn(feature,datediff(to_date(df[apply_date]),to_date(df[feature])))
        df = df.withColumn(feature,fn.when(df[feature]<0,0).otherwise(df[feature]))
    return df

#数据映射
def datamap(df,mapping,label_features=None):
    '''
    realize map function (pandas) in Spark
    :param df:
    :param label_features:
    :param mapping:
    :return:
    '''
    if not label_features:
        label_features = df.columns
    mapping_expr = create_map([lit(x) for x in chain(*mapping.items())])
    for feature in label_features:
        df = df.withColumn(feature,mapping_expr.getItem(col(feature)))
    return df

#数据类型转化
def datatypecast(df,feature_list=None,dtype=FloatType()):
    '''
    change data in feature_list to dtype
    :param df:
    :param feature_list:features to cast type
    :param dtype:default float, dtype change to
    :return:
    '''
    if not feature_list:
        feature_list = [col for col in df.columns if col not in [IDcol,'is_buy']]
    for feature in feature_list:
        df = df.withColumn(feature,df[feature].cast(dtype))
    return df

#剔除高缺失率低方差变量
def low_mis_high_std_cols(df,feature_list=None,missing_ratio=0.5,std_threshold=1):
    """
    get columns with low missing rate and not low std
    @param:df 需要统计的数据
    @feature_list:建议 [col for col in df.columns if col not in [IDcol,TARGETcol]]
    @param:missing_ratio 缺失值删除的占比阈值
    @param:std_threshold 方差过小的删除阈值
    return: info_cols 剔除缺失值过多和方差过小变量后变量组
    """
    info_cols = []
    df = df.drop(IDcol).drop(target)
    cnt = df.count()
    if not feature_list:
        feature_list = df.columns
    df_desb = datatypecast(df.describe())
    #df_mis = df.agg(*[(1-(fn.count(c)/fn.count('*'))).alias(c+'_missing') for c in feature_list])
    for col in feature_list:
        if df_desb.select(col).collect()[0][0] > missing_ratio*cnt: #获得缺失率不超过阈值的特征
            #normlizer = Normalizer()  #改进方案：先归一化再求方差
            if df_desb.select(col).collect()[2][0] > std_threshold: #获得方差超过阈值的特征
                info_cols.append(col)
    return info_cols

#剔除共线性强的特征
def delete_colline(df,colline_threshold=0.8):
    """
    drop columns with high colline
    @param:df 原始数据
    @param:IDcol 数据唯一标识符
    @param:target 目标变量
    @param:threshold 变量相关性阈值
    return:共线性较低的变量名称
    """
    drop_cols,corr = [],[]
    cor_cols = [col for col in df.columns if col not in [IDcol,target]]
    #建立一个相关性矩阵
    len_cor = len(cor_cols)
    for i in range(0,len_cor):
        if cor_cols[i] in drop_cols:
            continue
        for j in range(i+1,len(cor_cols)):
            if cor_cols[j] in drop_cols:
                continue
            corr_value = abs(df.corr(cor_cols[i],cor_cols[j],'pearson'))
            if corr_value > colline_threshold:
                corr_i = abs(df.corr(cor_cols[i],target,'pearson'))
                corr_j = abs(df.corr(cor_cols[j],target,'pearson'))
                if corr_i>= corr_j:
                    drop_cols.append(cor_cols[j])
                else:
                    drop_cols.append(cor_cols[i])
    return [col for col in df.columns if col not in drop_cols]

#获取分类变量和连续变量
def data_class(df,threshold=100):
    """
    """
    continue_cols, class_cols = [],[]
    for col in [col for col in df.columns if col not in [IDcol]]:
        v_cnt = df.groupBy(col).count().withColumnRenamed('count','count').count
        if v_cnt > threshold:
            continue_cols.append(col)
        else:
            class_cols.append(col)
    return continue_cols, class_cols 

#异常值处理
def deal_outliers(df, cols):
    """
    deal continue features with outlier data
    @param:df 原始数据
    @param:col 连续变量
    return:
    """
    for col in cols:
        quantiles = df.approxQuantile(col, [0.25, 0.75], 0.05)
        IQR = quantiles[1] - quantiles[0]
        bounds = [quantiles[0] - 1.5*IQR, quantiles[1] + 1.5*IQR]
        df.withColumn(col, fn.when(df[col] > bounds[1], bounds[1]).when(df[col] < bounds[0], bounds[0]).otherwise(df[col]))
    return df

#特征聚合
def comb_feas(df,infocols):
    """
    Combine features to fit some ml algorithms.
    @param df: dataframe (剔除infocols之后的df的变量要求全部为数值型变量，且不能有空值)
    @param infocols: list of columns which do not need to combine as features,such as IDcol and target columns 
    @return: df with combined features and infocols   
    """
    from pyspark.ml.feature import VectorAssembler
    assm_feas = [col for col in df.columns if col not in infocols]
    vecAssembler = VectorAssembler(inputCols=assm_feas, outputCol="features")
    df = vecAssembler.transform(df)
    infocols.append('features')
    df = df.select(infocols)
    return df

#特征拆分
def extract(row):
    return (row[IDcol],)+(row[targetCol],) + tuple(row.features.toArray().tolist())

def split_feas(df,colnames):
    """
    Split features to which combined in comb_feas().
    @param df: dataframe
    @param colnames: list of columns which to name for new splited columns 
    @return: df with splited features 
    """
    df = df.rdd.map(extract).toDF()
    for col1,col2 in zip(df.columns,colnames):
        df = df.withColumnRenamed(col1,col2)
    return df


if __name__ == '__main__':  
    """
    数据与处理前，需要区分以下数据类型：
    时间类数据：dateType = []
    分类变量：classType = [] （某种程度上需要区分益博睿和非益博睿类）
    """  
    print('loading data ...')
    df = get_data()
    df = drop_duplicate(df)
    print('the num of data is:',df.count())
    print('the num of the features is:',len(df.columns))
    print('preprocessing missing features ...')
    df = datareplacena(df)
    print('saving data with missing features done ...')
    df.write.mode('overwrite').saveAsTable('ft_tmp.')
    print('preprocessing different datatype features ...')
    print('preprocessing dateType features ...')
    dateType = []
    df = df.withColumn('apply_date',fn.lit(yesterday))
    df = datatimepro(df,feature_time=dateType,apply_date='apply_date')
    df = df.drop('apply_date')
    print('preprocessing classType features ...')
    k = list('kabcdefghijlmnopqrstuvwxyz')
    v = range(26)
    YBR_LABEL_MAPPING = dict(zip(k,v))
    #益博睿类标签
    classType = []
    df = datamap(df,YBR_LABEL_MAPPING,classType)
    print('preprocessing strType features ...')
    #特征类型
    continue_cols, class_cols = data_class(df)
    #字符串类特征的处理
    strClassType = [col for col in class_cols if col not in classType]
    # stringindexer
    for col in strClassType:
        stringIndexer = StringIndexer(inputCol=col, outputCol="indexed")
        model = stringIndexer.fit(df)
        df = model.transform(df)
        df = df.withColumnRenamed('indexed',col)
    # one hot
    feas_type = {}
    df_type = df.dtypes #各特征类型
    for i in range(0,len(df.columns)):
        feas_type[df_type[i][0]] = df_type[i][1]
    strType = {k:v for k,v in feas_type.items() if v=='string' and k not in [IDcol]}

    df = datatypecast(df,feature_list=strType.keys(),dtype=IntegerType())
    print('saving data which features with correct format ...')
    df.write.mode("overwrite").saveAsTable('ft_tmp.yhj_sleepuser_v1')
    print('filtering low value features ...')
    info_cols_1 = low_mis_high_std_cols(df,missing_ratio=0.5,std_threshold=0.1)
    info_cols_1.append(IDcol)
    info_cols_1.append(target)
    df1 = df.select(info_cols_1)
    print('saving data which features with lower missing ratio and higher sttdev...')
    df1.write.mode("overwrite").saveAsTable('ft_tmp.yhj_sleepuser_v2')
    print('filtering colline features ...')
    info_cols_2 = delete_colline(df1)
    df2 = df1.select(info_cols_2)
    print('saving data which features with lower colline value ...')
    df2.write.mode('overwrite').saveAsTable('ft_tmp.yhj_sleepuser_v3')
    continue_cols, class_cols = data_class(df2)
    print('deal and save outlier values with smoothing method ...')
    df3 = deal_outliers(df, continue_cols)
    df3.write.mode('overwrite').saveAsTable('ft_tmp.yhj_sleepuser_v4')





    

    

