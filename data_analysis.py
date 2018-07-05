# -*- coding: utf-8 -*-  
from pyspark.sql import SparkSession
import pyspark.sql.functions as fn
from pyspark.sql.functions import udf
from pyspark.sql.types import  *
from datetime import datetime
from pyspark.sql.functions import lit,unix_timestamp,datediff,to_date,col,create_map
from itertools import chain
from spark_constant import YBR_LABEL_MAPPING,YH_NAME_MAPPING
import pandas as pd
from datetime import datetime

"""
@author: Lael
@date: 2018-07-05
@desc: This script is used to analysis the original data from specified hive tables,
       it outputs the null rate, feature distribution in order to select the features.
"""
spark = SparkSession.builder.appName("dataprocess").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel('WARN')



def read_sql(sql,verbose=False):
    '''
    create dataframe from hive sql
    :param sql:sql statement
    :param verbose:if true,print some basic information about df
    :return:dataframe
    '''
    df = spark.sql(sql)
    if verbose:
        print("----------------------24----------------------")
        print("df count:{0}".format(df.count()))
        print("----------------------26----------------------")
        print("df head show:")
        df.show(5)
        df.printSchema()
    return df

def dataeda(df,key,tbl,cols=None,resultpath=None):
    '''
    some eda abount df,including count,key distinct count,null rate
    :param df:dataframe
    :param key:key col in df
    :param cols:selected cols to analyse
    :param resultpath:
    :return:
    '''
    print("###############################################-Processing %s################################################"%tbl)
    print("df count:{0}".format(df.count()))
    print("%s distinct count: %d"%(key, df.select(key).distinct().count()))
    #df.agg(fn.countDistinct(key).alias(key+'_distinct')).show()
    if not cols:
        cols = df.columns
    print("col null rate:")
    nullrate = df.agg(*[(1-(fn.count(c)/fn.count('*'))).alias(c) for c in cols])
    nullrate.cache()
    df.cache()
    cols = df.columns
    nullrate.show()
    '''
    for c in cols:
        tmp = df.toPandas()[c]
        print(type(tmp))
        print(tmp)
        print(c + "------->" +str(tmp))
    '''
    res_tbl_name = "ft_tmp.yhj_"+tbl+"_0426"
    
    nullrate.write.mode("overwrite").saveAsTable(res_tbl_name)
    
    '''
    try:
        nullrate.write.mode("append").saveAsTable("ft_tmp.yhj_nullrate_all6_0426")
    except:
        nullrate.write.mode("overwrite").saveAsTable("ft_tmp.yhj_nullrate_all6_0426")
    '''
    #nullrate.write.mode("overwrite").csv(resultpath,header=True)


def datajoin(primary_df,df_list=None,on=None,how='inner'):
    '''
    join multiple dfs based on primary_df
    :param primary_df:primary df to join
    :param df_list:a list,include other dfs
    :param on: field name to join on,must be found in all dfs
    :param how:way to join,default inner
    :return: joined df
    '''
    for df in df_list:
        primary_df = primary_df.join(df,on,how)
    return primary_df

def datatypecast(df,feature_list=None,dtype=FloatType()):
    '''
    change data in feature_list to dtype
    :param df:
    :param feature_list:features to cast type
    :param dtype:default float, dtype change to
    :return:
    '''
    if not feature_list:
        feature_list = df.columns
    for feature in feature_list:
        df = df.withColumn(feature,df[feature].cast(dtype))
    return df

def datareplace(df,to_replace,value,feature_list=None):
    '''
    replace a value with another value,not support replace to none in spark 2.1.1
    :param df:
    :param to_replace: single or list
    :param value:
    :param feature_list:subset of all features
    :return:
    '''
    df = df.replace(to_replace,value,feature_list)
    return df

def datareplacena(df,to_replace=None,feature_list=None):
    '''
    replace values in to_replace to None in spark 2.1.1 using sql.functions.when
    :param df:
    :param to_replace:
    :param feature_list:
    :return:
    '''
    if not feature_list:
        feature_list = df.columns
    if not to_replace:
        to_replace = ['N','\\N','-1',-1,'9999',9999,'-9999',-9999]
    for feature in feature_list:
        df = df.withColumn(feature,fn.when(~fn.col(feature).isin(to_replace),fn.col(feature)))
    return df

def datadropcol(df,drop_features = None):
    '''
    drop columns in drop_features
    :param df:
    :param drop_features:list,features to drop
    :return:
    '''
    for feature in drop_features:
        df = df.drop(feature)
    return df

def datatimepro(df,feature_time=None,todate=None):
    '''
    time process,calculate the date diff
    :param df:
    :param feature_time:features need to caculate date diff
    :param todate:default datetime now, from which day to calculate
    :return:
    '''
    if not todate:
        todate = datetime.now().strftime("%Y-%m-%d")
    if not feature_time:
        feature_time = df.columns
    for feature in feature_time:
        df = df.withColumn(feature,datediff(to_date(lit(todate)),to_date(unix_timestamp(feature,'yyyy-MM-dd').cast("timestamp"))))
    return df

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

def datadaystotime(df,feature_days=None,yhdate=None,apply_date = "apply_date"):
    '''
    recover some days features in yinhe to its origial time
    :param df:
    :param feature_days:
    :param yhdate:date from yinhe dt
    :return:
    '''
    if not feature_days:
        feature_days = df.columns
    for feature in feature_days:
        df = df.withColumn(feature+"apply_till_now",datediff(to_date(lit(yhdate)),to_date(df[apply_date])))
        df = df.withColumn(feature,df[feature] - df[feature+"apply_till_now"])
        df = df.drop(feature+"apply_till_now")
        df = df.withColumn(feature, fn.when(df[feature] < 0, 0).otherwise(df[feature]))
    return df

def datanullrate(df,threshold=0.5,feature_list=None):
    '''
    get columns that null rate bigger than threshold
    :param df:
    :param threshold:
    :param feature_list:
    :return: column list
    '''
    if not feature_list:
        feature_list = df.columns
    df_miss = df.agg(*[(1-fn.count(c)/fn.count('*')).alias(c) for c in feature_list])
    nullrate = df_miss.toPandas()
    nullrate = nullrate.iloc[0]
    nullrate_list = list(nullrate[nullrate>threshold].index)
    return nullrate_list

def datamap(df,label_features=None,mapping=YBR_LABEL_MAPPING):
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

def datarename(df,rename_map = YH_NAME_MAPPING):
    '''
    rename df cols name based on mapping dict
    :param df:
    :param rename_map:
    :return:
    '''
    colsname = df.columns
    for name in colsname:
        if name in rename_map:
            df = df.withColumnRenamed(name,YH_NAME_MAPPING[name])
    return df

def datadiscretize(df,dis_features=None,*args):
    '''
    discretize a col based on args,eg:value < 60,1,  60<value<70 ,2 ...
    :param df:
    :param args:
    :return:
    '''
    if not dis_features:
        dis_features = df.columns
    for feature in dis_features:
        n = 1
        for v in args:
            if n == 1:
                df = df.withColumn(feature,fn.when(df.feature < v,1))
            else:
                pass
    return df

def datadiscretize(df,dis_features=None):
    '''
    discretize xbscore
    :param df:
    :param dis_features:
    :return:
    '''
    if not dis_features:
        dis_features = df.columns
    for feature in dis_features:
        df = df.withColumn(feature,fn.when(df[feature] < 60,1).otherwise(df[feature]))
        df = df.withColumn(feature,fn.when((df[feature] >= 60)&(df[feature] <=69),2).otherwise(df[feature]))
        df = df.withColumn(feature,fn.when((df[feature] >= 70)&(df[feature] < 79),3).otherwise(df[feature]))
        df = df.withColumn(feature, fn.when((df[feature] >= 80) & (df[feature] < 89), 4).otherwise(df[feature]))
        df = df.withColumn(feature, fn.when((df[feature] >= 90) & (df[feature] < 99), 5).otherwise(df[feature]))
        df = df.withColumn(feature, fn.when(df[feature] > 100,6).otherwise(df[feature]))
    return df


def datamultiply(df,multiply_features=None,multiplier=30):
    '''
    multiply some columns with constant c
    :param df:
    :param multiply_features:
    :param multiplier:
    :return:
    '''
    if not multiply_features:
        multiply_features = df.columns
    for feature in multiply_features:
        df = df.withColumn(feature,df[feature]*30-15)
    return df

def udf_first_element(v):
    '''
    udf,get first element in a vector like the probability predicted by ml.lr
    :param v:
    :return:
    '''
    return float(v[0])


def udf_test(x):
    '''
    udf test pay attention the return type
    :param x:
    :return:
    '''
    if x == 'M':
        return None
    #return float(x*x)


if __name__ == '__main__':
    tbls = ["dmt.dmt_tags_yhj_cnhk_pred_model_a_d","dmt.dmt_tags_yhj_cnhk_pred_model_1_a_d",
            "dmt.dmt_tags_yhj_cnhk_pred_model_02_a_d","dmt.dmt_tags_yhj_cnhk_pred_model_03_a_d",
            "dmt.dmt_tags_yhj_cnhk_pred_model_04_a_d","dmt.dmt_tags_yhj_cnhk_pred_model_05_a_d"]
    for tbl in tbls:     
        curr_time = datetime.now()
        print("\n\nProcessing table %s------> at %s"%(tbl,curr_time))
        sql = "select * from dmt."+tbl
        df = read_sql(sql,verbose=True)
        dataeda(df,'user_id',tbl,resultpath='/exportfs/home/cuitingting/yhj_dl')
        null_list = datanullrate(df,threshold=0.3)
        print(null_list)
        print("\n------------------------------319------------------------------")
        