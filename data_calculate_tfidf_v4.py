# -*- coding: utf-8 -*-  
import sys
import datetime
import numpy as np
import pandas as pd
from itertools import chain
from pyspark.ml.feature import Bucketizer
import pyspark.sql.functions as fn
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType,IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorSlicer
from pyspark.sql.functions import lit,unix_timestamp,datediff,to_date,col,create_map
"""
@author: DR
@date: 2018-07-19
@desc: This script is used to filter the informational features from specified hive tables,
@question: 特征的值需要是单调递增，且越大的值越重要
@v2版本说明：实验算法跑通的能力,解决特征划分区域问题
@example:spark-submit yhj_data_analysis_laxin_tfidf_v1.py --executor-memory 6g --executor-num 20  --master yarn
@测试数据存储：
        df_comb_all.write.mode("overwrite").saveAsTable('ft_tmp.yhj_lx_tfidf_df_comb_all_v2')
        df_comb.write.mode("overwrite").saveAsTable('ft_tmp.yhj_lx_tfidf_df_comb') ft_tmp.yhj_laxin_df_test
"""

IDcol = 'jdpin'
####函数库#######
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


def tf_idf(df):
    """
    calculate features tf-idf value
    @公式：
        tf = w(fi)/sum(w(f))
        idf = df[fi].count()/df.count()*n
        tf-idf = tf*idf
    @example
    df = spark.createDataFrame([
          (0,0.1,5,5,100,0,'a'),
          (1,0.5,6,6,173,1,'b'),
          (2,0.7,5,None,150,0,'c'),
          (3,None,None,7,186,1,'d'),
          (4,0.65,3,3,None,0,'e'),
          (5,0.43,None,2,None,1,'f')
     ], ["id1", "id2","id3","id4","id5","id6","id7"])
    df_tfidf = tf_idf(df,'id1')
    """
    print("\nProcessing table------> at %s"%(datetime.datetime.now()))
    cnt = df.count()
    # df = df.na.fill(0)
    df = df.withColumn('sum_feas',sum(df.select([col for col in df.columns if col not in [IDcol]])))
    fea_cols = [col for col in df.columns if col not in [IDcol,'sum_feas']]
    for col in fea_cols:
        print('########################## the feature which is processing is:',col)
        v_cnt = df.groupBy(col).count().withColumnRenamed('count','count')
        section_num = v_cnt.count() # 区间数量 
        v_idf = v_cnt.withColumn('idf',fn.round(v_cnt['count']/cnt,3))  #计算每个标签的idf值
        df = df.withColumn('tf',fn.round(df[col]/df['sum_feas'],3)) #计算每个用户的tf值
        df = df.join(v_idf,[col],'left')   #这个有空优化一下，左表越来越大，join起来费时
        df = df.withColumn(col,fn.round(df['tf']*df['idf']*section_num,3))#计算每个用户的tf-idf值
        df = df.drop('idf').drop('tf').drop('count')
    print("\nProcessing table------> at %s"%(datetime.datetime.now()))
    return df


def feas_type(df,feas,threshold=30):
    feas_type = {'conu':[],'clas':[]}
    for col in feas:
        df = df.withColumn(col,df[col].cast(IntegerType()))
        value_cnt = df.groupBy([col]).count().count()
        if value_cnt>30:
            feas_type['conu'].append(col)
        else:
            feas_type['clas'].append(col)
    return feas_type


def col_rename(df,target_name):
    i = 0
    for col in df.columns:
        df = df.withColumnRenamed(col,target_name[i])
    return df


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


if __name__ == "__main__":
    #step1:环境设置
spark = SparkSession.builder.appName("dataprocess").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel('WARN')
k = list('kabcdefghijlmnopqrstuvwxyz')
v = range(26)
YBR_LABEL_MAPPING = dict(zip(k,v))
#step2:数据预处理
CURR_TIME = datetime.datetime.now()
delta = datetime.timedelta(days=1)
YESD_DATE = (CURR_TIME - delta).strftime("%Y-%m-%d")
##step2.1:表及特征准备
target_df = spark.sql("SELECT all_users.jdpin jdpin,CASE WHEN trade_users.jdpin is NULL THEN 0 ELSE 1 END is_buy FROM \
(SELECT jdpin FROM dwd.dwd_app_jr_fp_user_i_d WHERE valid_sign=1 GROUP BY jdpin)all_users LEFT JOIN \
(SELECT jdpin FROM dwd.dwd_app_jr_fp_trade_history_i_d WHERE trade_type=0 AND valid_sign='1' AND trade_state='00' GROUP BY jdpin)trade_users \
ON all_users.jdpin = trade_users.jdpin")  #银行+用户（130W，包含存量13W）
targert_table = "(SELECT jdpin FROM ft_app.ftapp_yhj_user_lifestyle_s_d where dt='"+ YESD_DATE + "' GROUP BY jdpin)a"
tbls ={'tb0':["ft_app.ftapp_ybr_a_s_m",'dt',['jdpin','mob_bt','danbizuida','dizhiwendingxing','dingdanshu','shifujine']],
       'tb1':["ft_app.ftapp_zr_s_m",'dt',['user_log_acct','amt_total_offer']],
       'tb2':["dmt.dmt_tags_yhj_cnhk_pred_model_a_d",'dt',['user_id','pay_syt_f0014']],
       'tb3':["dmt.dmt_tags_yhj_cnhk_pred_model_1_a_d",'dt',['user_id','jdmall_up_m0016','jdmall_up_m0001']],
       'tb4':["dmt.dmt_tags_yhj_cnhk_pred_model_03_a_d",'dt',['user_id','jdmall_user_p0035','jdmall_jdmuser_p0002816','jdmall_user_p0033']],
       'tb5':["dmt.dmt_tags_yhj_cnhk_pred_model_05_a_d",'dt',['user_id','mem_mem_f0005266']],
       }  #特征来源表
##step2.2:获取表及各特征更新时间  
tbls_feas = [] 
for i in range(0,len(tbls)):
    tbl_index = 'tb' + str(i)
    tbls[tbl_index].append(str(spark.sql("SELECT MAX("+tbls[tbl_index][1]+") FROM "+ tbls[tbl_index][0]).collect()[0][0]))
    tbls_feas.append(spark.sql("SELECT a.jdpin,"+ ','.join(tbls[tbl_index][2][1:]) + " FROM " + targert_table +" LEFT JOIN "+ \
              tbls[tbl_index][0] + " b ON a.jdpin = b." +tbls[tbl_index][2][0] + " AND b."+tbls[tbl_index][1]+"='"+tbls[tbl_index][3]+"'"))
    ##step2.3:特征数据聚合
df_comb_all = target_df
for df in tbls_feas:
    df_comb_all = df_comb_all.join(df,'jdpin','left')
    ##step2.4:连续变量离散化处理
    fea_d_cols = {
                  'amt_total_offer':[-float("inf"), 0, 10, 20, 30, 40, 50, 100, 200, 500, float("inf")]
                   
                 } # 用于存储连续变量及其分割点
    for col_buckt in fea_d_cols:
        bucketizer = Bucketizer(splits=fea_d_cols[col_buckt], inputCol=col_buckt, outputCol=col_buckt+'b')
        df_comb_all = bucketizer.transform(df_comb_all)
        ##step2.5:分类变量映射
        # fea_c_cols = {
        #               'mob_bt':{'k':0,'a':1,'b':2,'c':3,'d':4,'e':5,'f':6},
        #               'complete_amt1':{'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6},
        #               'sale_ord':{'k':1,'a':2,'b':3,'c':4,'d':5,'e':6,'f':7},
        #               'dizhiwendingxing':{'a':1,'b':2,'c':3,'d':4,'e':5,'f':6}
        #              } # 用于存储连续变量及其分割点
    fea_c_cols = ['mob_bt','danbizuida','dizhiwendingxing','dingdanshu','shifujine']
    df_comb_all = datamap(df_comb_all,YBR_LABEL_MAPPING,fea_c_cols)
    df_comb_all = datareplacena(df_comb_all)#异常缺失值替换 
    # df_comb_all.groupBy().max().show() 
    df_comb_all = spark.sql("SELECT * FROM ft_tmp.yhj_lx_tfidf_df_comb_all")
    fea_cols = [col for col in df_comb_all.columns if col not in [IDcol,'is_buy','amt_total_offer']]
    df_comb_all = datatypecast(df_comb_all,fea_cols,IntegerType())
    df_comb_all = df_comb_all.na.fill(0)  #缺失值处理
    df_comb_all = df_comb_all.select(fea_cols.append('is_buy'))
    #df_comb_all.write.mode("overwrite").saveAsTable('ft_tmp.yhj_lx_tfidf_df_comb_all_v2')

    df_comb = df_comb_all.filter(df_comb_all.is_buy == 1).drop('is_buy')
    df_tfidf = tf_idf(df_comb)
    df_weight = df_tfidf.groupBy().sum()
    df_weight = df_weight.drop('sum(sum_feas)')
    df_weight = df_weight.withColumn('sum_feas',sum(df_weight.select([col for col in df_weight.columns if col not in [IDcol]]))).rdd.toDF()#计算各标签的权重值
    # feas_w = [col for col in df_weight.columns if col not in ['sum_feas']]
    df_weight_row = {}
    for col in df_weight.columns:
        df_weight = df_weight.withColumn(col,fn.round(df_weight[col]/df_weight['sum_feas'],3))
        df_weight = df_weight.withColumnRenamed(col,col[4:len(col)-1])
    df_weight_row = df_weight.collect()[0]
    print('################## the weight for each feature is as follow:',df_weight_row,'#####################')
    for col in [col for col in df_comb.columns if col not in [IDcol,'is_buy']]:
        df_tfidf = df_tfidf.withColumnRenamed(col,col+'f')
    df_final = df_comb.join(df_tfidf,'jdpin','inner')   # 13196552911_p 用户有空值
    fea_col_f = []
    for col in fea_cols:
        print(col)
        fea_col_f.append(col+'f')
        print(fea_col_f)
        df_final_sum = df_final.groupBy(col).sum(col+'f').rdd.toDF()
        col_sum = df_final_sum.agg({'sum('+col+'f)':'sum'}).collect()[0][0]
        df_final_sum = df_final_sum.withColumn(col+'region_w',fn.round(df_final_sum['sum('+col+'f)']/col_sum,3))# 归一化每个特征每个区间的权重
        print('################## the weight for each feature in each area is as follow:#####################')
        # df_final_sum = df_final_sum.drop(col)
        df_final_sum.show()
        df_comb_all = df_comb_all.join(df_final_sum,col,'left')
        df_comb_all = df_comb_all.na.fill(0) #对于存量用户没的值非存量用户有的值填充为0
        df_comb_all = df_comb_all.withColumn(col+'f',df_comb_all[col]*df_comb_all[col+'region_w']*df_weight_row[col])
    fea_cols.append('is_buy')
    df_comb_all = df_comb_all.withColumn('tfidf_weight',sum(df_comb_all.select([col for col in fea_col_f])))
    fea_cols.extend(['jdpin','tfidf_weight'])
    print(fea_cols)
    df_comb_all = df_comb_all.select(fea_cols)
    df_comb_all.write.mode("overwrite").saveAsTable('ft_tmp.yhj_lx_weight_tfidf_v2')








#结果分析
SELECT score, COUNT(is_buy) AS total, SUM(is_buy) AS buy_cnt
FROM 
(
SELECT CASE WHEN tfidf_miss_weight <0.1 AND tfidf_miss_weight>=0 THEN '[0,0.1)'
            WHEN tfidf_miss_weight <0.2 AND tfidf_miss_weight>=0.1 THEN '[0.1,0.2)'
            WHEN tfidf_miss_weight <0.3 AND tfidf_miss_weight>=0.2 THEN '[0.2,0.3)'
            WHEN tfidf_miss_weight <0.4 AND tfidf_miss_weight>=0.3 THEN '[0.3,0.4)'
            WHEN tfidf_miss_weight <0.5 AND tfidf_miss_weight>=0.4 THEN '[0.4,0.5)'
            WHEN tfidf_miss_weight <0.6 AND tfidf_miss_weight>=0.5 THEN '[0.5,0.6)'
            WHEN tfidf_miss_weight <0.7 AND tfidf_miss_weight>=0.6 THEN '[0.6,0.7)'
            WHEN tfidf_miss_weight <0.8 AND tfidf_miss_weight>=0.7 THEN '[0.7,0.8)'
            WHEN tfidf_miss_weight <0.9 AND tfidf_miss_weight>=0.8 THEN '[0.8,0.9)'
            WHEN tfidf_miss_weight <=1.0 AND tfidf_miss_weight>=0.9 THEN '[0.9,1.0]'
      ELSE NULL END score,
      is_buy
FROM
(
SELECT (tfidf_miss_weight-0.463)/(3.560-0.539) tfidf_miss_weight ,is_buy FROM 
(SELECT round(tfidf_miss_weight,3) tfidf_miss_weight ,is_buy FROM ft_tmp.yhj_lx_weight_tfidf_miss_0725_14)a
)b
)c
GROUP BY score









