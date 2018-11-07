#bank_desire验证:银行plus
#20181015:银行+总用户量：547485
#20180914-20181015新增用户:85565
#20180914-20181015购买用户:220472

#step1:数据准备
SET hive.support.quoted.identifiers=None;
USE ft_tmp;
CREATE TABLE ft_tmp.yhj_lx_promotion AS
WITH buy_users AS (
    SELECT id
          ,jdpin
          ,channel_id
          ,project_id
          ,project_name
          ,MIN(trade_amount) trade_amount
          ,to_date(trade_date) trade_date
        FROM dwd.dwd_app_jr_fp_trade_history_i_d 
        WHERE trade_type=0 AND valid_sign='1' AND trade_state='00'
        GROUP BY
             id
            ,jdpin
            ,channel_id
            ,project_id
            ,project_name
            ,to_date(trade_date) 
    ),--trade_his表去重

new_users AS ( 
    SELECT a.jdpin user_id
      FROM
        (SELECT jdpin FROM buy_users GROUP BY jdpin) a 
        LEFT JOIN 
        (SELECT jdpin FROM buy_users WHERE trade_date <'2018-09-14' GROUP BY jdpin) old_buy_users
        ON a.jdpin=old_buy_users.jdpin
        WHERE old_buy_users.jdpin IS NULL
),

user_score AS (
SELECT `(user_id)?+.+` 
    FROM 
    (
    SELECT jdpin
          ,bank_desire
          ,CASE WHEN bank_desire <=10 THEN '(,10]'
                 WHEN bank_desire <=20 THEN '(10,20]'
                 WHEN bank_desire <=30 THEN '(20,30]'
                 WHEN bank_desire <=40 THEN '(30,40]'
                 WHEN bank_desire <=50 THEN '(40,50]'
                 WHEN bank_desire <=60 THEN '(50,60]'
                 WHEN bank_desire <=70 THEN '(60,70]'
                 WHEN bank_desire <=80 THEN '(70,80]'
                 WHEN bank_desire <=90 THEN '(80,90]'
                 WHEN bank_desire <=100 THEN '(90,100]'
            ELSE NULL END score 
        FROM ft_tmp.bank_sleep_customers_probability
        WHERE dt='20180914'
    )a 
    LEFT JOIN 
    (
    SELECT `(dt)?+.+` 
        FROM useful_features_from_bz
        WHERE dt='20180921'
    )b ON a.jdpin=b.user_id
) 

SELECT CASE WHEN user_id IS NOT NULL THEN 1 ELSE 0 END is_new
       ,`(user_id)?+.+` 
    FROM user_score a LEFT JOIN new_users b ON a.jdpin=b.user_id
;

#数据分布
分值    预测购买用户    实际购买新用户    占比
NULL                    16909    
(,10]               214111975    7120    0.003%
(10,20]                58224905    5713    0.010%
(20,30]                27765444    5087    0.018%
(30,40]                17402727    4930    0.028%
(40,50]                11472169    4625    0.040%
(50,60]                 7255407    4071    0.056%
(60,70]                 6240539    5286    0.085%
(70,80]                 4102043    5155    0.126%
(80,90]                 3563443    7075    0.199%
(90,100]             3037961    19594    0.645%

#与用户购买与否最相关top10特征

#高分预测购买且真实购买用户特征与低分不购买用户差异
#低分预测不够买且真实购买用户特征与低分不购买用户差异

#银行理财加入金融标签对各分数区间人数的影响
#1.登录金融app距今时长
SELECT score,time_scope,COUNT(*),SUM(label)
    FROM 
    (
    SELECT jdpin
          ,label
          ,CASE  WHEN probability1 <=0.10 THEN '(,10]'
                 WHEN probability1 <=0.20 THEN '(10,20]'
                 WHEN probability1 <=0.30 THEN '(20,30]'
                 WHEN probability1 <=0.40 THEN '(30,40]'
                 WHEN probability1 <=0.50 THEN '(40,50]'
                 WHEN probability1 <=0.60 THEN '(50,60]'
                 WHEN probability1 <=0.70 THEN '(60,70]'
                 WHEN probability1 <=0.80 THEN '(70,80]'
                 WHEN probability1 <=0.90 THEN '(80,90]'
                 WHEN probability1 <=1.00 THEN '(90,100]'
            ELSE NULL END score 
        FROM ft_tmp.yhj_randomforestclassifier_bank_desire_probability1_all --20181022
    )a 
LEFT JOIN 
    (
    SELECT user_id
          ,fin_dlc_f0045 time_scope
        FROM dmt.dmt_tags_lhyy_fin_icbc_a_d
        WHERE dt='2018-10-16'
    )b ON a.jdpin=b.user_id
GROUP BY score,time_scope


#2.银行plus理财持仓
SELECT score,position,COUNT(*),SUM(label)
    FROM 
    (
    SELECT jdpin
          ,label
          ,CASE  WHEN probability1 <=0.10 THEN '(,10]'
                 WHEN probability1 <=0.20 THEN '(10,20]'
                 WHEN probability1 <=0.30 THEN '(20,30]'
                 WHEN probability1 <=0.40 THEN '(30,40]'
                 WHEN probability1 <=0.50 THEN '(40,50]'
                 WHEN probability1 <=0.60 THEN '(50,60]'
                 WHEN probability1 <=0.70 THEN '(60,70]'
                 WHEN probability1 <=0.80 THEN '(70,80]'
                 WHEN probability1 <=0.90 THEN '(80,90]'
                 WHEN probability1 <=1.00 THEN '(90,100]'
            ELSE NULL END score 
        FROM ft_tmp.yhj_randomforestclassifier_bank_desire_probability1_all --20181022
    )a 
LEFT JOIN 
    (
    SELECT user_id
          ,mem_mem_f0005266 position
        FROM dmt.dmt_tags_lhyy_fin_icbc_a_d
        WHERE dt='2018-10-16'
    )b ON a.jdpin=b.user_id
GROUP BY score,position

#以上两项设置规则
a)	附加金融标签-最近一次登录金融APP距今天的天数，规则建议
①	最近一次登录金融app距今时长(0,7] (fin_dlc_f0045=1)且预测评分10分以下用户评分90分；
②	最近一次登录金融app距今时长(180,]( fin_dlc_f0045>=7)且预测评分80分以上，预测评分设置为10分
③	最近一次登录金融app距今时长(90,180]( fin_dlc_f0045=6)且预测评分80分以上，预测评分设置为20分
b)	附件金融标签-理财持仓(含plus)，规则建议
①	理财持仓20W+( mem_mem_f0005266>=12)，预测评分设置为90分
②	理财持仓1000-( mem_mem_f0005266<=3)，预测评分设置为10分

#规则数据存储
ft_tmp.yhj_randomforestclassifier_bank_desire_probability1_all
USE ft_tmp;
CREATE TABLE ft_tmp.yhj_randomforestclassifier_bank_desire_probability1_r_all AS
SELECT jdpin
      ,label
      ,CASE  WHEN probability1 <=0.10 THEN '(,10]'
             WHEN probability1 <=0.20 THEN '(10,20]'
             WHEN probability1 <=0.30 THEN '(20,30]'
             WHEN probability1 <=0.40 THEN '(30,40]'
             WHEN probability1 <=0.50 THEN '(40,50]'
             WHEN probability1 <=0.60 THEN '(50,60]'
             WHEN probability1 <=0.70 THEN '(60,70]'
             WHEN probability1 <=0.80 THEN '(70,80]'
             WHEN probability1 <=0.90 THEN '(80,90]'
             WHEN probability1 <=1.00 THEN '(90,100]'
        ELSE probability1 END score 
FROM 
(
SELECT jdpin
      ,label
      , CASE WHEN mem_mem_f0005266 >= 12 THEN COALESCE('(90,100]',probability1)
             WHEN mem_mem_f0005266 <= 3 THEN COALESCE('(,10]',probability1)
             WHEN fin_dlc_f0045 = 1  AND probability1 <=0.1 THEN COALESCE('(90,100]',probability1)
             WHEN fin_dlc_f0045 >= 7 AND probability1 >0.90 THEN COALESCE('(,10]',probability1)
             WHEN fin_dlc_f0045 = 6 AND probability1 >0.90 THEN COALESCE('(10,20]',probability1)
        ELSE probability1 END probability1 
    FROM 
    (
    SELECT jdpin
          ,label
          ,probability1
        FROM ft_tmp.yhj_randomforestclassifier_bank_desire_probability1_all --20181022
    )a 
    LEFT JOIN
    (
    SELECT user_id,mem_mem_f0005266,fin_dlc_f0045
        FROM dmt.dmt_tags_lhyy_fin_icbc_a_d
        WHERE dt='2018-10-16'
    )b ON a.jdpin=b.user_id
)a 

CREATE TABLE ft_tmp.yhj_randomforestclassifier_bank_desire_probability1_r_all AS


SELECT score
      ,COUNT(label)
      ,SUM(label)
    FROM 
    (
    SELECT jdpin
          ,label
          ,CASE  WHEN probability1 <=0.10 THEN '(,10]'
                 WHEN probability1 <=0.20 THEN '(10,20]'
                 WHEN probability1 <=0.30 THEN '(20,30]'
                 WHEN probability1 <=0.40 THEN '(30,40]'
                 WHEN probability1 <=0.50 THEN '(40,50]'
                 WHEN probability1 <=0.60 THEN '(50,60]'
                 WHEN probability1 <=0.70 THEN '(60,70]'
                 WHEN probability1 <=0.80 THEN '(70,80]'
                 WHEN probability1 <=0.90 THEN '(80,90]'
                 WHEN probability1 <=1.00 THEN '(90,100]'
            ELSE probability1 END score
    FROM 
    (
    SELECT jdpin
          ,label
          , CASE WHEN mem_mem_f0005266 >=12 THEN (0.8 + probability1/5)
                 WHEN mem_mem_f0005266 >=7  THEN (0.7 + probability1/4)
                 WHEN mem_mem_f0005266 >=4  THEN (0.5 + probability1/2)
                 WHEN mem_mem_f0005266 =0  AND probability1 <0.9 THEN probability1*0.5
                 WHEN mem_mem_f0005266 =0  AND probability1 >=0.9 THEN (0.2 + probability1/2)
                 WHEN mem_mem_f0005266 IN(1,2,3) AND probability1 <0.95 THEN (probability1-(0.6/mem_mem_f0005266))
                 WHEN fin_dlc_f0045 = 1  AND probability1 <=0.2 THEN (0.8 + probability1)
                 WHEN fin_dlc_f0045 >= 7 AND probability1 >0.80 THEN (0.8 - fin_dlc_f0045/20)
            ELSE probability1 END probability1 
        FROM 
        (
        SELECT jdpin
              ,label
              ,probability1
            FROM ft_tmp.yhj_randomforestclassifier_bank_desire_probability1_all --20181022
        )a 
        LEFT JOIN
        (
        SELECT user_id,mem_mem_f0005266,fin_dlc_f0045
            FROM dmt.dmt_tags_lhyy_fin_icbc_a_d
            WHERE dt='2018-10-16'
        )b ON a.jdpin=b.user_id
    )a 
    )a
    GROUP BY score;


SELECT score
      ,COUNT(label)
      ,SUM(label)
    FROM 
    (
    SELECT jdpin
          ,label
          ,CASE  WHEN probability1 <=0.10 THEN '(,10]'
                 WHEN probability1 <=0.20 THEN '(10,20]'
                 WHEN probability1 <=0.30 THEN '(20,30]'
                 WHEN probability1 <=0.40 THEN '(30,40]'
                 WHEN probability1 <=0.50 THEN '(40,50]'
                 WHEN probability1 <=0.60 THEN '(50,60]'
                 WHEN probability1 <=0.70 THEN '(60,70]'
                 WHEN probability1 <=0.80 THEN '(70,80]'
                 WHEN probability1 <=0.90 THEN '(80,90]'
                 WHEN probability1 <=1.00 THEN '(90,100]'
            ELSE probability1 END score 
    FROM ft_tmp.yhj_randomforestclassifier_bank_desire_probability1_all
    )a 
    GROUP BY score






#####数据提取，计算特征与目标特征的相关性
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import time
import datetime
from pyspark.sql import SparkSession


reload(sys)
sys.setdefaultencoding('utf-8')
IDcol='jdpin'
seed=12345
target='is_new'

def colline_features(dataset, IDcol, target, colline_threshold=0.6):
    info_cols = {}
    cor_cols = [col for col in dataset.columns if col not in [IDcol,target]]
    len_cor = len(cor_cols)
    for i in range(0,len_cor):
        corr_value = abs(dataset.corr(cor_cols[i],target,'pearson'))
        if corr_value > colline_threshold:
            info_cols[cor_cols[i]] = corr_value
    return info_cols

def getData(df, IDcol, target, n_samp_ratio=[0.5,0.5]):
    """
    @param sql: string 读取数据的SQL语句
    @param n_samp_ratio: int 负样本采样比例
    return: 训练集和验证集
    """
    # features= [col for col in df.columns if col not in [IDcol, target]]
    #分层采样
    df_sample = df.sampleBy(target, fractions={0: n_samp_ratio[0], 1: n_samp_ratio[1]}, seed=seed)
    a = df_sample.groupBy(target).count()
    b = a.sort(target).collect()
    good = b[1][1]
    bad = b[0][1]
    ratio = (good*1.0)/(good+bad)
    print('{sampleBy dataset: user number}:', good+bad)
    print('{sampleBy dataset: good}:',good)
    print('{sampleBy dataset: bad }:',bad)
    print('{sampleBy dataset: good ratio}:', ratio)
    df_sample = df_sample.na.fill(0)
    # feas_type = {}
    # df_type = df_sample.dtypes #各特征类型
    # for i in range(0,len(df_sample.columns)):
    #     feas_type[df_type[i][0]] = df_type[i][1]
    # strType = {k:v for k,v in feas_type.items() if v=='string' and k not in [IDcol]}
    # for col in strType:
    # stringIndexer = StringIndexer(inputCol='jdmall_user_p0011', outputCol="indexed")
    # model = stringIndexer.fit(df_sample)
    # df_sample = model.transform(df_sample)
    # df_sample = df_sample.withColumnRenamed('indexed','jdmall_user_p0011')
    # df_sample = df_sample.na.fill(-1)
    df_sample = df_sample.drop('jdmall_user_p0011').drop('risk_prior')
    return df_sample

if __name__ == '__main__':
    try:
        IDcol = sys.argv[1]
        target = sys.argv[2]
    except KeyboardInterrupt:
        pass

spark = SparkSession.builder.appName("user_cluster").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel('WARN')
data = spark.sql("select * from ft_tmp.yhj_lx_promotion")
data = data.na.fill(0)
a = data.groupBy(target).count()
b = a.sort(target).collect()
good = b[1][1]
bad = b[0][1]
ratio = (good*1.0)/(good+bad)
if good<500000:
    good_ratio = 1
    bad_ratio = round(500000.0/bad,5)
else:
    good_ratio = round(500000.0/good,5)
    bad_ratio = round(500000.0/bad,5)
n_samp_ratio = [bad_ratio, good_ratio]
data = data.withColumnRenamed(target,'label')
#数据采样
data_sample = getData(data, IDcol, target, n_samp_ratio=n_samp_ratio)
#特征与目标相关性分析
data_sample_1 = data_sample.drop('score')
data_sample_1 = data_sample_1.na.fill(0)
-- info_cols= colline_features(df_sample, IDcol, target, colline_threshold=0.8)

#百分位sql
def q_sql(col,q):
    sql = "SELECT quantiles ,ROUND(SUM(is_new)/COUNT(*),5) {} FROM (SELECT jdpin ,is_new,CASE WHEN {} <= {} THEN 'q1' WHEN {} <= {} THEN 'q2' WHEN {} <= {} THEN 'q3' WHEN {} <= {} THEN 'q4' ELSE NULL END quantiles FROM ft_tmp.yhj_lx_promotion )a GROUP BY quantiles ".format(col,col,q[0],col,q[1],col,q[2],col,q[3])
    return sql


bounds = {}
features = [col for col in data_sample_1.columns if col not in ['is_new','jdpin','bank_desire','user_ord_until_now']]
col1 = 'user_ord_until_now'
quantiles = data_sample_1.approxQuantile(col1,[0.25,0.5,0.75,1.0],0.05)
sql = q_sql(col,quantiles)
df_des = spark.sql(sql)
    

for col in features:
    quantiles = data_sample_1.approxQuantile(col,[0.25,0.5,0.75,1.0],0.05)
    sql = q_sql(col,quantiles)
    ratio = spark.sql(sql)
    df_des = df_des.join(ratio, df_des.quantiles == ratio.quantiles, 'left').drop(ratio.quantiles)



df_des = df_des.join(de_1, df_des.quantiles == de_1.quantiles, 'left').drop(de_1.quantiles)
