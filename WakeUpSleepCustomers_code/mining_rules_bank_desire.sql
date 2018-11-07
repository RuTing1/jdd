#bank_desire验证:银行plus
#20181015:银行+总用户量：
#20180914-20181015新增用户:
#20180914-20181015购买用户:

#step1:数据准备
SET hive.support.quoted.identifiers=None;
USE ft_tmp;
CREATE TABLE ft_tmp.yhj_lx_promotion AS
WITH buy_users AS (
    SELECT id

          ,channel_id
          ,project_id
          ,project_name
          ,MIN(trade_amount) trade_amount
          ,to_date(trade_date) trade_date
        FROM 
        WHERE
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
        FROM 
        WHERE dt='20180914'
    )a 
    LEFT JOIN 
    (
    SELECT `(dt)?+.+` 
        FROM 
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
#1.登录app距今时长
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
          ,
        FROM 
        WHERE dt='2018-10-16'
    )b ON a.jdpin=b.user_id
GROUP BY score,time_scope


#2.银行理财持仓
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
zz
        FROM 
        WHERE dt='2018-10-16'
    )b ON a.jdpin=b.user_id
GROUP BY score,position

#以上两项设置规则
a)	附加金融标签-最近一次登录金融APP距今天的天数，规则建议
①	最近一次登录金融app距今时长(0,7] 且预测评分10分以下用户评分
②	最近一次登录金融app距今时长(180,]且预测评分80分以上，预测评分设置为
③	最近一次登录金融app距今时长(90,180]且预测评分80分以上，预测评分设置为
b)	附件金融标签-理财持仓(含plus)，规则建议
①	理财持仓20W+，预测评分设置为
②	理财持仓1000-，预测评分设置为

#规则数据存储
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
