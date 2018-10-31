#!/bin/bash

#+++++++++++++++++++++++++++++++全局变量+++++++++++++++++++++++++++++++++#
today=`date +"%Y%m%d"`
yesterday=`date +"%Y-%m-%d" -d "-1 days"`
yesterday_mago=`date +"%Y-%m-%d" -d "-40 days"`
#脚本涉及表
tables=(ft_app.ftapp_ybr_a_s_m dmt.dmt_tags_lhyy_jdmall_icbc_1_a_d)

#+++++++++++++++++++++++++++++++输出文件+++++++++++++++++++++++++++++++++#
#用户各指标数据表：bank_sleep_customers_probability PARTITION(dt='${today}')

#+++++++++++++++++++++++++++++++函数++++++++++++++++++++++++++++++++++++#
#表有效时间提取
max_partition(){
    partition="set hive.cli.print.header=flase;
    SELECT dt,COUNT(*) FROM $1 WHERE dt BETWEEN '${yesterday_mago}' AND '${yesterday}' GROUP BY dt;"
    max_part=`hive -S -e  "$partition" | awk '$2>1000'| sort -nr -k 1 | head -n 1 ` #取最新有数据的分区
    echo "$1 $max_part"
}
#+++++++++++++++++++++++++++++++主程序++++++++++++++++++++++++++++++++++++#
for ((i=0;i<${#tables[@]};i++)); do
    a=`max_partition ${tables[i]}`
    echo $a
    dates[$i]=`echo $a | awk '{print $2}'`
done

sql0="
USE ft_tmp;
CREATE TABLE IF NOT EXISTS bank_sleep_customers_probability_v2(
                jdpin                  STRING   COMMENT    '用户pin',
                inv_desire             DOUBLE   COMMENT    '投资意愿',
                load_desire            DOUBLE   COMMENT    '借贷意愿',
                current_fin            DOUBLE   COMMENT    '活期产品投资意愿分',
                bank_desire            DOUBLE   COMMENT    '银行类产品投资意愿分',
                fund_deisre            DOUBLE   COMMENT    '基金类产品投资意愿分',
                risk_deisre            DOUBLE   COMMENT    '投资风险偏好分',
                card_deisre            DOUBLE   COMMENT    '办信用卡意愿',
                assert_value           DOUBLE   COMMENT    '资产价值',
                markt_senstivity       BIGINT   COMMENT    '促销敏感度',
                profession             BIGINT   COMMENT    '职业预测',
                city_level             BIGINT   COMMENT    '城市预测',             
                user_aging             BIGINT   COMMENT    '京东商城账龄等级'            
)COMMENT '用户金融属性刻画' PARTITIONED BY (dt STRING COMMENT '操作日期')
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' STORED AS TEXTFILE;
"


sql1= "
USE ft_tmp;
ALTER TABLE bank_sleep_customers_probability_v2 DROP IF EXISTS PARTITION(dt='${today}');
INSERT OVERWRITE TABLE bank_sleep_customers_probability_v2 PARTITION(dt='${today}')
SELECT inv_desire.jdpin jdpin
      ,COALESCE(inv_desire,-1) inv_desire
      ,COALESCE(load_desire,-1)load_desire
      ,COALESCE(current_fin,-1)current_fin
      ,COALESCE(bank_desire,-1)bank_desire
      ,COALESCE(fund_deisre,-1)fund_deisre
      ,COALESCE(risk_deisre,-1)risk_deisre 
      ,COALESCE(card_deisre,-1)card_deisre 
      ,COALESCE(assert_value,-1)assert_value    
      ,COALESCE(markt_senstivity,-1) markt_senstivity
      ,COALESCE(profession,-1) profession
      ,COALESCE(city_level,-1) city_level
      ,COALESCE(user_aging,-1) user_aging
    FROM
    (
    SELECT  jdpin
           ,round(probability1*100,3) inv_desire
        FROM ft_tmp.yhj_RandomForestClassifier_inv_desire_probability1_all
    ) inv_desire
LEFT JOIN
    (
    SELECT  jdpin
       ,round(probability1*100,3) load_desire
    FROM ft_tmp.yhj_RandomForestClassifier_load_desire_probability1_all
    ) load_desire ON inv_desire.jdpin = load_desire.jdpin
LEFT JOIN
    (
    SELECT  jdpin
       ,round(probability1*100,3) current_fin
    FROM ft_tmp.yhj_RandomForestClassifier_current_fin_probability1_all
    ) current_fin ON inv_desire.jdpin = current_fin.jdpin
LEFT JOIN
    (
    SELECT jdpin
       ,round(probability1*100,3) bank_desire
    FROM ft_tmp.yhj_RandomForestClassifier_bank_desire_probability1_all
    ) is_buy ON inv_desire.jdpin = is_buy.jdpin
LEFT JOIN
    (
    SELECT jdpin
       ,round(probability1*100,3) fund_deisre
    FROM ft_tmp.yhj_RandomForestClassifier_fun_desire_probability1_all
    ) is_fun ON inv_desire.jdpin = is_fun.jdpin
LEFT JOIN
    (
    SELECT jdpin
          ,round(probability1*100,3) risk_deisre
    FROM ft_tmp.yhj_RandomForestClassifier_stock_desire_probability1_all
    ) stock_desire ON inv_desire.jdpin = stock_desire.jdpin
LEFT JOIN
    (
    SELECT jdpin
          ,round(probability1*100,3) card_deisre
    FROM ft_tmp.yhj_RandomForestClassifier_card_desire_probability1_all
    )card_deisre ON inv_desire.jdpin = card_deisre.jdpin
LEFT JOIN
    (
    SELECT jdpin
          ,score assert_value
      FROM ft_tmp.bank_sleep_customers_assert_value_${today}
    )assert_value ON inv_desire.jdpin = assert_value.jdpin
LEFT JOIN
    (
     SELECT user_id jdpin
          ,CASE WHEN jdmall_jdmall_f0003019 IN(1,2,3) THEN 1 
                WHEN jdmall_jdmall_f0003019 =4 THEN 2 
                WHEN jdmall_jdmall_f0003019 =5 THEN 3 
                WHEN jdmall_jdmall_f0003019 =6 THEN 4 
                ELSE -1 END markt_senstivity--促销敏感度
          ,CASE WHEN jdmall_jdmup_m0000641 =-9999 THEN -1 WHEN jdmall_jdmup_m0000641<>0 THEN jdmall_jdmup_m0000641 ELSE -1 END profession --职业预
测
          ,CASE WHEN jdmall_jdmuser_p0002816 =-9999 THEN -1 WHEN jdmall_jdmuser_p0002816<>0 THEN jdmall_jdmuser_p0002816 ELSE -1 END city_level--城
市级别
        FROM dmt.dmt_tags_lhyy_jdmall_icbc_1_a_d
        WHERE dt= '${dates[0]}'
    )label_reflect ON inv_desire.jdpin = label_reflect.jdpin
LEFT JOIN
    (
     SELECT jdpin
           ,CASE WHEN user_aging='a' THEN 1 
                 WHEN user_aging='b' THEN 2
                 WHEN user_aging='c' THEN 3
                 WHEN user_aging='d' THEN 4
                 WHEN user_aging='e' THEN 5
                 WHEN user_aging='f' THEN 6
            ELSE -1 END user_aging
       FROM ft_app.ftapp_ybr_a_s_m 
       WHERE dt='${dates[1]}'
    )aging ON inv_desire.jdpin = aging.jdpin
;"

hive -S -e "$sql0" && hive -S -e "$sql1"
