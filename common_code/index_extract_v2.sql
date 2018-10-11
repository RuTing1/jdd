#!/bin/bash
#Created on 20180912
#author: dingru1
#this script is used to mark y labels for bank sleep users 
yesterday=`date +"%Y-%m-%d" -d "-1 days"`
today=`date +"%Y%m%d"`

max_partition(){
    partition="set hive.cli.print.header=flase;
    show partitions $1;"
    max_part=`hive -S -e  "$partition" | grep '^dt' | sort | tail -n 1` #取最新的分区
    echo ${max_part:3:10}
}

date2=`max_partition 'ft_app.ftapp_ybr_b_s_m'`
date3=`max_partition 'dmt.dmt_tags_lhyy_fin_icbc_a_d'`

hive -e"
SET hive.support.quoted.identifiers=None;
USE ft_tmp;
DROP TABLE if EXISTS ft_tmp.bank_sleep_customers_indexs_${today};
CREATE TABLE ft_tmp.bank_sleep_customers_indexs_${today} AS
WITH a AS (
  SELECT user_pin user_id
        ,ord_type
        ,sku_id
      FROM dws.dws_fin_dlc_tx_ordr_det_i_d
      WHERE dt<='$yesterday' AND is_tx_succ=1 
            AND ord_type IN('fin_insu','fin_xby','fin_ylbz','fin_fund') AND user_pin IS NOT NULL 
      GROUP BY user_pin,ord_type,sku_id
),
b AS (
  SELECT CASE WHEN src_sys='insu' THEN 'fin_insu' WHEN src_sys='fb-xby' THEN 'fin_xby' WHEN src_sys='insu-ylbz' THEN 'fin_ylbz' ELSE NULL END ord_type
        ,sku_id
      FROM dwb.dwb_fin_prod_info_jf_s_d 
    WHERE dt='$yesterday' AND src_sys IN ('insu','fb-xby','insu-ylbz') AND prod_term=1
  GROUP BY src_sys,sku_id
)
SELECT all_users.user_id 
      ,CASE WHEN inv_users.user_id IS NOT NULL THEN 1 ELSE 0 END inv_desire
      ,CASE WHEN current_users.user_id IS NOT NULL THEN 1 ELSE 0 END current_fin
      ,CASE WHEN load_users.user_id IS NOT NULL THEN 1 ELSE 0 END load_desire
      ,CASE WHEN fund_users.user_id IS NOT NULL THEN 1 ELSE 0 END fun_desire
      ,CASE WHEN bank_users.user_id IS NOT NULL THEN 1 ELSE 0 END bank_desire
      ,CASE WHEN stock_users.user_id IS NOT NULL THEN 1 ELSE 0 END stock_desire
    FROM 
    (
      SELECT user_log_acct user_id
        FROM ft_app.ftapp_ybr_b_s_m 
        WHERE dt = '${date2}'
    )all_users
    LEFT JOIN
    (
    SELECT user_pin user_id
      FROM dws.dws_fin_tx_ordr_det_i_d 
      WHERE dt<='${yesterday}' AND is_tx_succ = '1' AND user_pin IS NOT NULL
      GROUP BY user_pin
    UNION
    SELECT jd_pin user_id 
      FROM dwd.dwd_basic_fin_xjk_open_acct_s_d  
      WHERE dt='${yesterday}' AND is_open_succ=1
            AND yn=1 AND jd_pin IS NOT NULL AND jd_pin<>''
      GROUP BY jd_pin
    )inv_users ON all_users.user_id = inv_users.user_id
    LEFT JOIN
    (
    SELECT user_pin user_id
      FROM dws.dws_fin_dlc_tx_ordr_det_i_d
      WHERE dt<='$yesterday' AND is_tx_succ=1 AND ord_type='fin_xjk' 
      GROUP BY user_pin
    UNION
    SELECT user_id FROM a JOIN b ON (a.sku_id=b.sku_id AND a.ord_type=b.ord_type)
    UNION
    SELECT user_id 
      FROM 
      (
       SELECT user_id
             ,sku_id
          FROM a WHERE ord_type='fin_fund'
      )t1 
      left semi join 
      (
      SELECT sku_id 
        FROM dwb.dwb_fin_prod_info_jf_s_d 
        WHERE dt='$yesterday' AND src_sys='fund-fd' AND biz_nm4 in ('理财型','货币型')
      GROUP BY sku_id
      ) t2 ON t1.sku_id=t2.sku_id
    GROUP BY user_id
    ) current_users ON all_users.user_id = current_users.user_id
    LEFT JOIN
    (
    SELECT user_jrid user_id
      FROM  dwd.dwd_wallet_crpl_loan_apply_i_d 
      GROUP BY user_jrid
    UNION 
    SELECT jd_pin  user_id
      FROM   dwb.dwb_bt_order_s_d
      WHERE  dt = '$yesterday' and bizcode = '32'
      GROUP BY jd_pin
    ) load_users ON all_users.user_id = load_users.user_id
LEFT JOIN 
    (
    SELECT jd_pin user_id
      FROM dwd.dwd_basic_fin_fund_trade_s_d 
      WHERE dt='$yesterday' AND sell_type IN ('基金直销','基金代销') 
            AND tx_type='purch' AND is_tx_succ=1 AND jd_pin IS NOT NULL 
      GROUP BY jd_pin  
    ) fund_users ON all_users.user_id = fund_users.user_id
LEFT JOIN
    (
    SELECT jdpin user_id
      FROM dwd.dwd_app_jr_fp_trade_history_i_d 
      WHERE trade_type=0 AND valid_sign='1' AND trade_state='00'
      GROUP BY jdpin
    ) bank_users ON all_users.user_id = bank_users.user_id
LEFT JOIN
    (
    SELECT pin user_id
        FROM dwd.dwd_fin_stock_apply_account_s_d
        WHERE dt='$yesterday' AND yn=1 AND result='ok'
        AND pin IS NOT NULL
        GROUP BY pin 
    UNION
    SELECT pin user_id
        FROM dwd.dwd_fin_stock_apply_account_anxin_s_d
        WHERE dt='$yesterday' AND yn=1 AND result='ok'
        AND pin IS NOT NULL
        GROUP BY pin
    )stock_users ON all_users.user_id = stock_users.user_id
;"






