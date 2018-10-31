#!/bin/bash
#Created on 20181030
#author: dingru1
#this script is used to mark y labels for bank sleep users 
#!/bin/bash
#Created on 20180912
#author: dingru1
#this script is used to mark y labels for bank sleep users 

#+++++++++++++++++++++++++++++++全局变量+++++++++++++++++++++++++++++++++#
today=`date +"%Y%m%d"`
yesterday=`date +"%Y-%m-%d" -d "-1 days"`
yesterday_mago=`date +"%Y-%m-%d" -d "-40 days"`
#脚本涉及表
tables=(ft_app.ftapp_ybr_b_s_m dmt.dmt_tags_lhyy_fin_icbc_a_d dmt.dmt_tags_lhyy_jdmall_icbc_1_a_d ft_app.ftapp_zr_s_m ft_app.ftapp_ybr_a_s_m)
#+++++++++++++++++++++++++++++++输出文件+++++++++++++++++++++++++++++++++#
#用户label表：ft_tmp.bank_sleep_customers_indexs_${today}

#+++++++++++++++++++++++++++++++函数++++++++++++++++++++++++++++++++++++#
#根据表和字段筛选字段有效分区的函数
max_partition(){
    partition="set hive.cli.print.header=flase;
    SELECT dt,COUNT(*) FROM $1 WHERE dt BETWEEN '${yesterday_mago}' AND '${yesterday}' GROUP BY dt;"
    max_part=`hive -S -e  "$partition" | awk '$2>1000'| sort -nr -k 1 | head | awk '{print $1}' ` #取最近有数据的10个分区
    int=1
    cnt=1
    while (("$cnt<2"))
    do
      date_i=`echo $max_part | awk '{print $"'$int'"}'`
      cnt=`hive -S -e "set hive.cli.print.header=flase; 
      SELECT $2,COUNT(*) FROM $1 WHERE dt='${date_i}' GROUP BY $2" | wc -l `
      let "int++"
    done
    echo $date_i  #取最近有数据的分区
}

max_partition_v2(){
    partition="set hive.cli.print.header=flase;
    SELECT dt,COUNT(*) FROM $1 WHERE dt BETWEEN '${yesterday_mago}' AND '${yesterday}' GROUP BY dt;"
    max_part=`hive -S -e  "$partition" | awk '$2>1000'| sort -nr -k 1 | head -n 1 ` #取最新有数据的分区
    echo "$1 $max_part"
}

#+++++++++++++++++++++++++++++++主程序++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++是否运行程序+++++++++++++++++++++++++++++#
hive -S -e "select * from ft_tmp.bank_sleep_customers_indexs_${today}"
if [$?==0];then
  echo "table exists"
else
  #特征提取使用到的表，对于数据不稳定的表进行有效特征时间推算
  dates[0]=`max_partition_v2 ft_app.ftapp_ybr_b_s_m`
  dates[1]=`max_partition dmt.dmt_tags_lhyy_fin_icbc_a_d fin_fin_f0002792`
  dates[2]=`max_partition dmt.dmt_tags_lhyy_fin_icbc_a_d fin_fund_f0002098`
  dates[3]=`max_partition dmt.dmt_tags_lhyy_jdmall_icbc_1_a_d pay_syt_f0013`
  dates[4]=`max_partition dmt.dmt_tags_lhyy_fin_icbc_a_d mem_mem_f0005266`
  dates[5]=`max_partition_v2 ft_app.ftapp_zr_s_m`
  dates[6]=`max_partition_v2 ft_app.ftapp_ybr_a_s_m dangqianzichan`
# date[1]=`(echo ${date_a[@]} | tr ' ' '\n' | sort -nr )`
  sql_index="
  SET hive.support.quoted.identifiers=None;
  USE ft_tmp;
  CREATE TABLE IF NOT EXISTS ft_tmp.bank_sleep_customers_indexs_${today} AS
  SELECT all_users.user_id 
        ,CASE WHEN inv_users.user_id IS NOT NULL THEN 1 ELSE 0 END inv_desire
        ,CASE WHEN current_users.user_id IS NOT NULL THEN 1 ELSE 0 END current_fin
        ,CASE WHEN load_users.user_id IS NOT NULL THEN 1 ELSE 0 END load_desire
        ,CASE WHEN fund_users.user_id IS NOT NULL THEN 1 ELSE 0 END fun_desire
        ,CASE WHEN bank_users.user_id IS NOT NULL THEN 1 ELSE 0 END bank_desire
        ,CASE WHEN stock_users.user_id IS NOT NULL THEN 1 ELSE 0 END stock_desire
        ,CASE WHEN card_desire.user_id IS NOT NULL THEN 1 ELSE 0 END card_desire
      FROM 
      (
        SELECT user_log_acct user_id
          FROM ft_app.ftapp_ybr_b_s_m 
          WHERE dt = '${dates[0]}'
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
      SELECT user_id
        FROM dmt.dmt_tags_lhyy_fin_icbc_a_d
      --  WHERE dt='${dates[1]}' AND (fin_qyhz_f0007976=1 OR sec_stock_f0000736=1)
        WHERE dt='${dates[1]}' AND fin_fin_f0002792=1
      )current_users ON all_users.user_id = current_users.user_id
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
      UNION
      SELECT user_id
          FROM dmt.dmt_tags_lhyy_fin_icbc_a_d
          WHERE  dt='${dates[2]}' AND fin_fund_f0002098=5
      )stock_users ON all_users.user_id = stock_users.user_id
    LEFT JOIN
      (
      SELECT user_id
          FROM dmt.dmt_tags_lhyy_jdmall_icbc_1_a_d
          WHERE dt='${dates[3]}' AND pay_syt_f0013 IN (2,3)
      ) card_desire ON all_users.user_id = card_desire.user_id
  ;"

  sql_assertvalue="
  SET hive.support.quoted.identifiers=None;
  USE ft_tmp;
  CREATE TABLE IF NOT EXISTS ft_tmp.bank_sleep_customers_assert_value_${today} AS
  SELECT
    jdpin,
    CASE WHEN actu_pay_amt_last_12_level+sale_amt_per_ord_max_last_12_level+sale_amt_avg_last_12_level+pct_3_pay_amt_level+pay_amt_incr_level+order_car_level+dangqianzichan_level+virtual_level+mem_level>100 THEN 100
         WHEN actu_pay_amt_last_12_level+sale_amt_per_ord_max_last_12_level+sale_amt_avg_last_12_level+pct_3_pay_amt_level+pay_amt_incr_level+order_car_level+dangqianzichan_level+virtual_level+mem_level<0 THEN 0
         ELSE actu_pay_amt_last_12_level+sale_amt_per_ord_max_last_12_level+sale_amt_avg_last_12_level+pct_3_pay_amt_level+pay_amt_incr_level+order_car_level+dangqianzichan_level+virtual_level+mem_level END as score
  FROM
  (
  SELECT 
      a.user_log_acct AS jdpin,
      CASE WHEN actu_pay_amt_last_12<=100 THEN 5
           WHEN actu_pay_amt_last_12<=400 THEN 8
           WHEN actu_pay_amt_last_12<=1100 THEN 14
           WHEN actu_pay_amt_last_12<=2700 THEN 16
           WHEN actu_pay_amt_last_12<=5000 THEN 19
           WHEN actu_pay_amt_last_12<=9000 THEN 22
           WHEN actu_pay_amt_last_12<=15000 THEN 25
           WHEN actu_pay_amt_last_12>15000 THEN 30
           ELSE 5 END AS actu_pay_amt_last_12_level,
      CASE WHEN sale_amt_per_ord_max_last_12<=90 THEN 3
           WHEN sale_amt_per_ord_max_last_12<=160 THEN 8
           WHEN sale_amt_per_ord_max_last_12<=300 THEN 12
           WHEN sale_amt_per_ord_max_last_12<=800 THEN 14
           WHEN sale_amt_per_ord_max_last_12<=1500 THEN 16
           WHEN sale_amt_per_ord_max_last_12<=2600 THEN 18
           WHEN sale_amt_per_ord_max_last_12<=4000 THEN 20
           WHEN sale_amt_per_ord_max_last_12>4000 THEN 22
           ELSE 3 END AS sale_amt_per_ord_max_last_12_level,
      CASE WHEN sale_amt_avg_last_12<=60 THEN 3
           WHEN sale_amt_avg_last_12<=130 THEN 8
           WHEN sale_amt_avg_last_12<=200 AND sale_ord_cnt_last_12>=4 THEN 14
           WHEN sale_amt_avg_last_12>200 AND sale_amt_avg_last_12<=300 AND sale_ord_cnt_last_12>=5 THEN 16
           WHEN sale_amt_avg_last_12>300 AND sale_amt_avg_last_12<=570 AND sale_ord_cnt_last_12>=7 THEN 18
           WHEN sale_amt_avg_last_12>570 AND sale_ord_cnt_last_12>=10 THEN 20
           ELSE 3 END AS sale_amt_avg_last_12_level,
      CASE WHEN actu_pay_amt_last_3 / actu_pay_amt_last_12<=0.13 THEN 3
           WHEN actu_pay_amt_last_3 / actu_pay_amt_last_12<=0.4 THEN 6
           WHEN actu_pay_amt_last_3 / actu_pay_amt_last_12<=0.8 THEN 8
           WHEN actu_pay_amt_last_3 / actu_pay_amt_last_12>0.8 THEN 10
           ELSE 3 END AS pct_3_pay_amt_level,
      CASE WHEN actu_pay_amt_incr_last_12<=1 THEN 4
           WHEN actu_pay_amt_incr_last_12=2 THEN 6
           WHEN actu_pay_amt_incr_last_12=3 THEN 8
           WHEN actu_pay_amt_incr_last_12>3 THEN 10
           ELSE 4 END AS pay_amt_incr_level,
      CASE WHEN amt_order_car>300 AND amt_order_car<=460 THEN 1
           WHEN amt_order_car>460 AND amt_order_car<=900 THEN 3
           WHEN amt_order_car>900 THEN 5
           ELSE 0 END AS order_car_level,
      CASE WHEN dangqianzichan='e' THEN 5
           WHEN dangqianzichan = 'f' THEN 10
           ELSE 0 END AS dangqianzichan_level,
      CASE WHEN amt_virtual_order>10000 AND pct_amt_virtual_vs_tot=1 THEN -25
           ELSE 0 END AS virtual_level,
      CASE WHEN mem_mem_f0005266 in ('8', '9', '10') THEN 5
           WHEN mem_mem_f0005266 in ('11', '12', '13') THEN 10
           WHEN mem_mem_f0005266 in ('14', '15', '16', '17') THEN 15
           ELSE 0 END AS mem_level
  FROM 
    (
    SELECT user_log_acct
          ,actu_pay_amt_last_12
          ,sale_amt_per_ord_max_last_12
          ,sale_amt_avg_last_12
          ,actu_pay_amt_last_3
          ,actu_pay_amt_incr_last_12
          ,sale_ord_cnt_last_12
        FROM ft_app.ftapp_ybr_b_s_m 
        WHERE dt='${dates[0]}' 
    )a
  LEFT JOIN
    (
    SELECT user_log_acct
          ,amt_order_car
          ,amt_virtual_order
          ,pct_amt_virtual_vs_tot
        FROM ft_app.ftapp_zr_s_m 
        WHERE dt='${dates[5]}'
    ) b ON a.user_log_acct=b.user_log_acct
  LEFT JOIN 
    (
    SELECT jdpin
          ,dangqianzichan
     FROM ft_app.ftapp_ybr_a_s_m 
     WHERE dt='${dates[6]}'
    )c ON a.user_log_acct=c.jdpin
  LEFT JOIN
    (
    SELECT user_id
          ,mem_mem_f0005266 
        FROM dmt.dmt_tags_lhyy_fin_icbc_a_d 
        WHERE dt='${dates[4]}'
    ) d ON a.user_log_acct=d.user_id
  ) a;"
  if [ !${dates[0]} ] && [ !${dates[5]} ] && [ !${dates[6]} ]; then
      hive -S -e "$sql_index" && hive -S -e "$sql_assertvalue"
  else 
    echo "tables that update every month maybe with no data recently...please check them..."
  fi
fi
