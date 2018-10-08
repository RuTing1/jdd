#!/bin/bash
#author:dingru
#create time: 2018-09-10
#This script is used to extract useful features from tables such as ybrb|zr|yuheng
yesterday=`date +"%Y-%m-%d" -d "-1 days"`
today=`date +"%Y%m%d"`

max_partition(){
    partition="set hive.cli.print.header=flase;
    show partitions $1;"
    max_part=`hive -S -e  "$partition" | grep '^dt' | sort | tail -n 1` #取最新的分区
    echo ${max_part:3:10}
}

date2=`max_partition 'ft_app.ftapp_ybr_b_s_m'`
date3=`max_partition 'ft_app.ftapp_zr_s_m'`
#date4=`max_partition 'dmt.dmt_tags_yuheng_score_params_jdmall_a_d'`
#date5=`max_partition 'dmt.dmt_tags_yuheng_score_params_a_d'`

#统计各表数据量
cnt2=`hive -S -e "select count(*) from ft_app.ftapp_ybr_b_s_m where dt='${date2}'"`
cnt3=`hive -S -e "select count(*) from ft_app.ftapp_zr_s_m where dt='${date3}'"`
#cnt4=`hive -S -e "select count(*) from dmt.dmt_tags_yuheng_score_params_jdmall_a_d where dt='${date4}'"`
#cnt5=`hive -S -e "select count(*) from dmt.dmt_tags_yuheng_score_params_a_d where dt='${date5}'"`
#输出各表最新日期以及统计数据结果
echo "ft_app.ftapp_ybr_b_s_m :'${date2}':$cnt2"
echo "ft_app.ftapp_zr_s_m  :'${date3}':$cnt3"
#echo "dmt.dmt_tags_yuheng_score_params_jdmall_a_d :'${date4}':$cnt4"
#echo "dmt.dmt_tags_yuheng_score_params_a_d :'${date5}':$cnt5"

#数据读取
sql="
SET hive.support.quoted.identifiers=None;
USE ft_tmp;
CREATE TABLE if not EXISTS ft_tmp.features_from_bz_${today} AS
SELECT ybrb.user_id
      ,\`(user_id)?+.+\` 
    FROM 
        (
        SELECT   user_log_acct user_id
                ,user_ord_until_now
                ,sale_ord_cnt_last_12
                ,sku_cnt_last_12
                ,actu_pay_amt_last_12
                ,sale_amt_avg_curr
                ,sale_amt_avg_last_3
                ,sale_amt_avg_last_6
                ,sale_amt_avg_last_12
                ,sale_amt_per_ord_max_last_12
                ,xiaofeiyuefei
                ,dingdanshu
                ,shifujine
                ,danbizuida
                ,sale_amt_per_ord_min_last_12
            FROM ft_app.ftapp_ybr_b_s_m 
            WHERE dt='${date2}'
        ) ybrb
        LEFT JOIN 
        (
        SELECT   user_log_acct user_id
                ,user_aging
                ,amt_actual_payment
                ,amt_pay_order
                ,amt_daofu_order
                ,amt_on_line_order
                ,amt_total_offer
                ,amt_full_minus_offer
                ,amt_dq_and_jq_pay
                ,amt_gift_cps_pay
                ,amt_rebate
                ,amt_ziti_order
                ,amt_lock_order
                ,amt_pause_order
                ,amt_object_order
                ,amt_virtual_order
                ,amt_game_order
                ,amt_lottery_order
                ,amt_order_book
                ,amt_order_car
                ,amt_order_digi
                ,amt_order_elec
                ,amt_order_fashion
                ,amt_order_food
                ,amt_order_gift
                ,amt_order_home
                ,amt_order_jewelry
                ,amt_order_med
                ,amt_order_sports
                ,amt_order_emor
                ,amt_order_mor
                ,amt_order_aft
                ,amt_order_nig
                ,amt_order_midnig
                ,cnt_order
                ,cnt_sku
                ,cnt_pay_order
                ,cnt_daofu_order
                ,cnt_on_line_order
                ,cnt_ziti_order
                ,cnt_object_ord
                ,cnt_virtual_ord
                ,cnt_game_order
                ,cnt_lottery_order
                ,cnt_order_book
                ,cnt_order_elec
                ,cnt_order_food
                ,cnt_order_gift
                ,cnt_order_med
                ,cnt_order_sports
                ,cnt_order_emor
                ,cnt_order_mor
                ,cnt_order_aft
                ,cnt_order_nig
                ,cnt_order_midnig
                ,pct_amt_daofu_vs_ol
                ,avg_amt_order
                ,avg_amt_pay_order
                ,avg_amt_on_line_order
                ,avg_amt_object_order
            FROM ft_app.ftapp_zr_s_m
            WHERE dt ='${date3}'
        )zr ON ybrb.user_id = zr.user_id
--        LEFT JOIN 
--        (
--       SELECT \`(etl_dt|user_key|jdmall_jdmuser_p0002816|dt)?+.+\` 
--            FROM dmt.dmt_tags_yuheng_score_params_jdmall_a_d
--            WHERE dt = '${date4}'
--        )yha ON ybrb.user_id = yha.user_id
--        LEFT JOIN
--        (
--        SELECT \`(etl_dt|user_key|cs_cs_f0002020|dt)?+.+\` 
--            FROM dmt.dmt_tags_yuheng_score_params_a_d
--            WHERE dt = '${date5}'
--        )yhb ON ybrb.user_id=yhb.user_id
        ;"

if [[ $cnt2 -lt  1000 || $cnt3 -lt  1000 ]]; then
        echo "there are empty tables!!!"
else
        hive -S -e "$sql"
fi




