#!/bin/bash
#author:dingru
#create time: 2018-10-08
#This script is used to extract useful features from tables such as ybrb|zr
#example:  sh useful_features.sh   #命令行后面跟任意参数表示重新提取当前日期的useful_features,否则不重新提取

yesterday=`date +"%Y-%m-%d" -d "-1 days"`
today=`date +"%Y%m%d"`

#标记跑数据的日期，方便对日志进行追踪
echo '##################################################################'
echo '##############ft_tmp.useful_features_from_bz '$today'#############'
echo '##################################################################'

max_partition(){
    partition="set hive.cli.print.header=flase;
    show partitions $1;"
    max_part=`hive -S -e  "$partition" | grep '^dt' | sort | tail -n 1` #取最新的分区
    echo ${max_part:3:10}
}

tables=(ft_app.ftapp_ybr_b_s_m ft_app.ftapp_zr_s_m)
#建模前各表最新数据监控
for ((i=0;i<${#tables[@]};i++)); do
        dates[$i]=`max_partition ${tables[i]}`
done

sql0="
USE ft_tmp;
CREATE TABLE IF NOT EXISTS useful_features_from_bz(
                user_id                               STRING   COMMENT    '用户pin',
                user_ord_until_now                    DOUBLE   COMMENT    '客户最近订单距今天数',
                sale_ord_cnt_last_12                  DOUBLE   COMMENT    '客户近12个月总订单数',
                sku_cnt_last_12                       DOUBLE   COMMENT    '客户近12个月购买商品总数量',
                actu_pay_amt_last_12                  DOUBLE   COMMENT    '客户近12个月实付总金额',
                sale_amt_avg_curr                     DOUBLE   COMMENT    '客户当月平均单笔订单金额',
                sale_amt_avg_last_3                   DOUBLE   COMMENT    '客户近3个月平均单笔订单金额',
                sale_amt_avg_last_6                   DOUBLE   COMMENT    '客户近6个月平均单笔订单金额',
                sale_amt_avg_last_12                  DOUBLE   COMMENT    '客户近12个月平均单笔订单金额',
                sale_amt_per_ord_max_last_12          DOUBLE   COMMENT    '客户近12个月单笔订单最大金额',
                xiaofeiyuefei                         DOUBLE   COMMENT    '近12个月有消费月份数',
                dingdanshu                            DOUBLE   COMMENT    '近12个月消费订单数',
                shifujine                             DOUBLE   COMMENT    '近12个月消费实付金额',
                danbizuida                            DOUBLE   COMMENT    '近12个月单笔最大消费',
                sale_amt_per_ord_min_last_12          DOUBLE   COMMENT    '客户近12个月单笔订单最小金额',
                amt_actual_payment                    DOUBLE   COMMENT    '实际付款金额',
                amt_pay_order                         DOUBLE   COMMENT    '先款订单金额',
                amt_daofu_order                       DOUBLE   COMMENT    '到付订单金额',
                amt_on_line_order                     DOUBLE   COMMENT    '在线支付订单金额',
                amt_total_offer                       DOUBLE   COMMENT    '总优惠金额',
                amt_full_minus_offer                  DOUBLE   COMMENT    '满减优惠金额',
                amt_dq_and_jq_pay                     DOUBLE   COMMENT    '东券/京券支付金额',
                amt_gift_cps_pay                      DOUBLE   COMMENT    '礼品卡支付金额',
                amt_rebate                            DOUBLE   COMMENT    '返现金额',
                amt_ziti_order                        DOUBLE   COMMENT    '自提订单金额',
                amt_lock_order                        DOUBLE   COMMENT    '锁定订单金额',
                amt_pause_order                       DOUBLE   COMMENT    '暂停订单金额',
                amt_object_order                      DOUBLE   COMMENT    '实物订单金额',
                amt_virtual_order                     DOUBLE   COMMENT    '虚拟订单金额',
                amt_game_order                        DOUBLE   COMMENT    '游戏订单金额',
                amt_lottery_order                     DOUBLE   COMMENT    '彩票订单定额',
                amt_order_book                        DOUBLE   COMMENT    '图书音像类商品消费金额',
                amt_order_car                         DOUBLE   COMMENT    '汽车用品消费金额',
                amt_order_digi                        DOUBLE   COMMENT    '电子商品消费金额',
                amt_order_elec                        DOUBLE   COMMENT    '家用电器消费金额',
                amt_order_fashion                     DOUBLE   COMMENT    '时尚类商品消费金额',
                amt_order_food                        DOUBLE   COMMENT    '生鲜食品类消费金额',
                amt_order_gift                        DOUBLE   COMMENT    '礼品类商品消费金额',
                amt_order_home                        DOUBLE   COMMENT    '家居用品消费金额',
                amt_order_jewelry                     DOUBLE   COMMENT    '珠宝首饰消费金额',
                amt_order_med                         DOUBLE   COMMENT    '医药器械消费金额',
                amt_order_sports                      DOUBLE   COMMENT    '运动户外下单金额',
                amt_order_emor                        DOUBLE   COMMENT    '凌晨下单金额',
                amt_order_mor                         DOUBLE   COMMENT    '早上下单金额',
                amt_order_aft                         DOUBLE   COMMENT    '下午下单金额',
                amt_order_nig                         DOUBLE   COMMENT    '晚上下单金额',
                amt_order_midnig                      DOUBLE   COMMENT    '深夜下单金额',
                cnt_order                             DOUBLE   COMMENT    '总订单数',
                cnt_sku                               DOUBLE   COMMENT    '总sku数',
                cnt_pay_order                         DOUBLE   COMMENT    '先款订单数',
                cnt_daofu_order                       DOUBLE   COMMENT    '到付订单数',
                cnt_on_line_order                     DOUBLE   COMMENT    '在线支付订单数',
                cnt_ziti_order                        DOUBLE   COMMENT    '自提订单数',
                cnt_object_ord                        DOUBLE   COMMENT    '实物订单数',
                cnt_virtual_ord                       DOUBLE   COMMENT    '虚拟订单数',
                cnt_game_order                        DOUBLE   COMMENT    '游戏相关订单数',
                cnt_lottery_order                     DOUBLE   COMMENT    '彩票订单定额',
                cnt_order_book                        DOUBLE   COMMENT    '图书音像类商品消费订单数',
                cnt_order_elec                        DOUBLE   COMMENT    '家用电器消费订单数',
                cnt_order_food                        DOUBLE   COMMENT    '生鲜食品类消费订单数',
                cnt_order_gift                        DOUBLE   COMMENT    '礼品类商品消费订单数',
                cnt_order_med                         DOUBLE   COMMENT    '医药器械消费订单数',
                cnt_order_sports                      DOUBLE   COMMENT    '运动户外下单订单数',
                cnt_order_emor                        DOUBLE   COMMENT    '凌晨下单订单数',
                cnt_order_mor                         DOUBLE   COMMENT    '早上下单订单数',
                cnt_order_aft                         DOUBLE   COMMENT    '下午下单订单数',
                cnt_order_nig                         DOUBLE   COMMENT    '晚上下单订单数',
                cnt_order_midnig                      DOUBLE   COMMENT    '深夜下单订单数',
                pct_amt_daofu_vs_ol                   DOUBLE   COMMENT    '到付金额vs在线支付金额',
                avg_amt_order                         DOUBLE   COMMENT    '平均订单金额',
                avg_amt_pay_order                     DOUBLE   COMMENT    '平均先款订单金额',
                avg_amt_on_line_order                 DOUBLE   COMMENT    '平均在线支付订单金额',
                avg_amt_object_order                  DOUBLE   COMMENT    '平均实物订单金额'
)COMMENT '建模特征表' PARTITIONED BY (dt STRING COMMENT '操作日期')
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' STORED AS TEXTFILE;
"

sql1="
SET hive.support.quoted.identifiers=None;
USE ft_tmp;
ALTER TABLE useful_features_from_bz DROP IF EXISTS PARTITION(dt='${today}');
INSERT OVERWRITE TABLE useful_features_from_bz PARTITION(dt='${today}')
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
            WHERE dt='${dates[0]}'
        ) ybrb
        LEFT JOIN 
        (
        SELECT   user_log_acct user_id
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
            WHERE dt ='${dates[1]}'
        )zr ON ybrb.user_id = zr.user_id
        ;"

echo "create table if not exists..."
hive -S -e "$sql0" && hive -S -e "$sql1"

#if [[ $dates_new -lt $today ]]; then   #对useful_features_from_bz的最新时间进行判断，若小于当前日期则执行SQL1
        #feas=($(hive -S -e "DESC ft_tmp.useful_features_from_bz" | awk '{print $1}'))
                #for ((i=0;i<${#feas[@]};i++)); do
                        #hive -S -e "select '$today','${feas[i]}', count(${feas[i]})/count(*) from ft_tmp.useful_features_from_bz;">>yhj_feas_missin
gratio.csv
                #done
#fi
