#!/bin/bash
#Created on 20181030
#author: dingru1
#this script is used to control for running marketing models,contains:
#1.know your tables: 最新分区是否有数据，Y-转2，N-stop
#2.extract features: 提取建模需要的特征，为保证后续模型分析，建立分区表
#3.extract labels: 提取建模的目标变量
#4.combine data: 特征+标签组合成模型数据
#5.train models: 训练模型
#6.save data：对模型训练过程中产生的结果数据进行保存
#7.delete data：对模型训练过程中产生的临时性数据进行删除
#8.monitor model: 对模型后续表现的稳定性进行监控

#可能存在的问题
#1.数据表虽然有数，特定的字段全部为空值,如pay_syt_f0013 dmt.dmt_tags_lhyy_jdmall_icbc_1_a_d dt='2018-10-21'

#+++++++++++++++++++++++++++++++全局变量+++++++++++++++++++++++++++++++++#
yesterday=`date +"%Y-%m-%d" -d "-1 days"`
yesterday_mago=`date +"%Y-%m-%d" -d "-32 days"`
today=`date +"%Y%m%d"`
IDcol=jdpin
targets=(inv_desire load_desire current_fin bank_desire fund_deisre risk_deisre card_desire)
model=rf
trainmodelflag=0
#+++++++++++++++++++++++++++++++函数++++++++++++++++++++++++++++++++++++#
#提取表最新有效数据分区
max_partition(){
    partition="set hive.cli.print.header=flase;
    SELECT dt,COUNT(*) FROM $1 WHERE dt BETWEEN '${yesterday_mago}' AND '${yesterday}' GROUP BY dt;"
    max_part=`hive -S -e  "$partition" | awk '$2>1000'| sort -nr -k 1 | head -n 1 ` #取最新有数据的分区
    echo "$1 $max_part"
}
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++脚本需要的表及存储的表++++++++++++++++++++++++++++++#
#提取特征存储表
table_feas=ft_tmp.useful_features_from_bz
#提取标签存储表：临时表
table_index=ft_tmp.bank_sleep_customers_indexs_${today}
table_all=ft_tmp.bank_sleep_customers_${today}
#数据存储表：
#各模型验证用户预测数据：ft_tmp.yhj_{classifier}_{flag}_probability1
#各模型全量用户预测数据：ft_tmp.yhj_{classifier}_{flag}_probability1_all
#各模型最终聚合结果数据：ft_tmp.bank_sleep_customers_probability (dt='20181010')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++step2：提取建模需要的特征++++++++++++++++++++++++++#
start_time=`date +%s`

dates_new=`max_partition ${table_feas} | awk '{print $2}'`
cnt=`hive -S -e "select count(*) from  $table_feas where dt='${today}'"`
if [[ $dates_new != $today ]]||[[ $cnt -lt 1000 ]]; then
    sh useful_features.sh
    state1=$?
else
    echo "useful_features table is already exists..."
    state1=0
fi

end_time=`date +%s`
echo "step1...extract features ...time cost...: $((end_time-start_time)) s"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++step3：提取建模目标变量++++++++++++++++++++++++++++#
#index_extract.sh 此文件涉及的相关表最新有数据分区需要核实
#数据存储：ft_tmp.bank_sleep_customers_indexs_${today}
start_time=`date +%s`
if (( $state1==0 )); then
    sh index_extract.sh
    state2=$?
    cnt=`hive -S -e "select count(*) from  $table_index"`
    if (( $state2 != 0 ))||[[ $cnt -lt 1000 ]]; then
        $state2=1
        echo "index table is empty..."
    else
        state2=0
    fi
fi
end_time=`date +%s`
echo "step2...extract labels ...time cost...: $((end_time-start_time)) s"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++step4：特征+标签组合成模型数据+++++++++++++++++++++#
start_time=`date +%s`
if (( $state1==0 ))&&(( $state2==0 )) ; then
    hive -S -e "
    SET hive.support.quoted.identifiers=None;
    USE ft_tmp;
    CREATE TABLE IF NOT EXISTS $table_all AS
    SELECT a.user_id jdpin
          ,\`(user_id)?+.+\` 
        FROM (SELECT * FROM $table_feas WHERE dt='${today}') a
        LEFT JOIN $table_index b
        ON a.user_id = b.user_id;"
    state3=$?
fi
end_time=`date +%s`
echo "step3...combine data ...time cost...: $((end_time-start_time)) s"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++step5：训练模型 ++++++++++++++++++++++++++++++++#
start_time=`date +%s`
if (( $state1==0 ))&&(( $state2==0 ))&&(( $state3==0 )); then
    for ((i=0;i<${#targets[@]};i++)); do
        /soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model.py $IDcol ${targets[i]}  $model $trainmodelflag --executor-memory 6g --execu
tor-num 20  --master yarn 1>>result/${targets[i]}.out 2>e.out
        if (( $?==0 )); then
            echo "${targets[i]} successed..."
            state4=$?
        else
            break
        fi
    done
fi
end_time=`date +%s`
echo "step4...train models ...time cost...: $((end_time-start_time)) s"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++step6：训练结果保存 +++++++++++++++++++++++++++++#
start_time=`date +%s`
if (( $state1==0 ))&&(( $state2==0 ))&&(( $state3==0 ))&&(( $state4==0 )); then
    sh agg_results.sh
    state5=$?
fi
end_time=`date +%s`
echo "step5...save data ...time cost...: $((end_time-start_time)) s"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++step7：临时性数据删除 +++++++++++++++++++++++++++#
start_time=`date +%s`
if (( $state1==0 ))&&(( $state2==0 ))&&(( $state3==0 ))&&(( $state4==0 ))&&(( $state5==0 )); then
    hive -S -e "
    USE ft_tmp;
    DROP TABLE if EXISTS $table_index;
    DROP TABLE if EXISTS $table_all;
    "
fi
end_time=`date +%s`
echo "step6...delete temp data ...time cost...: $((end_time-start_time)) s"
