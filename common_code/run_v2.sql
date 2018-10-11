#!/bin/bash
#Created on 20181010
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

#+++++++++++++++++++++++++++++++全局变量+++++++++++++++++++++++++++++++++#

yesterday=`date +"%Y-%m-%d" -d "-1 days"`
today=`date +"%Y%m%d"`

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++需要输入的参数++++++++++++++++++++++++++++++#
#获取数据前，统一查看所涉及的表的数据是否可靠
tables=(ft_app.table1 ft_app.table2 ft_app.table3 dmt.table4)
#提取特征存储表
table_feas=ft_tmp.useful_features
#提取标签存储表：临时表
table_index=ft_tmp._indexs_${today}
table_all=ft_tmp.b_customers_${today}
#数据存储表：
#各模型验证用户预测数据：ft_tmp._{classifier}_{flag}_probability1
#各模型全量用户预测数据：ft_tmp._{classifier}_{flag}_probability1_all
#各模型最终聚合结果数据：ft_tmp.customers_probability (dt='20181010')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++step1：最新分区是否有数据++++++++++++++++++++++++++#
#对每一个表查看最新分区是否有数据
#若没有，需要在useful_features.sh和index_extract.sh文件对应位置输入有数据的分区
start_time=`date +%s`
max_partition(){
    partition="set hive.cli.print.header=flase;
    show partitions $1;"
    max_part=`hive -S -e  "$partition" | grep '^dt' | sort | tail -n 1` #取最新的分区
    echo ${max_part:3:10}
}

#建模前各表数据量统计

for ((i=0;i<${#tables[@]};i++)); do
    dates[$i]=`max_partition ${tables[i]}`
    cnt[$i]=`hive -S -e "select count(*) from ${tables[i]} where dt='${dates[i]}'"`
    echo "${tables[i]}:${dates[i]}:${cnt[i]}"
    if [[ ${cnt[i]} -lt 1000 ]]; then
        echo "${tables[i]} is empty table!!!"
        flag=1
    else
        flag=0
    fi
done

end_time=`date +%s`
echo "step1...know your tables ...time cost...: $((end_time-start_time)) s"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++step2：提取建模需要的特征++++++++++++++++++++++++++#
start_time=`date +%s`
if (( $flag == 0 )); then
    dates_new=`max_partition $table_feas`
    cnt=`hive -S -e "select count(*) from  $table_feas where dt='${today}'"`
    if [[ $dates_new != $today ]]||[[ $cnt -lt 1000 ]]; then
        sh useful_features.sh
        state1=$?
    else
        echo "useful_features table is already exists..."
        state1=0
    fi
fi
end_time=`date +%s`
echo "step2...extract features ...time cost...: $((end_time-start_time)) s"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++step3：提取建模目标变量++++++++++++++++++++++++++++#
#index_extract.sh 此文件涉及的相关表最新有数据分区需要核实
#数据存储：ft_tmp.bank_sleep_customers_indexs_${today}
start_time=`date +%s`
if (( $flag == 0 ))&&(( $state1==0 )) ; then
    sh index_extract.sh
    state2=$?
    cnt=`hive -S -e "select count(*) from  $table_index"`
    if (( $state2 != 0 ))||[[ $cnt -lt 1000 ]]; then
        state2=1
        echo "index table is empty..."
    else
        state2=0
    fi
fi 
end_time=`date +%s`
echo "step3...extract labels ...time cost...: $((end_time-start_time)) s"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++step4：特征+标签组合成模型数据+++++++++++++++++++++#
start_time=`date +%s`
if (( $flag == 0 ))&&(( $state1==0 ))&&(( $state2==0 )) ; then
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
echo "step4...combine data ...time cost...: $((end_time-start_time)) s"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++step5：训练模型 ++++++++++++++++++++++++++++++++#
start_time=`date +%s`
if (( $flag == 0 ))&&(( $state1==0 ))&&(( $state2==0 ))&&(( $state3==0 )); then
    /soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model.py jdpin inv_desire rf --executor-memory 6g --executor-num 20  --master yarn 1>>result/inv.out 2>e.out
    if (( $?==0 )); then
        echo "success 1"
        /soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model.py jdpin load_desire rf --executor-memory 6g --executor-num 20  --master yarn 1>>result/load.out 2>e.out
    fi
    if (( $?==0 )); then
        echo "success 2"
        /soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model.py jdpin current_fin rf --executor-memory 6g --executor-num 20  --master yarn 1>>result/current.out 2>e.out
    fi
    if (( $?==0 )); then
        echo "success 3"
        /soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model.py jdpin bank_desire rf --executor-memory 6g --executor-num 20  --master yarn 1>>result/bank.out 2>e.out
    fi
    if (( $?==0 )); then
        echo "success 4"
        /soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model.py jdpin fun_desire rf --executor-memory 6g --executor-num 20  --master yarn 1>>result/fund.out 2>e.out
    fi   
    if (( $?==0 )); then
        echo "success 4"
        /soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model.py jdpin stock_desire rf --executor-memory 6g --executor-num 20  --master yarn 1>>result/stock.out 2>e.out
    fi 
    state4==$?
fi
end_time=`date +%s`
echo "step5...train models ...time cost...: $((end_time-start_time)) s"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++step6：训练结果保存 +++++++++++++++++++++++++++++#
start_time=`date +%s`
if (( $flag == 0 ))&&(( $state1==0 ))&&(( $state2==0 ))&&(( $state3==0 ))&&(( $state4==0 )); then
    sh agg_results.sh
    state5==$?
fi
end_time=`date +%s`
echo "step6...save data ...time cost...: $((end_time-start_time)) s"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++step7：临时性数据删除 +++++++++++++++++++++++++++#
start_time=`date +%s`
if (( $flag == 0 ))&&(( $state1==0 ))&&(( $state2==0 ))&&(( $state3==0 ))&&(( $state4==0 ))&&(( $state5==0 )); then
    hive -S -e "
    USE ft_p;
    DROP TABLE if EXISTS $table_index;
    DROP TABLE if EXISTS $table_all;
    "
end_time=`date +%s`
echo "step7...delete temp data ...time cost...: $((end_time-start_time)) s"
