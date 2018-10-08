#!/bin/bash

#dataExtract:ft_tmp.bank_sleep_customers_v2
#dataExtract:ft_tmp.sleep_user_with_info_cols_3mall_v2
#trainModel:
#agg_results:ft_tmp.bank_sleep_customers_probability

start_time=`date +%s`
sh data_extract.sh 
end_time1=`date +%s`
echo "data ready...time cost: $((end_time-start_time)) s"
/soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model_v1.py  inv_desire dt --executor-memory 6g --executor-num 20  --master yarn 1>inv.out 2>e.out
end_time2=`date +%s`
echo "model for investment desire is done...time cost: $((end_time2-end_time1)) s"
/soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model_v1.py  load_desire dt --executor-memory 6g --executor-num 20  --master yarn 1>load.out 2>e.out
end_time3=`date +%s`
echo "model for load desire is done...time cost: $((end_time3-end_time2)) s"
/soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model_v1.py  current_fin dt --executor-memory 6g --executor-num 20  --master yarn 1>current.out 2>e.out
end_time4=`date +%s`
echo "model for current product is done...time cost: $((end_time4-end_time3)) s"
/soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model_v1.py  is_buy dt --executor-memory 6g --executor-num 20  --master yarn 1>buy.out 2>e.out
end_time5=`date +%s`
echo "model for bank product is done...time cost: $((end_time5-end_time4)) s"
/soft/client/spark-2.1.1-bin-2.6.0/bin/spark-submit train_model_v1.py  is_fun dt --executor-memory 6g --executor-num 20  --master yarn 1>fun.out 2>e.out
end_time6=`date +%s`
echo "model for financial product is done...time cost: $((end_time6-end_time5)) s"
sh agg_results.sh
end_time7=`date +%s`
echo "aggregate results is done... time cost: $((end_time7-end_time6)) s"
end_time=`date +%s`
echo "time cost for recalling sleeping customer program is $((end_time-start_time)) s"
