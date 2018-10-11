
## 一、本版本包含的文件及其功能：
### 1.index_extract.sh:目标变量提取
### 2.useful_features_v2.sh:特征提取
         ####  创建分区特征表
         #### 往当前时间分区中插入特征数据
### 3.trainModel.py :分类模型训练脚本(含决策树算法和随机森林算法) 
                   classifier=[decisiontreeclassifier  randomforestclassifier]
### 4.agg_results.sh:将各分类模型的评分汇集到一张表中
                   flag=[inv_desire load_desire current_fin bank_desire fun_desire stock_desire]
### 5.run.sh:
        1.know your tables: 最新分区是否有数据，Y-转2，N-stop
        2.extract features: 提取建模需要的特征，为保证后续模型分析，建立分区表
        3.extract labels: 提取建模的目标变量
        4.combine data: 特征+标签组合成模型数据
        5.train models: 训练模型
        6.save data：对模型训练过程中产生的结果数据进行保存
        7.delete data：对模型训练过程中产生的临时性数据进行删除
        8.monitor model: 对模型后续表现的稳定性进行监控
        #7.evaluate.py: 模型稳定性监测脚本(主要是PSI算法) 暂无

## 二、数据
### 1.数据获取：
    1. 特征及标签提取表：
      (ft_app.table1 ft_app.table2 ft_app.table3 dmt.table4) #方便数据回溯
    2.标签映射表(agg_results.sh)：
      (ft_app.table1 ft_app.table2 ft_app.table3 dmt.table4)

### 2.数据存储：
    1.特征存储分区表:ft_tmp.useful_features_from_bz
    2.模型结果汇集分区表:ft_tmp.bank_sleep_customers_probability
    3.临时数据存储:
      ·Y值:ft_tmp.bank_sleep_customers_indexs_${today}
      ·Y+特征:ft_tmp.bank_sleep_customers_${today}
      ·各模型验证用户预测数据：ft_tmp.yhj_{classifier}_{flag}_probability1
      ·各模型全量用户预测数据：ft_tmp.yhj_{classifier}_{flag}_probability1_all

## 三、待改进项
### 1.程序的运行效率：
    1.index_extract.sh 部分运行时间较长
    2.trainModel.py 目前是串行实现，可尝试探索并行实现且想实现啥实现啥？
### 2.代码的简洁性：
    1.复用性不够：run.sh 部分基本是命令，可尝试函数实现
    2.灵活性不够：trainModel.py部分代码灵活性不够,每次运行需要寻最优参数,实际情况一次即可
