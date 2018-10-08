# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import time
import datetime
import pyspark.sql.types
from pyspark.sql import functions as fn
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import SparkSession
from pyspark.ml import tuning as tune
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from pyspark.sql.functions import rand
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.stat import Statistics
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler,MinMaxScaler
from pyspark.sql.types import StructField, StructType, FloatType, StringType, DoubleType,IntegerType
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,GBTClassifier,DecisionTreeClassifier
# import xgboost
# from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from data_source import sleep_user

"""
@author: DR
@date: 2018-08-13
@desc: This script is used to training classification models for the project of wakeing up sleeping customers
@question: 
@v1版本说明：
 目前可以选择的模型：lr rf
@example:spark-submit train_model.py  inv_desire rf --executor-memory 6g --executor-num 20  --master yarn >
@测试数据存储：
投资意愿:  spark-submit train_model.py  inv_desire rf --executor-memory 6g --executor-num 20  --master yarn
借贷意愿： spark-submit train_model.py  load_desire rf --executor-memory 6g --executor-num 20  --master yarn
活期产品： spark-submit train_model.py  current_fin rf --executor-memory 6g --executor-num 20  --master yarn
银行理财： spark-submit train_model.py  is_buy rf --executor-memory 6g --executor-num 20  --master yarn


"""
reload(sys)
sys.setdefaultencoding('utf-8')

#全局变量
seed = 12345

def extract(row):
    """
    split vector into columns
    @example:df.rdd.map(extract).toDF()
    """
    return (row[IDcol],)+(row['label'],) + tuple(row.probability.toArray().tolist())

def getData(df, IDcol, target, n_samp_ratio=[0.5,0.5]):
    """
    @param sql: string 读取数据的SQL语句
    @param n_samp_ratio: int 负样本采样比例
    return: 训练集和验证集
    """
    # features= [col for col in df.columns if col not in [IDcol, target]]
    #分层采样
    df_sample = df.sampleBy(target, fractions={0: n_samp_ratio[0], 1: n_samp_ratio[1]}, seed=seed)
    a = df_sample.groupBy(target).count()
    b = a.sort(target).collect()
    good = b[1][1]
    bad = b[0][1]
    ratio = (good*1.0)/(good+bad)
    print('{sampleBy dataset: user number}:', good+bad)
    print('{sampleBy dataset: good}:',good)
    print('{sampleBy dataset: bad }:',bad)
    print('{sampleBy dataset: good ratio}:', ratio)
    df_sample = df_sample.na.fill(0)
    # feas_type = {}
    # df_type = df_sample.dtypes #各特征类型
    # for i in range(0,len(df_sample.columns)):
    #     feas_type[df_type[i][0]] = df_type[i][1]
    # strType = {k:v for k,v in feas_type.items() if v=='string' and k not in [IDcol]}
    # for col in strType:
    # stringIndexer = StringIndexer(inputCol='jdmall_user_p0011', outputCol="indexed")
    # model = stringIndexer.fit(df_sample)
    # df_sample = model.transform(df_sample)
    # df_sample = df_sample.withColumnRenamed('indexed','jdmall_user_p0011')
    # df_sample = df_sample.na.fill(-1)
    df_sample = df_sample.drop('jdmall_user_p0011').drop('risk_prior')
    df_sample.cache()
    #训练集测试集划分
    train,validation = df_sample.randomSplit([0.7, 0.3])
    print('{train dataset: user number}:',train.count())
    print('{validation dataset: user number}:',validation.count())
    return train,validation

 
 def trainModel(IDcol, target, flag, classifier, paramGrid, train, validation):
    """
    @param classifier: 实例化的分类模型
    @param paramGrid：分类模型对应的参数空间
    @param train:训练集+验证集
    @param validation: 测试集
    return：模型对应参数空间所能找到的最优模型
    """
    print('########################{}#############################'.format(str(classifier).split('_')[0]))
    # start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time = datetime.datetime.now()
    if flag =='current_fin':
        features= [col for col in train.columns if col not in [IDcol, target, 'inv_desire', 'is_buy', 'is_fun']]
    elif flag == 'is_buy':
        features= [col for col in train.columns if col not in [IDcol, target, 'inv_desire', 'current_fin', 'is_fun']]
    else:
        features= [col for col in train.columns if col not in [IDcol, target,'inv_desire', 'current_fin']]
    print('模型特征筛选中 ...')
    df_feasi = trainModelTest(train, features, classifier)
    df_feasi = df_feasi.filter((df_feasi.importance>0.001)&(df_feasi.importance<0.8))
    df_feasi = df_feasi.select('features').rdd.collect()
    features_retain = []
    for i in range(len(df_feasi)):
        features_retain.append(str(df_feasi[i]['features']))
    df_feasi_new = trainModelTest(train, features_retain, classifier)
    df_feasi_new = df_feasi_new.select('features').rdd.collect()
    features_retain = []
    for i in range(len(df_feasi_new)):
        features_retain.append(str(df_feasi_new[i]['features']))
    print('模型调参中 ...')
    evaluator=BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol=target, metricName='areaUnderROC')
    cv = tune.CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator)
    featureAssembler = VectorAssembler(inputCols=features_retain, outputCol="features")
    train = featureAssembler.transform(train)
    cvModel = cv.fit(train)
    # 提取模型最佳参数
    results = [([{key.name: paramValue}
           for key, paramValue in zip(params.keys(), params.values())], metric)
           for params, metric in zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics)]
    best_result = sorted(results, key=lambda el: el[1],reverse=True)[0]
    print('the best model is like this:', best_result)
    best_param = {}
    for param in best_result[0]:
        best_param =  dict( best_param, **param)
    # best_model = classifier.fit(train, best_param)
    best_model = classifier.setParams(**best_param)
    pipeline = Pipeline(stages=[featureAssembler, best_model])
    train = train.drop('features')
    model = pipeline.fit(train)
    # model.save('{}_MODEL'.format(str(classifier).split('_')[0]))
    model.write().overwrite().save('{}_{}_MODEL'.format(str(classifier).split('_')[0], flag))
    # model_in = PipelineModel.load('{}_MODEL'.format(str(classifier).split('_')[0]))
    #'DecisionTreeClassifier'
    prediction = model.transform(validation)
    auc=evaluator.evaluate(prediction)
    print('the auc of {} in validation dataset is:'.format(str(classifier).split('_')[0]), auc)
    prediction = prediction.select(IDcol, target, "probability")
    # prediction = prediction.withColumnRenamed(target,'label')
    prediction = prediction.rdd.map(extract).toDF()
    for col1,col2 in zip(prediction.columns,[IDcol, target,'probability0','probability1']):
        prediction = prediction.withColumnRenamed(col1,col2)
    prediction = prediction.withColumn('probability1',fn.round(prediction.probability1,3))
    prediction.write.mode("overwrite").saveAsTable('ft_tmp.yhj_{}_{}_probability1'.format(str(classifier).split('_')[0], flag))
    end_time = datetime.datetime.now()
    print('the time cost for model training is:', (end_time-start_time).seconds/60, 'min')

    
def trainModelTest(train, features, classifier):
    featureAssembler =VectorAssembler(inputCols=features, outputCol="features")
    train = featureAssembler.transform(train)
    model = classifier.fit(train)
    fi = model.featureImportances
    fi_list = []
    for i in range(0, len(features)):
        fi_list.append([features[i],np.float64(fi[i]).item()])
    df_feasi = spark.createDataFrame(fi_list, ["features", "importance"])
    df_feasi = df_feasi.sort(df_feasi.importance.desc())
    df_feasi = df_feasi.withColumn('importance',fn.round(df_feasi.importance,3))
    print('features importance are:')
    df_feasi.show()
    train = train.drop('features')
    return df_feasi

def results(data, classifier, flag):
    print('start saving prediction results for all users...')
    start_time = datetime.datetime.now()
    model = PipelineModel.load('{}_{}_MODEL'.format(str(classifier).split('_')[0], flag))
    prediction = model.transform(data)
    prediction = prediction.select(IDcol, 'label', "probability")
    prediction = prediction.rdd.map(extract).toDF()
    for col1,col2 in zip(prediction.columns,[IDcol, 'label','probability0','probability1']):
        prediction = prediction.withColumnRenamed(col1,col2)
    prediction = prediction.withColumn('probability1',fn.round(prediction.probability1,3))
    prediction.write.mode("overwrite").saveAsTable('ft_tmp.yhj_{}_{}_probability1_all'.format(str(classifier).split('_')[0], flag))
    end_time = datetime.datetime.now()
    print('the time cost for saving data is:', (end_time-start_time).seconds/60, 'min')

if __name__ == '__main__':
    try:
        IDcol = sys.argv[1]
        target = sys.argv[2]
        # dataname = sys.argv[3]
        classifier_name = sys.argv[3]
    except KeyboardInterrupt:
        pass

    today=time.strftime('%Y%m%d')
    #today='20180926'   
    print('#########################{}########################'.format(today))
    print('loading data ...')
    #实验数据
    spark = SparkSession.builder.appName("user_cluster").enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    data = spark.sql("select * from ft_tmp.bank_sleep_customers_{}".format(today))
    data = data.drop('dt')
    data = data.withColumn('current_fin',data['current_fin'].cast(IntegerType()))
    data = data.na.fill(0)
    a = data.groupBy(target).count()
    b = a.sort(target).collect()
    good = b[1][1]
    bad = b[0][1]
    ratio = (good*1.0)/(good+bad)
    print('{original dataset: user number}:',data.count)
    print('{original dataset: good}:',good)
    print('{original dataset: bad }:',bad)
    print('{original dataset: good ratio}:', ratio)
    if good<500000:
        good_ratio = 1
        bad_ratio = round(500000.0/bad,5)
    else:
        good_ratio = round(500000.0/good,5)
        bad_ratio = round(500000.0/bad,5)

    n_samp_ratio = [bad_ratio, good_ratio]
    print('get train and validation datasets ...')
    data = data.withColumnRenamed(target,'label')
    train,validation = getData(data, IDcol, 'label', n_samp_ratio=n_samp_ratio)
    #使用的模型及其默认参数
    classifiers = {
                        'lr':LogisticRegression(labelCol=target, featuresCol='features'),
                        'rf':RandomForestClassifier(featuresCol='features', labelCol='label', cacheNodeIds=True, seed=seed),
                        'dt':DecisionTreeClassifier(maxDepth=30, minInstancesPerNode=50,labelCol="label",impurity="gini",seed=seed),
                        # 'GBDT':GBTClassifier(featuresCol='Features', labelCol=target, maxIter=6, seed= seed)
                        # 'xgboost':xgboost()
                        }
    #使用模型的参数列表
    classifier_paramGrids = {
                            'lr':ParamGridBuilder().addGrid(classifiers['lr'].regParam, [.001, .01, .1, 1.0]) \
                                 .addGrid(classifiers['lr'].elasticNetParam, [0, 1]) \
                                 .build(),
                            'rf':ParamGridBuilder().addGrid(classifiers['rf'].maxDepth, [8,10,15]) \
                                 .addGrid(classifiers['rf'].numTrees, [20,50]) \
                                 .addGrid(classifiers['rf'].minInstancesPerNode, [5, 10]) \
                                 .build(),
                            'dt':ParamGridBuilder().addGrid(classifiers['dt'].maxDepth, [15, 25, 30, None]) \
                                .addGrid(classifiers['dt'].minInstancesPerNode,[5, 10, 50])
                                .build()
                            }
                            # 'GBDT':ParamGridBuilder().addGrid(classifiers['GBDT'].maxDepth, [3, 5, 7, 9, 12, 15, 17, 25]) \
                            #                          .addGrid(classifiers['GBDT'].subsamplingRate, [0.6, 0.7, 0.8, 0.9, 1.0]) \
                            #                          .build()
                            # }
    print('training models ...')
    # classifier_models = {}
    # for clas in ['lr', 'rf', 'GBDT']:
    #     classifier_models[clas] = trainModel(classifiers[clas], classifier_paramGrids[clas], train, validation)
    trainModel(IDcol, 'label', target, classifiers[classifier_name], classifier_paramGrids[classifier_name], train, validation)
    results(data, classifiers[classifier_name], target)
    print('#########################end########################')

    
       


   

