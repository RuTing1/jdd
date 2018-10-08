# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
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
@example:spark-submit train_model.py jdpin inv_desire rf --executor-memory 6g --executor-num 20  --master yarn >
@测试数据存储：
投资意愿:  spark-submit train_model.py jdpin inv_desire rf --executor-memory 6g --executor-num 20  --master yarn
借贷意愿： spark-submit train_model.py jdpin load_desire rf --executor-memory 6g --executor-num 20  --master yarn
活期产品： spark-submit train_model.py jdpin current_fin rf --executor-memory 6g --executor-num 20  --master yarn
银行理财： spark-submit train_model.py jdpin is_buy rf --executor-memory 6g --executor-num 20  --master yarn

spark-submit train_model.py jdpin inv_desire dt --executor-memory 6g --executor-num 20  --master yarn 1>inv.out 2>e.out
spark-submit train_model.py jdpin load_desire dt --executor-memory 6g --executor-num 20  --master yarn 1>load.out 2>e.out
spark-submit train_model.py jdpin current_fin dt --executor-memory 6g --executor-num 20  --master yarn 1>current.out 2>e.out
spark-submit train_model.py jdpin is_buy dt --executor-memory 6g --executor-num 20  --master yarn 1>buy.out 2>e.out
spark-submit train_model.py jdpin is_fun dt --executor-memory 6g --executor-num 20  --master yarn 1>fun.out 2>e.out

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
        features= [col for col in train.columns if col not in [IDcol, target, 'current_fin']]
    print('模型特征筛选中 ...')
    df_feasi = trainModelTest(train, features, classifier)
    df_feasi = df_feasi.filter((df_feasi.importance>0.001)&(df_feasi.importance<0.75))
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
    model = PipelineModel.load('{}_{}_MODEL'.format(str(classifier).split('_')[0], flag))
    prediction = model.transform(data)
    prediction = prediction.select(IDcol, 'label', "probability")
    prediction = prediction.rdd.map(extract).toDF()
    for col1,col2 in zip(prediction.columns,[IDcol, 'label','probability0','probability1']):
        prediction = prediction.withColumnRenamed(col1,col2)
    # prediction = prediction.withColumnRenamed('label',flag)
    prediction = prediction.withColumn('probability1',fn.round(prediction.probability1,3))
    prediction.write.mode("overwrite").saveAsTable('ft_tmp.yhj_{}_{}_probability1_all'.format(str(classifier).split('_')[0], flag))



if __name__ == '__main__':   
    try:
        IDcol = sys.argv[1]
        target = sys.argv[2]
        # dataname = sys.argv[3]
        classifier_name = sys.argv[3]
    except KeyboardInterrupt:
        pass

    print('loading data ...')
    #实验数据
    spark = SparkSession.builder.appName("user_cluster").enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    data = spark.sql("select * from ft_tmp.sleep_user_with_info_cols_3mall_v2")
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
                        'lr':LogisticRegression(labelCol=target, featuresCol='Features'),
                        'rf':RandomForestClassifier(featuresCol='Features', labelCol=target, cacheNodeIds=True, seed=seed),
                        'dt':DecisionTreeClassifier(maxDepth=30, minInstancesPerNode=50,labelCol="label",impurity="gini",seed=seed),
                        # 'GBDT':GBTClassifier(featuresCol='Features', labelCol=target, maxIter=6, seed= seed)
                        # 'xgboost':xgboost()
                        }
    #使用模型的参数列表
    classifier_paramGrids = {
                            'lr':ParamGridBuilder().addGrid(classifiers['lr'].regParam, [.001, .01, .1, 1.0]) \
                                 .addGrid(classifiers['lr'].elasticNetParam, [0, 1]) \
                                 .build(),  
                            'rf':ParamGridBuilder().addGrid(classifiers['rf'].maxDepth, [15, 25, 30, 50, None]) \
                                 .addGrid(classifiers['rf'].numTrees, [120, 300]) \
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


####################临时跑全量的效果###############
#单个特征全量用户数据预测
def predict_all_user(data, target):
    data = data.withColumnRenamed(target,'label')
    if target =='current_fin':
        features= [col for col in data.columns if col not in [IDcol, 'label', 'inv_desire', 'is_buy', 'is_fun']]
    elif target == 'is_buy':
        features= [col for col in data.columns if col not in [IDcol, 'label', 'inv_desire', 'current_fin', 'is_fun']]
    else:
        features= [col for col in data.columns if col not in [IDcol, 'label', 'current_fin']]
    # model.write().overwrite().save('{}_{}_MODEL'.format(str(classifier).split('_')[0], flag))
    model_in = PipelineModel.load('DecisionTreeClassifier_{}_MODEL'.format(target))
    prediction = model_in.transform(data)
    prediction = prediction.rdd.map(extract).toDF()
    for col1,col2 in zip(prediction.columns,[IDcol, target,'probability0','probability1']):
        prediction = prediction.withColumnRenamed(col1,col2)  
    prediction = prediction.drop('probability0')
    prediction.write.mode("overwrite").saveAsTable('ft_tmp.yhj_DecisionTreeClassifier_{}_probability1_all'.format(target))

IDcol = 'jdpin'
spark = SparkSession.builder.appName("user_cluster").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel('WARN')
data = spark.sql("select * from ft_tmp.sleep_user_with_info_cols_3mall_v2")
data = data.na.fill(0)
targets =['load_desire', 'is_buy', 'is_fun']
for target in targets:
    print(target)
    predict_all_user(data, target)


ft_tmp.yhj_DecisionTreeClassifier_load_desire_probability1_all



######多个特征全量用户数据预测
def predict_all_user(data):
    targets =['inv_desire', 'load_desire', 'is_buy', 'is_fun','current_fin']
    prediction = {}
    for target in targets:
        print(target)
        data0 = data.withColumnRenamed(target,'label')
        if target =='current_fin':
            features= [col for col in data0.columns if col not in [IDcol, 'label', 'inv_desire', 'is_buy', 'is_fun']]
        elif target == 'is_buy':
            features= [col for col in data0.columns if col not in [IDcol, 'label', 'inv_desire', 'current_fin', 'is_fun']]
        else:
            features= [col for col in data0.columns if col not in [IDcol, 'label', 'current_fin']]
        # model.write().overwrite().save('{}_{}_MODEL'.format(str(classifier).split('_')[0], flag))
        model_in = PipelineModel.load('DecisionTreeClassifier_{}_MODEL'.format(target))
        prediction{target} = model_in.transform(data0)
        prediction{target} = prediction{target}.rdd.map(extract).toDF()
        for col1,col2 in zip(prediction{target}.columns,[IDcol, target,'probability0','probability1']):
            prediction{target} = prediction{target}.withColumnRenamed(col1,col2)
        prediction{target} = prediction{target}.drop('probability0') 
        prediction{target} = prediction{target}.withColumnRenamed('probability1',target)









data0 = data.withColumnRenamed(target,'label')
if target =='current_fin':
    features= [col for col in data0.columns if col not in [IDcol, 'label', 'inv_desire', 'is_buy', 'is_fun']]
elif target == 'is_buy':
    features= [col for col in data0.columns if col not in [IDcol, 'label', 'inv_desire', 'current_fin', 'is_fun']]
else:
    features= [col for col in data0.columns if col not in [IDcol, 'label', 'current_fin']]
model = PipelineModel.load('DecisionTreeClassifier_{}_MODEL'.format(target))
prediction = model.transform(data0)
prediction = prediction.rdd.map(extract).toDF()
for col1,col2 in zip(prediction.columns,[IDcol, target,'probability0','probability1']):
    prediction = prediction.withColumnRenamed(col1,col2)   
prediction.write.mode("overwrite").saveAsTable('ft_tmp.yhj_DecisionTreeClassifier_{}_probability1_all'.format(target))




trainModel(IDcol, target, classifier, paramGrid, train, validation)

classifier = DecisionTreeClassifier(maxDepth=30, minInstancesPerNode=50,labelCol="label",impurity="gini",seed=seed)
paramGrid = ParamGridBuilder().addGrid(classifier.maxDepth, [15, None]) \
                              .addGrid(classifier.minInstancesPerNode,[10, 50]) \
                              .build()




ft_tmp.yhj_DecisionTreeClassifier_inv_desire_probability1
ft_tmp.yhj_DecisionTreeClassifier_load_desire_probability1
ft_tmp.yhj_DecisionTreeClassifier_is_buy_probability1


#数据验证
SELECT score, COUNT(is_fun) AS total, SUM(is_fun) AS buy_cnt
FROM
    (
    SELECT CASE WHEN probability1 <0.05 AND probability1>=0 THEN '[0,0.05)'
                WHEN probability1 <0.1 AND probability1>=0.05 THEN '[0.05,0.1)'
                WHEN probability1 <0.15 AND probability1>=0.1 THEN '[0.1,0.15)'
                WHEN probability1 <0.2 AND probability1>=0.15 THEN '[0.15,0.2)'
                WHEN probability1 <0.25 AND probability1>=0.2 THEN '[0.2,0.25)'
                WHEN probability1 <0.3 AND probability1>=0.25 THEN '[0.25,0.3)'
                WHEN probability1 <0.35 AND probability1>=0.3 THEN '[0.3,0.35)'
                WHEN probability1 <0.4 AND probability1>=0.35 THEN '[0.35,0.4)'
                WHEN probability1 <0.45 AND probability1>=0.4 THEN '[0.4,0.45)'
                WHEN probability1 <=0.5 AND probability1>=0.45 THEN '[0.45,0.5)'
                WHEN probability1 <=0.55 AND probability1>=0.5 THEN '[0.5,0.55)'
                WHEN probability1 <=0.6 AND probability1>=0.55 THEN '[0.55,0.6)'
                WHEN probability1 <=0.65 AND probability1>=0.6 THEN '[0.6,0.65)'
                WHEN probability1 <=0.7 AND probability1>=0.65 THEN '[0.65,0.7)'
                WHEN probability1 <=0.75 AND probability1>=0.7 THEN '[0.7,0.75)'
                WHEN probability1 <=0.8 AND probability1>=0.75 THEN '[0.75,0.8)'
                WHEN probability1 <=0.85 AND probability1>=0.8 THEN '[0.8,0.85)'
                WHEN probability1 <=0.9 AND probability1>=0.85 THEN '[0.85,0.9)'
                WHEN probability1 <=0.95 AND probability1>=0.9 THEN '[0.9,0.95)'
                WHEN probability1 <=1.0 AND probability1>=0.95 THEN '[0.95,1.0]'
          ELSE NULL END score,
          is_fun
    FROM ft_tmp.yhj_DecisionTreeClassifier_is_fun_probability1_all
    )c
    GROUP BY score;











# lr = LogisticRegression(labelCol=target, featuresCol='Features')
# lr_param = ParamGridBuilder().addGrid(lr.regParam, [.001, .01, .1, 1.0, 10, 100]) \
#                              .addGrid(lr.elasticNetParam, [0, 1]) \
#                              .build()

# best_model = trainModel(lr, lr_param, train, validation)

    
# def LRpipeline(train, validation):
#     print('*************LogisticRegression model **************************')
#     features= [col for col in train.columns if col not in [IDcol, target]]
#     featureAssembler =VectorAssembler(inputCols=features, outputCol="Features")
#     # 设置默认分类器
#     clf= LogisticRegression(labelCol=target, featuresCol='Features')
#     #LR 参数网格
#     paramGrid = ParamGridBuilder().addGrid(clf.regParam, [0.01,0.1]).build()
#     # modeling类调用方式
#     print('模型调参中 ...')
#     bestModel, best_epm = module.modeling()._fit(train, clf, paramGrid, 2)
#     print('最优pipeline模型训练中 ...')
#     pipeline=Pipeline(stages=[featureAssembler, bestModel])
#     model = pipeline.fit(train)
#     print('评估模型中...')
#     predictions=model.transform(validation)
#     # Evaluate model on test instances and compute test error
#     evaluator=BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol=target, metricName='areaUnderROC')
#     auc=evaluator.evaluate(predictions)
#     print('验证集logistic模型AUC为:'+str(auc))
#     return bestModel, auc



#stacking
## reference from: https://github.com/amazefan/Spark-final-exam/blob/61c87890b8ac7cd356429f83494e1d5baf68813c/pipeline.py
# import pyspark.sql.functions as fn
# vectorToColumn = fn.udf(lambda vec: vec[1].item(), DoubleType())


# class Stacking(object):
#     def __init__(self,models,train,validation):
#         self.models = models
#         self.train = train
#         self.validation = validation
    
#     @staticmethod    
#     def basic(train,validation,model):
#         colname = str(model)
#         features= train.columns[1:-1]
#         featureAssembler =VectorAssembler(inputCols=features, outputCol="Features")
#         pipeline=Pipeline(stages=[featureAssembler,model])
#         model = pipeline.fit(train)
#         print('评估模型中...')
#         predicted1=model.transform(train)
#         auc_calculator = model.transform(validation)
#         evaluator=BinaryClassificationEvaluator(rawPredictionCol='probability',labelCol='lable',metricName='areaUnderROC')
#         auc=evaluator.evaluate(auc_calculator)
#         print('验证集模型AUC为:'+str(auc))
#         predicted1 = predicted1.withColumn(colname,vectorToColumn(predicted1.probability))
#         predicted1 = predicted1.drop('Features','probability','prediction','rawPrediction')
#         predicted2 = auc_calculator.withColumn(colname,vectorToColumn(auc_calculator.probability))
#         predicted2 = predicted2.drop('Features','probability','prediction','rawPrediction') 
#         return predicted1.select('id',colname),predicted2.select('id',colname)

#     @staticmethod
#     def stacking(model_probs):
#         basic_DF = model_probs[0]
#         for DF in model_probs[1:]:
#             basic_DF = basic_DF.join(DF,basic_DF.id == DF.id).drop(DF.id)
#         return basic_DF
    
#     @staticmethod
#     def logistic_pipeline(train,validation):
#         features= train.columns[2:]
#         featureAssembler =VectorAssembler(inputCols=features, outputCol="Features")
#         LR= LogisticRegression(labelCol='lable',featuresCol='Features')
#         LR_pipeline=Pipeline(stages=[featureAssembler,LR])
#         LR_model = LR_pipeline.fit(train)
#         print('评估模型中...')
#         predicted=LR_model.transform(validation)
#         evaluator=BinaryClassificationEvaluator(rawPredictionCol='probability',labelCol='lable',metricName='areaUnderROC')
#         auc=evaluator.evaluate(predicted)
#         print('验证集Stacking模型AUC为:'+str(auc))
#         return auc,predicted

#     def trainStacking(self):
#         train_probs = [self.train.select('id','lable')]
#         validation_probs = [self.validation.select('id','lable')]
#         for model in self.models:
#             probs_train ,probs_validation= self.basic(train = self.train ,validation = self.validation ,model = model)
#             train_probs.append(probs_train)
#             validation_probs.append(probs_validation)
#         self.DF_train = self.stacking(train_probs)
#         self.DF_validation = self.stacking(validation_probs)
#         self.auc, self.predicted = self.logistic_pipeline(self.DF_train,self.DF_validation)
#         return self.auc
 

# train,validation = getdata('all1.csv')
# model1 = RandomForestClassifier(labelCol='lable',featuresCol='Features',maxDepth=10,numTrees=500)
# model2 = GBTClassifier(labelCol='lable',featuresCol='Features',maxIter=6)
# model3 = RandomForestClassifier(labelCol='lable',featuresCol='Features',maxDepth=20,numTrees=250)
# model4 = RandomForestClassifier(labelCol='lable',featuresCol='Features',maxDepth=10,numTrees=100)
# models = [model1,model2,model3,model4]
# ensemble = Stacking(models,train,validation)
# ensemble.trainStacking()