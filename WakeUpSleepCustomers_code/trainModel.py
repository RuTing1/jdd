import sys
import pandas as pd
import datetime
import pyspark.sql.types
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml import tuning as tune
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from pyspark.sql.functions import rand
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.stat import Statistics
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler,MinMaxScaler
from pyspark.sql.types import StructField, StructType, FloatType, StringType, DoubleType
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,GBTClassifier
# import xgboost
# from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator

"""
@author: DR
@date: 2018-08-13
@desc: This script is used to training classification models for the project of wakeing up sleeping customers
@question: 
@v1版本说明：
@example:spark-submit trainModel.py --executor-memory 6g --executor-num 20  --master yarn
@测试数据存储：

"""
reload(sys)
sys.setdefaultencoding('utf-8')
spark = SparkSession.builder.appName("user_cluster").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel('WARN')
# spark.sparkContext.setLogLevel('WARN')
#全局变量
seed = 12345
IDcol = 'jdpin'
target = 'is_buy'

def getData(df, n_samp_ratio=0.5):
    """
    @param sql: string 读取数据的SQL语句
    @param n_samp_ratio: int 负样本采样比例
    return: 训练集和验证集
    """
    features= [col for col in df.columns if col not in [IDcol, target]]
    #分层采样
    df_sampled = df.sampleBy(target, fractions={0: n_samp_ratio, 1: 1}, seed=seed)
    #训练集测试集划分
    train,validation = df_sampled.randomSplit([0.7, 0.3])
    return train,validation


def trainModel(classifier, paramGrid, train, validation):   
    """
    @param classifier: 实例化的分类模型
    @param paramGrid：分类模型对应的参数空间
    @param train:训练集+验证集
    @param validation: 测试集
    return：模型对应参数空间所能找到的最优模型
    """
    print('########################{}#############################'.format(str(classifier).split('_')[0]))
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('the training process start at:', start_time)
    features= [col for col in train.columns if col not in [IDcol, target]]
    #数据归一化
    featureAssembler =VectorAssembler(inputCols=features, outputCol="features1")
    mmScaler = MinMaxScaler(inputCol='features1', outputCol="Features")
    train0 = featureAssembler.transform(train)
    train0 = mmScaler.fit(train0).transform(train0)
    print('模型调参中 ...')
    evaluator=BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol=target, metricName='areaUnderROC')
    cv = tune.CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator)
    cvModel = cv.fit(train0)
    # results = cvModel.transform(validation)
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
    pipeline = Pipeline(stages=[featureAssembler, mmScaler, best_model])
    model = pipeline.fit(train)
    model.save('{}_MODEL'.format(str(classifier).split('_')[0]))
    model.write().overwrite().save('{}_MODEL'.format(str(classifier).split('_')[0]))
    # model_in = PipelineModel.load('{}_MODEL'.format(str(classifier).split('_')[0]))
    predictions=model.transform(validation)
    auc=evaluator.evaluate(predictions)
    print('the auc of the validation data of the best model is:', auc)
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('the train process end at:', end_time)



if __name__ == '__main__':   
    print('loading data ...')
    #实验数据
    data = spark.sql('select * from ft_tmp.')
    data = data.na.fill(0)
    data = data.select('*').limit(10000)
    print('get train and validation datasets ...')
    train,validation = getData(data, n_samp_ratio=0.5)
    #使用的模型及其默认参数
    classifiers = {
                        'lr':LogisticRegression(labelCol=target, featuresCol='Features'),
                        'rf':RandomForestClassifier(featuresCol='Features', labelCol=target, cacheNodeIds=True, seed=seed),
                        # 'GBDT':GBTClassifier(featuresCol='Features', labelCol=target, maxIter=6, seed= seed)
                        # 'xgboost':xgboost()
                        }
    #使用模型的参数列表
    classifier_paramGrids = {
                            'lr':ParamGridBuilder().addGrid(classifiers['lr'].regParam, [.001, .01, .1, 1.0, 10, 100]) \
                                 .addGrid(classifiers['lr'].elasticNetParam, [0, 1]) \
                                 .build(),  
                            'rf':ParamGridBuilder().addGrid(classifiers['rf'].maxDepth, [5, 8, 15, 25, 30, None]) \
                                 .addGrid(classifiers['rf'].numTrees, [120, 300, 500, 800, 1200]) \
                                 .addGrid(classifiers['rf'].minInstancesPerNode, [1, 2, 5, 10]) \
                                 .addGrid(classifiers['rf'].featureSubsetStrategy, ['auto', 'onethird', 'sqrt', 'log2']) \
                                 .build() \
                            }
                            # 'GBDT':ParamGridBuilder().addGrid(classifiers['GBDT'].maxDepth, [3, 5, 7, 9, 12, 15, 17, 25]) \
                            #                          .addGrid(classifiers['GBDT'].subsamplingRate, [0.6, 0.7, 0.8, 0.9, 1.0]) \
                            #                          .build()
                            # }
    print('training models ...')
    classifier_models = {}
    for clas in ['lr', 'rf', 'GBDT']:
        classifier_models[clas] = trainModel(classifiers[clas], classifier_paramGrids[clas], train, validation)
    spark.stop()



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
