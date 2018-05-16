# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
import pyspark.ml.recommendation as rec
import pyspark.ml.feature as ft
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
import pyspark.mllib.recommendation as mllib_rec
from pyspark.sql import Window
from pyspark.sql.functions import rank
from pyspark.sql import functions as F

# spark = SparkSession.builder.appName("dataprocess").getOrCreate()
spark = SparkSession.builder.appName("dataprocess").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel('WARN')


def read_sql(sql, verbose=False):
    '''
    create dataframe from hive sql
    :param sql:sql statement
    :param verbose:if true,print some basic information about df
    :return:dataframe
    '''
    df = spark.sql(sql)
    if verbose:
        print("----------------------24----------------------")
        print("df count:{0}".format(df.count()))
        print("----------------------26----------------------")
        print("df head show:")
        #df.show(5)
        #df.printSchema()
    return df

def num_transformer(df, cols):
    '''
    用于将从hive中读取得到的dataframe进行转换。
    由于als算法接受的user和item必须为数值型，所以对dataframe中的user和item列用StringIndexer转换为数值型
    :param df:dataframe
    :param cols:待转换的列
    :return:dataframe和修改后的列名及labels
    '''
    inputCols = {}
    for col in cols:
        col_num = col+'_num'
        indexer = ft.StringIndexer(inputCol=col, outputCol=col_num)
        model = indexer.fit(df)
        df = model.transform(df)
        inputCols[col] = {}
        inputCols[col]['numCol'] = col_num
        inputCols[col]['labels'] = model.labels
    return df, inputCols

def als_model(df):
    '''
        训练模型
        :param df:dataframe
        :return:als model
        '''
    model = mllib_rec.ALS.train(df, 10, seed=0)
    return model


def recommed(user, item, rating, k, train_table ,test_table):
    '''
            生成推荐结果
            :param user:user的列名
            :param item:item的列名
            :param rating:rating的列名
            :param k:推荐的产品数量
            :return:
            '''
    if train_table:
        df = read_sql(train_table)
    else:
        df = spark.read.csv('1.csv', header=True)
        # df = df.withColumn(rating, df[rating].cast(typ.DoubleType()))
    df, inputCols = num_transformer(df, [user, item])
    df = df.where(df[rating]>=0)
    rdd = df.select(inputCols[user]['numCol'], inputCols[item]['numCol'],rating).rdd.map(lambda row: [x for x in row])
    model = als_model(rdd)
    user_rec_rdd = model.recommendProductsForUsers(k)
    schema = typ.StructType([
        typ.StructField('no', typ.IntegerType(), False),
        typ.StructField('recommendations', typ.ArrayType(elementType=typ.StructType([
            typ.StructField('user', typ.IntegerType(), False),
            typ.StructField('product', typ.IntegerType(), False),
            typ.StructField('rating', typ.FloatType(), False)
        ])), False)
    ])
    user_rec = spark.createDataFrame(user_rec_rdd, schema)
    user_rec = user_rec.select(fn.explode(user_rec.recommendations).alias('item_rating'))
    user_rec = user_rec.select('item_rating.user', 'item_rating.product', 'item_rating.rating')
    for u, e in inputCols.items():
        if u == user:
            index_to_string = ft.IndexToString(inputCol='user', outputCol=u, labels=e['labels'])
        else:
            index_to_string = ft.IndexToString(inputCol='product', outputCol=u, labels=e['labels'])
        user_rec = index_to_string.transform(user_rec)
    window = Window.partitionBy('user').orderBy(user_rec.rating.desc())
    user_rec = user_rec.withColumn('rank',rank().over(window))
    return user_rec
###计算rank
#test_df_u_c = test_df_u_c.toPandas()
#test_df_u_c['rank_score'] = test_df_u_c['rank_p']*test_df_u_c['count']   
#test_df_u_c = sqlContext.createDataFrame(test_df_u_c)
#rank_part = test_df_u_c.agg(F.sum('rank_score').alias('rank_h'),F.sum('count').alias('rank_l')).collect()
#rank = rank_part[0]['rank_h']/rank_part[0]['rank_l']
#print('the rank of the als is',round(rank,2))

if __name__ == '__main__':
   
    user_rec = recommed('jdpin', 'class_channel', 'rating2', 22,
              train_table='select * from ft_tmp.yhj_recommendation_useritemrating_train_0515',
              test_table='select 1')
    user_rec = user_rec.join(train_table.select('class_channel','channel_id').distinct(),'class_channel','left')
    res_tbl_name = "ft_tmp.yhj_als_0515"
    user_rec = user_rec.withColumnRenamed('rating','confidence')
    user_rec = user_rec.withColumnRenamed('rank','product_order')
    user_rec.select('jdpin','channel_id','class_channel', 'confidence','product_order').write.mode("overwrite").saveAsTable(res_tbl_name)

###测试recommed函数
#user = 'jdpin'
#item = 'class_channel'
#rating = 'rating2'
#k = 6
#train_df = read_sql('select * from ft_tmp.yhj_recommendation_useritemrating_train')
#test_df = read_sql('select * from ft_tmp.yhj_recommendation_useritemrating_test')


#user = 'jdpin'
#item = 'class_channel'
#rating = 'rating2'
#k = 6
#train_df = read_sql('select * from ft_tmp.yhj_recommendation_useritemrating_train')
#test_df = read_sql('select * from ft_tmp.yhj_recommendation_useritemrating_test')