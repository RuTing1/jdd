# -*- coding: utf-8 -*-

import numpy
import pandas
import matplotlib
from matplotlib import pyplot as plt

font = {
    'family' : 'SimHei'
}
matplotlib.rc('font', **font)

data = pandas.read_csv(
    'D:\\PDA\\6.4\\data.csv'
)

result = data.pivot_table(
    values='月消费（元）', 
    index='手机品牌', 
    columns='通信品牌', 
    aggfunc=numpy.sum
)

index = numpy.arange(len(result))
minColor = (42/256, 87/256, 141/256, 1/3)
midColor = (42/256, 87/256, 141/256, 2/3)
maxColor = (42/256, 87/256, 141/256, 3/3)

result = result.sort_values(
    by="神州行", ascending=False
)

#使用排列的方式，把数据堆叠放好，即为多维条形图
plt.bar(
    index, result['神州行'], 
    color = maxColor
)
plt.bar(
    index, result['动感地带'], 
    bottom=result['神州行'], 
    color = midColor
)
plt.bar(
    index, result['全球通'], 
    bottom=result['神州行']+result['动感地带'], 
    color = minColor
)
plt.xticks(index, result.index)
plt.legend(['神州行', '动感地带', '全球通'])
plt.show()
