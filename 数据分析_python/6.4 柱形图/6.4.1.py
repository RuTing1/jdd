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

result = data.groupby(
    by=['手机品牌'], 
    as_index=False
)['月消费（元）'].agg({
    '月消费': numpy.sum
})

#竖向柱形图
index = numpy.arange(
    result.月消费.size
)
plt.bar(index, result['月消费'])
plt.show()

#优化点1、配置颜色
mainColor = (42/256, 87/256, 141/256, 1)
plt.bar(
    index, result['月消费'], 
    color=mainColor
)
plt.show()

#优化点2、配置X轴刻度
plt.bar(
    index, result['月消费'], 
    color=mainColor
)
plt.xticks(index, result.手机品牌)
plt.show()

#优化点3、对数据排序后再绘图
sgb = result.sort_values(
    by="月消费", 
    ascending=False
)
plt.bar(
    index, sgb.月消费, 
    color=mainColor
)
plt.xticks(index, sgb.手机品牌)
plt.show()


#横向柱形图
plt.barh(
    index, sgb.月消费, 
    color=mainColor
)
plt.yticks(index, sgb.手机品牌)
plt.show()
