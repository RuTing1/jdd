# -*- coding: utf-8 -*-

import numpy
import pandas
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

#%matplotlib qt
#设置不在交互式命令行绘图，在弹出新的窗口进行绘图

data = pandas.read_csv(
    'D:\\PDA\\6.3\\data.csv'
)

result = data.groupby(
    by=['通信品牌'], 
    as_index=False
)['号码'].agg({
    '用户数': numpy.size
})

#设置长宽分辨率
plt.figure(figsize=(30, 30), dpi=80)

#使用绝对路径获取字体的名称的方法
fontProp = font_manager.FontProperties(
    fname="C:\\Windows\\Fonts\\FZSTK.TTF"
)
#设置字体
font = {
    'family': fontProp.get_name(),
    'size': 20
}
matplotlib.rc('font', **font)

#设置为横轴和纵轴等长的饼图
#也就是圆形的饼图，而非椭圆形的饼图
plt.axis('equal')

plt.pie(
    result['用户数'], 
    labels=result['通信品牌'], 
    autopct='%.2f%%'
)


#设置突出的部分
explode = (0.1, 0.2, 0.3)
plt.axis('equal')
plt.pie(
    result['用户数'], 
    labels=result['通信品牌'], 
    autopct='%.2f%%',
    explode=explode,
    startangle=67
)

plt.show()
