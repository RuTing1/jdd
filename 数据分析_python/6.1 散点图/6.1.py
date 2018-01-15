# -*- coding: utf-8 -*-

import pandas
import matplotlib
import matplotlib.pyplot as plt

data = pandas.read_csv(
    'D:\\PDA\\6.1\\data.csv'
)

mainColor = (42/256, 87/256, 141/256, 1)

font = {
    'size': 20,
    'family': 'SimHei'
}
matplotlib.rc('font', **font)

#%matplotlib qt

#plt.grid(True)

#小点
plt.xlabel('广告费用', color=mainColor)
plt.ylabel('购买用户数', color=mainColor)
plt.tick_params(axis='x', colors=mainColor)
plt.tick_params(axis='y', colors=mainColor)
plt.plot(
    data['广告费用'], 
    data['购买用户数'], 
    '.', color=mainColor
)

#大点
plt.xlabel('广告费用', color=mainColor)
plt.ylabel('购买用户数', color=mainColor)
plt.tick_params(axis='x', colors=mainColor)
plt.tick_params(axis='y', colors=mainColor)
plt.plot(
    data['广告费用'], 
    data['购买用户数'], 
    "o", color=mainColor
)
