# -*- coding: utf-8 -*-
import pandas
import matplotlib
from matplotlib import pyplot as plt

font = {
    'family' : 'SimHei'
}
matplotlib.rc('font', **font)

data = pandas.read_csv(
    'D:\\PDA\\6.5\\data.csv'
)

mainColor = (42/256, 87/256, 141/256, 1)

plt.hist(data['购买用户数'], color=mainColor)
plt.show()

plt.hist(data['购买用户数'], bins=20, color=mainColor)
plt.show()

plt.hist(
    data['购买用户数'], bins=20, 
    cumulative=True, color=mainColor
)
plt.show()
