import pandas
import matplotlib
from matplotlib import pyplot as plt

data = pandas.read_csv(
    'D:\\PDA\\6.2\\data.csv'
)

#对日期格式进行转换
data['购买日期'] = pandas.to_datetime(
    data['日期']
)


mainColor = (42/256, 87/256, 141/256, 1);

font = {
    'size': 20,
    'family': 'SimHei'
}
matplotlib.rc('font', **font)

#%matplotlib qt

plt.xlabel(
    '购买日期', 
    color=mainColor
)
plt.ylabel(
    '购买用户数', 
    color=mainColor
)
plt.tick_params(
    axis='x', 
    colors=mainColor
)
plt.tick_params(
    axis='y', 
    colors=mainColor
)

#'-'	顺滑的曲线
plt.plot(
    data['购买日期'], 
    data['购买用户数'], 
    '-', color=mainColor
)

plt.title('购买用户数')
plt.show()

#设置线条粗细
plt.plot(
    data['购买日期'], 
    data['购买用户数'], 
    '-', color=mainColor, 
    lineWidth=10
)

#'--'	虚线
plt.plot(data['购买日期'], data['购买用户数'], '--');
#'-.'	线加点
plt.plot(data['购买日期'], data['购买用户数'], '-.');
#':'	由点组成的曲线
plt.plot(data['购买日期'], data['购买用户数'], ':');
#'.'	散点图
plt.plot(data['购买日期'], data['购买用户数'], '.');
#','	像素点的散点图
plt.plot(data['购买日期'], data['购买用户数'], ',');
#'o'	大点的散点图
plt.plot(data['购买日期'], data['购买用户数'], 'o');
#'v'	下三角标记的散点图
plt.plot(data['购买日期'], data['购买用户数'], 'v');
#'^'	上上角标记的散点图
plt.plot(data['购买日期'], data['购买用户数'], '^');
#'<'	左角标记的散点图
plt.plot(data['购买日期'], data['购买用户数'], '<');
#'>'	右角标记的散点图
plt.plot(data['购买日期'], data['购买用户数'], '>');
#'1'	伞形下的标记散点图
#'2'	伞形上的标记散点图
#'3'	伞形左的标记散点图
#'4'	伞形右的标记散点图
plt.plot(data['购买日期'], data['购买用户数'], '4');
#'s'	正方形标记的散点图
plt.plot(data['购买日期'], data['购买用户数'], 's');
#'p'	五角形标记的散点图
plt.plot(data['购买日期'], data['购买用户数'], 'p');
#'*'	五角星标记的散点图
plt.plot(data['购买日期'], data['购买用户数'], '*');
#'h'	多边形标记的散点图
#'H'	hexagon2 marker
plt.plot(data['购买日期'], data['购买用户数'], 'h');
#'+'	plus marker
#'x'	x marker
#'D'	diamond marker
#'d'	thin_diamond marker
plt.plot(data['购买日期'], data['购买用户数'], 'D');
#'|'	vline marker
#'_'	hline marker
plt.plot(data['购买日期'], data['购买用户数'], '|');

