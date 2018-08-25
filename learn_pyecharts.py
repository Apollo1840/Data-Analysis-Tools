# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 18:00:34 2018

@author: zouco
"""

import pandas as pd
from pyecharts import Bar,Pie,Line,Funnel
from pyecharts import Gauge,Funnel,Boxplot,Scatter3D,Bar3D
from pyecharts import Page,Overlap
from pyecharts import Style
from pyecharts import create_default_environment as pch_env

df = pd.read_excel('PUBG.xlsx')

def RGB(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def HEX(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))



# ----------------------------------------------------------------------------
# scatter
from pyecharts import Scatter
v1 = [10, 20, 30, 40, 50, 60]
v2 = [10, 20, 30, 40, 50, 60]

scatter = Scatter("散点图示例")
scatter.add("A", v1, v2)
scatter.add("B", v1[::-1], v2, is_visualmap=True,
            visual_type='size', visual_range_size=[20, 80])
scatter.render()

# babble
data = [
        [28604, 77, 17096869],
        [31163, 77.4, 27662440],
        [1516, 68, 1154605773],
        [13670, 74.7, 10582082],
        [28599, 75, 4986705],
        [29476, 77.1, 56943299],
        [31476, 75.4, 78958237],
        [28666, 78.1, 254830],
        [1777, 57.7, 870601776],
        [29550, 79.1, 122249285],
        [2076, 67.9, 20194354],
        [12087, 72, 42972254],
        [24021, 75.4, 3397534],
        [43296, 76.8, 4240375],
        [10088, 70.8, 38195258],
        [19349, 69.6, 147568552],
        [10670, 67.3, 53994605],
        [26424, 75.7, 57110117],
        [37062, 75.4, 252847810]
    ]

x_lst = [v[0] for v in data]
y_lst = [v[1] for v in data]
extra_data = [v[2] for v in data]
sc = Scatter()
sc.add("scatter", x_lst, y_lst, extra_data=extra_data, is_visualmap=True,
        visual_dimension=2, visual_orient='horizontal',
        visual_type='size', visual_range=[254830, 1154605773],
        visual_text_color='#000')
sc.render()

# effect scatter
from pyecharts import EffectScatter

v1 = [10, 20, 30, 40, 50, 60]
v2 = [25, 20, 15, 10, 60, 33]
es = EffectScatter("动态散点图示例")
es.add("effectScatter", v1, v2)
es.render()

# bubble chart
import pandas as pd
dt = pd.DataFrame({
        'x':[-1,0,0.5,0,1]+[-3,-2,-2,-1,-1,-1,  0,0,0,   1,1,1,1,2,2,3],
        'y':[0,1,0,-1,0]+[0 , 3,-3,5, 4,-5,   6,8,-7,  3,-4,5,-6,5,-3,0],
        'z':[3,2.8,3,3,2.8]+[0.5,1,1,1,0.5,1,         1,1,1,   1,0.5,1,1,0.5,1,0.5]
        })

vx=[i+5 for i in dt.x]
vy=[i+10 for i in dt.y]
vz=dt.z
    
bubble = Scatter()
bubble.add('', vx, vz, extra_data=vy,is_visualmap=True, visual_type='size', visual_range=[0,20])
bubble.render()

# ----------------------------------------------------------------------------
# 3D scatter
import random
data = [[random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)] for _ in range(80)]

scatter3D =  Scatter3D("3D scatter example")
scatter3D.add("data", data)
scatter3D.render('3D_scatter.html')


# advance
scatter3D =  Scatter3D("3D scatter example")

range_color = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
             '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
scatter3D.add("data", data, is_visualmap=True, visual_range_color=range_color, is_grid3d_rotate=True)
scatter3D.render('3D_scatter.html')



# ----------------------------------------------------------------------------
# line plot
from pyecharts import Line

lines = Line()
lines.add('level', df.ID, df.Level)
lines.add('time', df.ID, df.Time, line_width=3, line_color=RGB(255, 0, 0), line_type='dashed',line_opacity=0.5)
lines.render('line.html')

# error : legend doesnot update
line=Line()
line.add(" ", [5,2,3,4,1],[1,2,3,4,5], label_color=["#ff78ff","#ff78ff"], label_style='dashed', xaxis_rotate=-20)
line.render()

# smooth curve
attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
v1 = [5, 20, 36, 10, 10, 100]
line.add("商家", attr, v1, is_smooth=True)
line.render()

# stage
line = Line("折线图-阶梯图示例")
line.add("商家A", attr, v1, is_step='end')  #'start''middle'
line.render()

# area
attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
v1 = [5, 20, 36, 10, 75, 90]
v2 = [10, 25, 8, 60, 20, 80]

style = Style()
l_style = style.add(
        is_fill=True,
        area_opacity = 0.3,
        is_smooth=True
        )

line = Line("折线图-面积图示例")    
line.add("商家A", attr, v1, symbol=None, **l_style)
line.add("商家B", attr, v2, symbol=None, area_color='#000', **l_style)
line.render('area.html')

# 3D line
import math
_data = []
for t in range(0, 25000):
    _t = t / 1000
    x = (1 + 0.25 * math.cos(75 * _t)) * math.cos(_t)
    y = (1 + 0.25 * math.cos(75 * _t)) * math.sin(_t)
    z = _t + 2.0 * math.sin(75 * _t)
    _data.append([x, y, z])

from pyecharts import Line3D 

line3d = Line3D("3D 折线图示例")
line3d.add("", _data, is_visualmap=True, visual_range=[0, 30])
line3d.render()

# ----------------------------------------------------------------------------
# Bar chart

# supper simple example
bar =  Bar('PUBG')
bar.add('Time', df.Name, df.Time,bar_category_gap='70%')
bar.render()

# version 2 (another type & more utils)
bar =  Bar('PUBG')
bar.add('Time', df.Name, df.Time, is_more_utils=True, is_datazoom_show=True, datazoom_type='both')  
# is_more_utils means more functions beside the chart
# bar.print_echarts_options() # for debug

bar.use_theme('dark')
#设置主题色系
#共5种，具体可以参考 http://pyecharts.org/#/zh-cn/themes

bar.render('barchart.html')

# version 3 (two data category & marks)
bar =  Bar('example')
attr = ['class 1','class 2','class 3','class 4', 'class 5']
v1 = [5, 20, 36, 12, 30]
v2 = [10, 25, 8, 9, 28]
bar =  Bar('bar example')
bar.add("A", attr, v1, mark_line=['average'], mark_point=['max'])
bar.add("B", attr, v2,  mark_line=['average'], mark_point=['max'])
bar.render('bar_chart.html')


# stacked bar (connected)
attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
v1 = [5, 20, 36, 10, 75, 90]
v2 = [10, 25, 8, 60, 20, 80]
bar = Bar("直方图示例")
bar.add("", attr * 2, v1 + v2, bar_category_gap=0)
bar.render('stacked_bar_chart.html')

# stacked bar (vertical)
bar =  Bar('example')
attr = ['class 1','class 2','class 3']
v1 = [5, 20, 36]
v2 = [10, 25, 8]
bar =  Bar('bar example')
bar.add("A", attr, v1, is_stack=True)
bar.add("B", attr, v2, is_stack=True, is_convert=True)
bar.render('stacked_bar_chart.html')

# stacked bar (vertical) - water fall
attr = ["{}月".format(i) for i in range(1, 8)]
v1 = [0, 100, 200, 300, 400, 220, 250]
v2 = [1000, 800, 600, 500, 450, 400, 300]
bar = Bar("瀑布图示例")
# 利用第一个 add() 图例的颜色为透明，即 'rgba(0,0,0,0)'，并且设置 is_stack 标志为 True
bar.add("", attr, v1, label_color=['rgba(0,0,0,0)'], is_stack=True)
bar.add("月份", attr, v2, is_label_show=True, is_stack=True, label_pos='inside')
bar.render('waterfall.html')

# Bar_3D
bar3d = Bar3D("3D 柱状图示例", width=1200, height=600)
x_axis = [
    "12a", "1a", "2a", "3a", "4a", "5a", "6a", "7a", "8a", "9a", "10a", "11a",
    "12p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "10p", "11p"
    ]
y_axis = [
    "Saturday", "Friday", "Thursday", "Wednesday", "Tuesday", "Monday", "Sunday"
    ]
data = [
    [0, 0, 5], [0, 1, 1], [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0],
    [0, 6, 0], [0, 7, 0], [0, 8, 0], [0, 9, 0], [0, 10, 0], [0, 11, 2],
    [0, 12, 4], [0, 13, 1], [0, 14, 1], [0, 15, 3], [0, 16, 4], [0, 17, 6],
    [0, 18, 4], [0, 19, 4], [0, 20, 3], [0, 21, 3], [0, 22, 2], [0, 23, 5],
    [1, 0, 7], [1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 4, 0], [1, 5, 0],
    [1, 6, 0], [1, 7, 0], [1, 8, 0], [1, 9, 0], [1, 10, 5], [1, 11, 2],
    [1, 12, 2], [1, 13, 6], [1, 14, 9], [1, 15, 11], [1, 16, 6], [1, 17, 7],
    [1, 18, 8], [1, 19, 12], [1, 20, 5], [1, 21, 5], [1, 22, 7], [1, 23, 2],
    [2, 0, 1], [2, 1, 1], [2, 2, 0], [2, 3, 0], [2, 4, 0], [2, 5, 0],
    [2, 6, 0], [2, 7, 0], [2, 8, 0], [2, 9, 0], [2, 10, 3], [2, 11, 2],
    [2, 12, 1], [2, 13, 9], [2, 14, 8], [2, 15, 10], [2, 16, 6], [2, 17, 5],
    [2, 18, 5], [2, 19, 5], [2, 20, 7], [2, 21, 4], [2, 22, 2], [2, 23, 4],
    [3, 0, 7], [3, 1, 3], [3, 2, 0], [3, 3, 0], [3, 4, 0], [3, 5, 0],
    [3, 6, 0], [3, 7, 0], [3, 8, 1], [3, 9, 0], [3, 10, 5], [3, 11, 4],
    [3, 12, 7], [3, 13, 14], [3, 14, 13], [3, 15, 12], [3, 16, 9], [3, 17, 5],
    [3, 18, 5], [3, 19, 10], [3, 20, 6], [3, 21, 4], [3, 22, 4], [3, 23, 1],
    [4, 0, 1], [4, 1, 3], [4, 2, 0], [4, 3, 0], [4, 4, 0], [4, 5, 1],
    [4, 6, 0], [4, 7, 0], [4, 8, 0], [4, 9, 2], [4, 10, 4], [4, 11, 4],
    [4, 12, 2], [4, 13, 4], [4, 14, 4], [4, 15, 14], [4, 16, 12], [4, 17, 1],
    [4, 18, 8], [4, 19, 5], [4, 20, 3], [4, 21, 7], [4, 22, 3], [4, 23, 0],
    [5, 0, 2], [5, 1, 1], [5, 2, 0], [5, 3, 3], [5, 4, 0], [5, 5, 0],
    [5, 6, 0], [5, 7, 0], [5, 8, 2], [5, 9, 0], [5, 10, 4], [5, 11, 1],
    [5, 12, 5], [5, 13, 10], [5, 14, 5], [5, 15, 7], [5, 16, 11], [5, 17, 6],
    [5, 18, 0], [5, 19, 5], [5, 20, 3], [5, 21, 4], [5, 22, 2], [5, 23, 0],
    [6, 0, 1], [6, 1, 0], [6, 2, 0], [6, 3, 0], [6, 4, 0], [6, 5, 0],
    [6, 6, 0], [6, 7, 0], [6, 8, 0], [6, 9, 0], [6, 10, 1], [6, 11, 0],
    [6, 12, 2], [6, 13, 1], [6, 14, 3], [6, 15, 4], [6, 16, 0], [6, 17, 0],
    [6, 18, 0], [6, 19, 0], [6, 20, 1], [6, 21, 2], [6, 22, 2], [6, 23, 6]
    ]
range_color = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
               '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
bar3d.add("", x_axis, y_axis, [[d[1], d[0], d[2]] for d in data],
          is_visualmap=True, visual_range=[0, 20],
          visual_range_color=range_color, grid3d_width=200, grid3d_depth=80)
print('rendering')
bar3d.render()


# ----------------------------------------------------------------------------
# pie chart
pie =  Pie('PUBG')
pie.add('Time', df.Name, df.Time)
pie.render('piechart.html')

# version 2 (two charts)
pie =  Pie('PUBG')
left_pos = [25, 50]
gap = 50
right_pos = [left_pos[0] + gap, 50]
pie.add('Time', df.Name, df.Time, center=left_pos)
pie.add('Kills',df.Name, df.Kills, center=right_pos)
pie.render('piechart.html')


# ring chart
pie =  Pie('PUBG')
left_pos = [25, 50]
gap = 50
right_pos = [left_pos[0] + gap, 50]
pie.add('Time', df.Name, df.Time, center=left_pos, radius = [30, 75])
pie.add('Kills',df.Name, df.Kills, center=right_pos, radius = [30, 75])
pie.render('ringchart.html')

# rose chart
def pos_assign(n):
    return [[(i+1)*float(100/(n+1)),50] for i in range(n)]

positions = pos_assign(3)

pie =  Pie('PUBG', width=1800)
pie.add('Time', df.Name, df.Time, center=positions[0], radius = [20, 75], rosetype='radius', is_legend_show=False, is_label_show=True)
pie.add('Kills',df.Name, df.Kills, center=positions[1], radius = [20, 75],rosetype='radius', is_legend_show=False, is_label_show=True)
pie.add('Kills',df.Name, df.Kills, center=positions[2], radius = [20, 75],rosetype='area', is_legend_show=False, is_label_show=True)
pie.render('rosechart.html')


# -----------------------------------------------------------------------------
# boxplot
from pyecharts import Boxplot

boxplot = Boxplot("箱形图")
x_axis = ['expr1', 'expr2']
y_axis1 = [
    [850, 740, 900, 1070, 930, 850, 950, 980, 980, 880,
    1000, 980, 930, 650, 760, 810, 1000, 1000, 960, 960],
    [960, 940, 960, 940, 880, 800, 850, 880, 900, 840,
    830, 790, 810, 880, 880, 830, 800, 790, 760, 800],
]
y_axis2 = [
    [890, 810, 810, 820, 800, 770, 760, 740, 750, 760,
    910, 920, 890, 860, 880, 720, 840, 850, 850, 780],
    [890, 840, 780, 810, 760, 810, 790, 810, 820, 850,
    870, 870, 810, 740, 810, 940, 950, 800, 810, 870]
]
    
print(boxplot.prepare_data(y_axis1))
boxplot.add("category1", x_axis, boxplot.prepare_data(y_axis1))
boxplot.add("category2", x_axis, boxplot.prepare_data(y_axis2))
boxplot.render()




# K line
from pyecharts import Kline
v1 = [[2320.26, 2320.26, 2287.3, 2362.94], [2300, 2291.3, 2288.26, 2308.38],
      [2295.35, 2346.5, 2295.35, 2345.92], [2347.22, 2358.98, 2337.35, 2363.8],
      [2360.75, 2382.48, 2347.89, 2383.76], [2383.43, 2385.42, 2371.23, 2391.82],
      [2377.41, 2419.02, 2369.57, 2421.15], [2425.92, 2428.15, 2417.58, 2440.38],
      [2411, 2433.13, 2403.3, 2437.42], [2432.68, 2334.48, 2427.7, 2441.73],
      [2430.69, 2418.53, 2394.22, 2433.89], [2416.62, 2432.4, 2414.4, 2443.03],
      [2441.91, 2421.56, 2418.43, 2444.8], [2420.26, 2382.91, 2373.53, 2427.07],
      [2383.49, 2397.18, 2370.61, 2397.94], [2378.82, 2325.95, 2309.17, 2378.82],
      [2322.94, 2314.16, 2308.76, 2330.88], [2320.62, 2325.82, 2315.01, 2338.78],
      [2313.74, 2293.34, 2289.89, 2340.71], [2297.77, 2313.22, 2292.03, 2324.63],
      [2322.32, 2365.59, 2308.92, 2366.16], [2364.54, 2359.51, 2330.86, 2369.65],
      [2332.08, 2273.4, 2259.25, 2333.54], [2274.81, 2326.31, 2270.1, 2328.14],
      [2333.61, 2347.18, 2321.6, 2351.44], [2340.44, 2324.29, 2304.27, 2352.02],
      [2326.42, 2318.61, 2314.59, 2333.67], [2314.68, 2310.59, 2296.58, 2320.96],
      [2309.16, 2286.6, 2264.83, 2333.29], [2282.17, 2263.97, 2253.25, 2286.33],
      [2255.77, 2270.28, 2253.31, 2276.22]]
kline = Kline("K 线图示例")
kline.add("日K", ["2017/7/{}".format(i + 1) for i in range(31)], v1)
kline.render()


#-----------------------------------------------------------------------------
#Parallel chart

from pyecharts import Parallel

c_schema = [
    {"dim": 0, "name": "data"},
    {"dim": 1, "name": "AQI"},
    {"dim": 2, "name": "PM2.5"},
    {"dim": 3, "name": "PM10"},
    {"dim": 4, "name": "CO"},
    {"dim": 5, "name": "NO2"},
    {"dim": 6, "name": "CO2"},
    {"dim": 7, "name": "等级",
    "type": "category",
    "data": ['优', '良', '轻度污染', '中度污染', '重度污染', '严重污染']}
]
data = [
    [1, 91, 45, 125, 0.82, 34, 23, "良"],
    [2, 65, 27, 78, 0.86, 45, 29, "良"],
    [3, 83, 60, 84, 1.09, 73, 27, "良"],
    [4, 109, 81, 121, 1.28, 68, 51, "轻度污染"],
    [5, 106, 77, 114, 1.07, 55, 51, "轻度污染"],
    [6, 109, 81, 121, 1.28, 68, 51, "轻度污染"],
    [7, 106, 77, 114, 1.07, 55, 51, "轻度污染"],
    [8, 89, 65, 78, 0.86, 51, 26, "良"],
    [9, 53, 33, 47, 0.64, 50, 17, "良"],
    [10, 80, 55, 80, 1.01, 75, 24, "良"],
    [11, 117, 81, 124, 1.03, 45, 24, "轻度污染"],
    [12, 99, 71, 142, 1.1, 62, 42, "良"],
    [13, 95, 69, 130, 1.28, 74, 50, "良"],
    [14, 116, 87, 131, 1.47, 84, 40, "轻度污染"]
]
parallel = Parallel("平行坐标系-用户自定义指示器")
parallel.config(c_schema=c_schema)
parallel.add("parallel", data, is_random=False)
parallel.render()

# -----------------------------------------------------------------------------
# polar
from pyecharts import Polar
from pyecharts import Page
page = Page()

import random
data = [(i, random.randint(1, 100)) for i in range(101)]  #[(radius, angle)]
polar = Polar("极坐标系")
polar.add("", data, is_random=False)
page.add(polar)

# error
polar2 = Polar("极坐标系-散点图示例")
polar2.add("", data, type='scatter',
          is_splitline_show=False, is_radiusaxis_show=False)
page.add(polar2) 


page.render()


page = Page()
radius = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
polar = Polar("极坐标系-堆叠柱状图示例", width=1200, height=600)
polar.add("A", [1, 2, 3, 4, 3, 5, 1], radius_data=radius,
          type='barRadius', is_stack=True)
polar.add("B", [2, 4, 6, 1, 2, 3, 1], radius_data=radius,
          type='barRadius', is_stack=True)
polar.add("C", [1, 2, 3, 4, 1, 2, 5], radius_data=radius,
          type='barRadius', is_stack=True)
page.add(polar)


polar = Polar("极坐标系-堆叠柱状图示例", width=1200, height=600)
polar.add("", [1, 2, 3, 4, 3, 5, 1], radius_data=radius,
          type='barAngle', is_stack=True)
polar.add("", [2, 4, 6, 1, 2, 3, 1], radius_data=radius,
          type='barAngle', is_stack=True)
polar.add("", [1, 2, 3, 4, 1, 2, 5], radius_data=radius,
          type='barAngle', is_stack=True)
page.add(polar)

page.render('polar.html')



# heatmap on polar



def render_item(params, api):
    values = [api.value(0), api.value(1)]
    coord = api.coord(values)
    size = api.size([1, 1], values)
    return {
        "type": "sector",
        "shape": {
            "cx": params.coordSys.cx,
            "cy": params.coordSys.cy,
            "r0": coord[2] - size[0] / 2,
            "r": coord[2] + size[0] / 2,
            "startAngle": coord[3] - size[1] / 2,
            "endAngle": coord[3] + size[1] / 2,
        },
        "style": api.style({"fill": api.visual("color")}),
    }


def heatmap_polar(df, value, attr1, attr2):
    polar = Polar("自定义渲染逻辑示例", width=1200, height=600)
    polar.add(
    "",
    [
        [
            list(set(df[attr1])).index(df[attr1][i]),
            list(set(df[attr2])).index(df[attr2][i]),
            df[value][i],
        ]
        for i in range(df.shape[0])
    ],
    render_item=render_item,
    type="custom",
    angle_data=list(set(df[attr1])),
    radius_data=list(set(df[attr2])),
    is_visualmap=True,
    visual_range=[0, max(df[value])]
    )
    polar.render()

df2=pd.DataFrame({
        'x':['A','A','A','A','B','B','B','B','C','C','C','C','D','D','D','D'],
        'y':['1','2','3','4','1','2','3','4','1','2','3','4','1','2','3','4'],
        'v':range(16)})
    
heatmap_polar(df2,'v','x','y')




#-----------------------------------------------------------------------------
# radar plot
from pyecharts import Radar
schema = [ 
    ("销售", 6500), ("管理", 16000), ("信息技术", 30000),
    ("客服", 38000), ("研发", 52000), ("市场", 25000)
]
v1 = [[4300, 10000, 28000, 35000, 50000, 19000]]
v2 = [[5000, 14000, 28000, 31000, 42000, 21000]]
radar = Radar()
radar.config(schema)
radar.add("预算分配", v1, is_splitline=True, is_axisline_show=True, symbol=None,
          area_color=RGB(255, 120, 0), item_color=RGB(255, 120, 0), is_area_show=True, area_opacity=0.5)
radar.add("实际开销", v2, symbol=None,
           area_color=RGB(50, 50, 0), item_color=RGB(50, 50, 0), is_area_show=True, area_opacity=0.5)
radar.render()


#------------------------------------------------------------------------------
# Funnel

attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
value = [1, 11, 15, 30, 45, 101]
funnel = Funnel("漏斗图示例")
funnel.add("商品", attr, value, is_label_show=True,
           label_pos="inside", label_text_color="#fff")
funnel.render('funnel.html')


# ----------------------------------------------------------------------------
# Gauge
from pyecharts import Gauge,Liquid

gauge = Gauge("仪表盘示例")
gauge.add("业务指标", "完成率", 166.66, angle_range=[180, 0],
          scale_range=[0, 200], is_legend_show=False)
gauge.render('Gauge.html')

liquid = Liquid("水球图示例")
liquid.add("Liquid", [0.6])
liquid.render()

# ----------------------------------------------------------------------------
# heatmap
import random
from pyecharts import HeatMap

x_axis = [
    "12a", "1a", "2a", "3a", "4a", "5a", "6a", "7a", "8a", "9a", "10a", "11a",
    "12p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "10p", "11p"]
y_axis = [
    "Saturday", "Friday", "Thursday", "Wednesday", "Tuesday", "Monday", "Sunday"]
data = [[i, j, random.randint(0, 50)] for i in range(24) for j in range(7)]
heatmap = HeatMap()
heatmap.add("热力图直角坐标系", x_axis, y_axis, data, is_visualmap=True,
            visual_text_color="#000", visual_orient='horizontal')
heatmap.render()


def heatmap_2attr(df, attr1, attr2, func=len, value=None):
    if value==None:
        df2=df.groupby([attr1, attr2])[attr1].agg([func])
    else:
        df2=df.groupby([attr1, attr2])[value].agg([func])
    df2=df2.reset_index()
    
    x_axis = list(set(df2[attr1]))
    y_axis = list(set(df2[attr2]))
    
    data = [[x_axis.index(i), y_axis.index(j), df2.loc[(df2[attr1]==i) & (df2[attr2]==j), func.__name__]] 
            for i in x_axis for j in y_axis]
    print(data)
    
    heatmap = HeatMap()
    heatmap.add("热力图直角坐标系", x_axis, y_axis, data, is_visualmap=True,
                visual_range=[0, max(df2[func.__name__])],
            visual_text_color="#000", visual_orient='horizontal')
    heatmap.render()

heatmap_2attr(df, 'Group','Place',func=np.mean, value='Kills')



# calender heatmap
import datetime
import random
from pyecharts import HeatMap

begin = datetime.date(2017, 1, 1)
end = datetime.date(2017, 12, 31)
data = [[str(begin + datetime.timedelta(days=i)),
        random.randint(1000, 25000)] for i in range((end - begin).days + 1)]
print(data)


heatmap = HeatMap("日历热力图示例", "某人 2017 年微信步数情况", width=1100)
heatmap.add("", data, is_calendar_heatmap=True,calendar_date_range="2017",
            calendar_cell_size=['auto', 30],
            # visual_orient="horizontal", visual_pos="center", visual_top="80%",
            is_visualmap=True, is_piecewise=True, visual_range=[1000, 25000])
heatmap.render()

#-----------------------------------------------------------------------------
# connected graph
from pyecharts import Graph

nodes = [{"name": "1", "symbolSize": 10},
         {"name": "2", "symbolSize": 20},
         {"name": "3", "symbolSize": 13},
         {"name": "4", "symbolSize": 4},
         {"name": "5", "symbolSize": 5},
         {"name": "6", "symbolSize": 4},
         {"name": "7", "symbolSize": 13},
         {"name": "8", "symbolSize": 20}]
links = []
for i in nodes:
    for j in nodes:
        links.append({"source": i.get('name'), "target": j.get('name')})
        
link_1 = [{'source': '1', 'target':'5'}, {'source': '1', 'target':'7'},{'source': '3', 'target':'6'}]
link_2 = [{'source': '2', 'target':'4'}, {'source': '4', 'target':'8'},{'source': '3', 'target':'7'}]
link_3 = [{'source': '1', 'target':'2'}, {'source': '2', 'target':'3'},{'source': '7', 'target':'8'}]
links = link_1 + link_2 + link_3
        
graph = Graph("关系图-力引导布局示例")
graph.add("", nodes, links, repulsion=80, graph_gravity=0.1) # graph_gravity 代表相心力 1最大。 repulsion 代表节点斥力 无上限
graph.render()

nodes = [{"name": "1", "symbolSize": 10},
         {"name": "2", "symbolSize": 20},
         {"name": "3", "symbolSize": 13},
         {"name": "4", "symbolSize": 4},
         {"name": "5", "symbolSize": 5},
         {"name": "6", "symbolSize": 4},
         {"name": "7", "symbolSize": 13},
         {"name": "8", "symbolSize": 20}]
        
link_1 = [{'source': '1', 'target':'5'}, {'source': '1', 'target':'7'},{'source': '3', 'target':'6'}]
link_2 = [{'source': '2', 'target':'4'}, {'source': '4', 'target':'8'},{'source': '3', 'target':'7'}]
link_3 = [{'source': '1', 'target':'2'}, {'source': '2', 'target':'3'},{'source': '7', 'target':'8'}]
links = link_1 + link_2+link_3
        
graph = Graph("关系图-力引导布局示例")
graph.add("", nodes, links, repulsion=80, graph_layout='circular',graph_edge_symbol='arrow')
graph.render()

# ----------------------------------------------------------------------------
# multi plot
attr = [1,2,3]

line = Line()
line.add('data', attr,[9,3,6])

bar =  Bar()
bar.add('data', attr, [4,2,5])

pie =  Pie()
bar.add('data', attr, [4,2,5])

overlap = Overlap()
overlap.add(bar)
overlap.add(line)
overlap.render('overlap.html')   


# plot multi times
pie = Pie('各类电影中"好片"所占的比例', "数据来着豆瓣", title_pos='center')
style = Style()
pie_style = style.add(
    label_pos="top",
    is_label_show=True,
    label_text_color=None
)

pie.add("", ["剧情", ""], [25, 75], center=[10, 30],
        radius=[18, 24], **pie_style)
pie.add("", ["奇幻", ""], [24, 76], center=[30, 30],
        radius=[18, 24], **pie_style)
pie.add("", ["爱情", ""], [14, 86], center=[50, 30],
        radius=[18, 24], **pie_style)
pie.add("", ["惊悚", ""], [11, 89], center=[70, 30],
        radius=[18, 24], **pie_style)
pie.add("", ["冒险", ""], [27, 73], center=[90, 30],
        radius=[18, 24], **pie_style)
pie.add("", ["动作", ""], [15, 85], center=[10, 70],
        radius=[18, 24], **pie_style)
pie.add("", ["喜剧", ""], [54, 46], center=[30, 70],
        radius=[18, 24], **pie_style)
pie.add("", ["科幻", ""], [26, 74], center=[50, 70],
        radius=[18, 24], **pie_style)
pie.add("", ["悬疑", ""], [25, 75], center=[70, 70],
        radius=[18, 24], **pie_style)
pie.add("", ["犯罪", ""], [28, 72], center=[90, 70],
        radius=[18, 24], legend_top="center", **pie_style)
pie.render()





# multi plot 
page =  Page()  
page.add(line)
page.add(bar)  
page.add(pie)  
page.render('multi00.html')     

# multi render:
env = pch_env("html")
# 为渲染创建一个默认配置环境
# create_default_environment(filet_ype)
# file_type: 'html', 'svg', 'png', 'jpeg', 'gif' or 'pdf'

env.render_chart_to_file(bar, path='bar.html')
env.render_chart_to_file(line, path='line.html')

# -----------------------------------------------------------------------------
# others

# title_pos: 'left''right''center''auto'
# chart_name = tpye(title_pos='center')
# 大小，位置，颜色

# add:

# xaxis_rotate=20
# xaxis_name
# xaxis_type = 'category'
# xaxis_formatter = func(params)

# xaxis3d_name -> str

# is_legend_show
# legend_pos = 'center'
# legend_orient = 'vertical'


# is_label_show
# label_formatter = str # {a}, {b}，{c}，{d}，{e}，分别表示系列名，数据名，数据值

# mark with information
line = Line("折线图示例")
line.add("商家A", attr, v1,
            mark_point=["average", {
                "coord": ["裤子", 10], "name": "这是我想要的第一个标记点"}])
line.add("商家B", attr, v2, is_smooth=True,
            mark_point=[{
                "coord": ["袜子", 80], "name": "这是我想要的第二个标记点"}])
line.render()






# Geo map
from pyecharts import Geo, GeoLines, Style

data = [
    ("海门", 9),("鄂尔多斯", 12),("招远", 12),("舟山", 12),("齐齐哈尔", 14),("盐城", 15),
    ("赤峰", 16),("青岛", 18),("乳山", 18),("金昌", 19),("泉州", 21),("莱西", 21),
    ("日照", 21),("胶南", 22),("南通", 23),("拉萨", 24),("云浮", 24),("梅州", 25),
    ("文登", 25),("上海", 25),("攀枝花", 25),("威海", 25),("承德", 25),("厦门", 26),
    ("汕尾", 26),("潮州", 26),("丹东", 27),("太仓", 27),("曲靖", 27),("烟台", 28),
    ("福州", 29),("瓦房店", 30),("即墨", 30),("抚顺", 31),("玉溪", 31),("张家口", 31),
    ("阳泉", 31),("莱州", 32),("湖州", 32),("汕头", 32),("昆山", 33),("宁波", 33),
    ("湛江", 33),("揭阳", 34),("荣成", 34),("连云港", 35),("葫芦岛", 35),("常熟", 36),
    ("东莞", 36),("河源", 36),("淮安", 36),("泰州", 36),("南宁", 37),("营口", 37),
    ("惠州", 37),("江阴", 37),("蓬莱", 37),("韶关", 38),("嘉峪关", 38),("广州", 38),
    ("延安", 38),("太原", 39),("清远", 39),("中山", 39),("昆明", 39),("寿光", 40),
    ("盘锦", 40),("长治", 41),("深圳", 41),("珠海", 42),("宿迁", 43),("咸阳", 43),
    ("铜川", 44),("平度", 44),("佛山", 44),("海口", 44),("江门", 45),("章丘", 45),
    ("肇庆", 46),("大连", 47),("临汾", 47),("吴江", 47),("石嘴山", 49),("沈阳", 50),
    ("苏州", 50),("茂名", 50),("嘉兴", 51),("长春", 51),("胶州", 52),("银川", 52),
    ("张家港", 52),("三门峡", 53),("锦州", 54),("南昌", 54),("柳州", 54),("三亚", 54),
    ("自贡", 56),("吉林", 56),("阳江", 57),("泸州", 57),("西宁", 57),("宜宾", 58),
    ("呼和浩特", 58),("成都", 58),("大同", 58),("镇江", 59),("桂林", 59),("张家界", 59),
    ("宜兴", 59),("北海", 60),("西安", 61),("金坛", 62),("东营", 62),("牡丹江", 63),
    ("遵义", 63),("绍兴", 63),("扬州", 64),("常州", 64),("潍坊", 65),("重庆", 66),
    ("台州", 67),("南京", 67),("滨州", 70),("贵阳", 71),("无锡", 71),("本溪", 71),
    ("克拉玛依", 72),("渭南", 72),("马鞍山", 72),("宝鸡", 72),("焦作", 75),("句容", 75),
    ("北京", 79),("徐州", 79),("衡水", 80),("包头", 80),("绵阳", 80),("乌鲁木齐", 84),
    ("枣庄", 84),("杭州", 84),("淄博", 85),("鞍山", 86),("溧阳", 86),("库尔勒", 86),
    ("安阳", 90),("开封", 90),("济南", 92),("德阳", 93),("温州", 95),("九江", 96),
    ("邯郸", 98),("临安", 99),("兰州", 99),("沧州", 100),("临沂", 103),("南充", 104),
    ("天津", 105),("富阳", 106),("泰安", 112),("诸暨", 112),("郑州", 113),("哈尔滨", 114),
    ("聊城", 116),("芜湖", 117),("唐山", 119),("平顶山", 119),("邢台", 119),("德州", 120),
    ("济宁", 120),("荆州", 127),("宜昌", 130),("义乌", 132),("丽水", 133),("洛阳", 134),
    ("秦皇岛", 136),("株洲", 143),("石家庄", 147),("莱芜", 148),("常德", 152),("保定", 153),
    ("湘潭", 154),("金华", 157),("岳阳", 169),("长沙", 175),("衢州", 177),("廊坊", 193),
    ("菏泽", 194),("合肥", 229),("武汉", 273),("大庆", 279)]

geo = Geo("全国主要城市空气质量", "data from pm2.5", title_color="#fff",
          title_pos="center", width=1200,
          height=600, background_color='#404a59')
attr, value = geo.cast(data)
geo.add("", attr, value, symbol_size=15, is_visualmap=True, visual_range=[0, 200], visual_text_color="#fff")
geo.render()

# piecewise visual map
geo = Geo("全国主要城市空气质量", "data from pm2.5", title_color="#fff",
          title_pos="center", width=1200,
          height=600, background_color='#404a59')
attr, value = geo.cast(data)
geo.add("", attr, value, visual_range=[0, 200], visual_text_color="#fff",
        symbol_size=15, is_visualmap=True, is_piecewise=True, visual_split_number=6)
geo.render()

# heatmap on geo
geo = Geo("全国主要城市空气质量", "data from pm2.5", title_color="#fff",
          title_pos="center", width=1200,
          height=600, background_color='#404a59')
attr, value = geo.cast(data)
geo.add("", attr, value, type="heatmap", is_visualmap=True, visual_range=[0, 300],
        visual_text_color='#fff', map_type='china')
geo.render()

# effect scatter
data =[
    ('汕头市', 50), ('汕尾市', 60), ('揭阳市', 35),
    ('阳江市', 44), ('肇庆市', 72)
    ]
geo = Geo("广东城市空气质量", "data from pm2.5", title_color="#fff",
          title_pos="center", width=1200,
          height=600, background_color='#404a59')
attr, value = geo.cast(data)
geo.add("", attr, value, maptype='广东', type="effectScatter",
        is_random=True, effect_scale=5, is_legend_show=False)
geo.render()




# geo lines

data_guangzhou = [
    ["广州", "上海"],
    ["广州", "北京"],
    ["广州", "南京"],
    ["广州", "重庆"],
    ["广州", "兰州"],
    ["广州", "杭州"]
]
geolines = GeoLines("GeoLines 示例")
geolines.add("从广州出发", data_guangzhou, is_legend_show=False)
geolines.render()

# airplane lines
fig_style = Style()
style_1 = fig_style.add(
    title_pos = "center",
    width=1200,
    height=600,
    background_color="#404a59"
)

style = Style()
style_2 = style.add(
    geo_effect_symbol="plane",
    geo_effect_symbolsize=15,
    
    line_curve=0.2,
    line_opacity=0.6,
    
    is_label_show=True,
    label_color=['#a6c84c', '#ffa022', '#46bee9'],
    label_pos="right",
    label_formatter="{b}",
    label_text_color="#eee",
    
    legend_pos="right",
)
geolines = GeoLines("GeoLines 示例", **style_1)
geolines.add("从广州出发", data_guangzhou, **style_2)
geolines.render()


# Map with section
from pyecharts import Map

value = [155, 10, 66, 78, 33, 80, 190, 53, 49.6]
attr = [
    "福建", "山东", "北京", "上海", "甘肃", "新疆", "河南", "广西", "西藏"
    ]
map = Map("Map 结合 VisualMap 示例", width=1200, height=600)
map.add("", attr, value, maptype='china', is_visualmap=True,
        visual_text_color='#000', is_map_symbol_show=False)  # maptype can be 'world' or '四川'
map.render()



#----------------------------------------------------------------------------
# sankey
from pyecharts import Sankey

nodes = [
    {'name': 'engineer', 'value': 20}, {'name': 'scientist', 'value': 10}, 
    {'name': 'business', 'value': 10}, {'name': 'officer', 'value': 8}, 
    {'name': 'team1', 'value': 20},{'name': 'team2', 'value': 10},
    {'name': 'team3', 'value': 10},{'name': 'team4', 'value': 5},
]

links = [
    {'source': 'engineer', 'target': 'team1', 'value': 10},
    {'source': 'scientist', 'target': 'team1', 'value': 5},
    {'source': 'business', 'target': 'team1', 'value': 3},
    {'source': 'designer', 'target': 'team1', 'value': 2},
    {'source': 'scientist', 'target': 'team2', 'value': 8},
    {'source': 'officier', 'target': 'team2', 'value': 2},
    {'source': 'officier', 'target': 'team3', 'value': 3},
    {'source': 'officier', 'target': 'team4', 'value': 4},
]
sankey = Sankey("桑基图示例", width=1200, height=600)
sankey.add("sankey", nodes, links, line_opacity=0.2,
           line_curve=0.5, line_color='source',
           is_label_show=True, label_pos='right')
sankey.render()


def cross_sankey(df, attr1, attr2, func=len, value=None):
    if value==None:
        df2=df.groupby([attr1, attr2])[attr1].agg([func])
    else:
        df2=df.groupby([attr1, attr2])[value].agg([func])
    df2=df2.reset_index()
    nodes = []
    for j in [attr1, attr2]:
        for i in set(df[j]):
            nodes.append({'name': str(i)})
    links = []
    for i in range(df2.shape[0]):
        if value==None:
            links.append({
                    'source':str(df2.loc[i,attr1]),
                    'target':str(df2.loc[i,attr2]),
                    'value':df2.loc[i,'len'],
                    })
        else:
            links.append({
                    'source':str(df2.loc[i,attr1]),
                    'target':str(df2.loc[i,attr2]),
                    'value':df2.loc[i,func.__name__],
                    })
    
    sankey = Sankey("桑基图示例", width=1200, height=600)
    sankey.add("sankey", nodes, links, line_opacity=0.8,
           line_curve=0.5, line_color='source',
           is_label_show=True, label_pos='right')
    sankey.render()
    
cross_sankey(df, 'Group','Place')
import numpy as np
cross_sankey(df, 'Group','Place', func=np.mean, value='Kills')

# ------------------------------------------------------------------------------
# theme river 
from pycharts import ThemeRiver
data = [
    ['2015/11/08', 10, 'DQ'], ['2015/11/09', 15, 'DQ'], ['2015/11/10', 35, 'DQ'],
    ['2015/11/14', 7, 'DQ'], ['2015/11/15', 2, 'DQ'], ['2015/11/16', 17, 'DQ'],
    ['2015/11/17', 33, 'DQ'], ['2015/11/18', 40, 'DQ'], ['2015/11/19', 32, 'DQ'],
    ['2015/11/20', 26, 'DQ'], ['2015/11/21', 35, 'DQ'], ['2015/11/22', 40, 'DQ'],
    ['2015/11/23', 32, 'DQ'], ['2015/11/24', 26, 'DQ'], ['2015/11/25', 22, 'DQ'],
    ['2015/11/08', 35, 'TY'], ['2015/11/09', 36, 'TY'], ['2015/11/10', 37, 'TY'],
    ['2015/11/11', 22, 'TY'], ['2015/11/12', 24, 'TY'], ['2015/11/13', 26, 'TY'],
    ['2015/11/14', 34, 'TY'], ['2015/11/15', 21, 'TY'], ['2015/11/16', 18, 'TY'],
    ['2015/11/17', 45, 'TY'], ['2015/11/18', 32, 'TY'], ['2015/11/19', 35, 'TY'],
    ['2015/11/20', 30, 'TY'], ['2015/11/21', 28, 'TY'], ['2015/11/22', 27, 'TY'],
    ['2015/11/23', 26, 'TY'], ['2015/11/24', 15, 'TY'], ['2015/11/25', 30, 'TY'],
    ['2015/11/26', 35, 'TY'], ['2015/11/27', 42, 'TY'], ['2015/11/28', 42, 'TY'],
    ['2015/11/08', 21, 'SS'], ['2015/11/09', 25, 'SS'], ['2015/11/10', 27, 'SS'],
    ['2015/11/11', 23, 'SS'], ['2015/11/12', 24, 'SS'], ['2015/11/13', 21, 'SS'],
    ['2015/11/14', 35, 'SS'], ['2015/11/15', 39, 'SS'], ['2015/11/16', 40, 'SS'],
    ['2015/11/17', 36, 'SS'], ['2015/11/18', 33, 'SS'], ['2015/11/19', 43, 'SS'],
    ['2015/11/20', 40, 'SS'], ['2015/11/21', 34, 'SS'], ['2015/11/22', 28, 'SS'],
    ['2015/11/14', 7, 'QG'], ['2015/11/15', 2, 'QG'], ['2015/11/16', 17, 'QG'],
    ['2015/11/17', 33, 'QG'], ['2015/11/18', 40, 'QG'], ['2015/11/19', 32, 'QG'],
    ['2015/11/20', 26, 'QG'], ['2015/11/21', 35, 'QG'], ['2015/11/22', 40, 'QG'],
    ['2015/11/23', 32, 'QG'], ['2015/11/24', 26, 'QG'], ['2015/11/25', 22, 'QG'],
    ['2015/11/26', 16, 'QG'], ['2015/11/27', 22, 'QG'], ['2015/11/28', 10, 'QG'],
    ['2015/11/08', 10, 'SY'], ['2015/11/09', 15, 'SY'], ['2015/11/10', 35, 'SY'],
    ['2015/11/11', 38, 'SY'], ['2015/11/12', 22, 'SY'], ['2015/11/13', 16, 'SY'],
    ['2015/11/14', 7, 'SY'], ['2015/11/15', 2, 'SY'], ['2015/11/16', 17, 'SY'],
    ['2015/11/17', 33, 'SY'], ['2015/11/18', 40, 'SY'], ['2015/11/19', 32, 'SY'],
    ['2015/11/20', 26, 'SY'], ['2015/11/21', 35, 'SY'], ['2015/11/22', 4, 'SY'],
    ['2015/11/23', 32, 'SY'], ['2015/11/24', 26, 'SY'], ['2015/11/25', 22, 'SY'],
    ['2015/11/26', 16, 'SY'], ['2015/11/27', 22, 'SY'], ['2015/11/28', 10, 'SY'],
    ['2015/11/08', 10, 'DD'], ['2015/11/09', 15, 'DD'], ['2015/11/10', 35, 'DD'],
    ['2015/11/11', 38, 'DD'], ['2015/11/12', 22, 'DD'], ['2015/11/13', 16, 'DD'],
    ['2015/11/14', 7, 'DD'], ['2015/11/15', 2, 'DD'], ['2015/11/16', 17, 'DD'],
    ['2015/11/17', 33, 'DD'], ['2015/11/18', 4, 'DD'], ['2015/11/19', 32, 'DD'],
    ['2015/11/20', 26, 'DD'], ['2015/11/21', 35, 'DD'], ['2015/11/22', 40, 'DD'],
    ['2015/11/23', 32, 'DD'], ['2015/11/24', 26, 'DD'], ['2015/11/25', 22, 'DD']
]
tr = ThemeRiver("主题河流图示例图")
tr.add(['DQ', 'TY', 'SS', 'QG', 'SY', 'DD'], data, is_label_show=True)
tr.render()




#---------------------------------------------------------------------------
# world cloud
from pyecharts import WordCloud


name = [
    'Sam S Club', 'Macys', 'Amy Schumer', 'Jurassic World', 'Charter Communications',
    'Chick Fil A', 'Planet Fitness', 'Pitch Perfect', 'Express', 'Home', 'Johnny Depp',
    'Lena Dunham', 'Lewis Hamilton', 'KXAN', 'Mary Ellen Mark', 'Farrah Abraham',
    'Rita Ora', 'Serena Williams', 'NCAA baseball tournament', 'Point Break']
value = [
    10000, 6181, 4386, 4055, 2467, 2244, 1898, 1484, 1112,
    965, 847, 582, 555, 550, 462, 366, 360, 282, 273, 265]

wordcloud_2 = WordCloud()
wordcloud_2.add("", name, value, word_size_range=[5, 30],
              shape='diamond')

# 'circle', 'cardioid', 'diamond', 'triangle-forward', 'triangle', 'pentagon', 'star'
wordcloud_3 = WordCloud()
wordcloud_3.add("", name, value, word_size_range=[5, 30],
              shape='cardioid')

wordcloud_3 = WordCloud()
wordcloud_3.add("", name, value, word_size_range=[5, 30],
              shape='triangle-forward')

wordcloud_3.render()








#------------------------------------------------------------------------------

from pyecharts import Bar, Line, Scatter, EffectScatter, Grid

attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
v1 = [5, 20, 36, 10, 75, 90]
v2 = [10, 25, 8, 60, 20, 80]
bar = Bar("柱状图示例", height=720, width=1200, title_pos="65%")
bar.add("商家A", attr, v1, is_stack=True)
bar.add("商家B", attr, v2, is_stack=True, legend_pos="80%")
line = Line("折线图示例")
attr = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
line.add(
    "最高气温",
    attr,
    [11, 11, 15, 13, 12, 13, 10],
    mark_point=["max", "min"],
    mark_line=["average"],
)
line.add(
    "最低气温",
    attr,
    [1, -2, 2, 5, 3, 2, 0],
    mark_point=["max", "min"],
    mark_line=["average"],
    legend_pos="20%",
)
v1 = [5, 20, 36, 10, 75, 90]
v2 = [10, 25, 8, 60, 20, 80]
scatter = Scatter("散点图示例", title_top="50%", title_pos="65%")
scatter.add("scatter", v1, v2, legend_top="50%", legend_pos="80%")
es = EffectScatter("动态散点图示例", title_top="50%")
es.add(
    "es",
    [11, 11, 15, 13, 12, 13, 10],
    [1, -2, 2, 5, 3, 2, 0],
    effect_scale=6,
    legend_top="50%",
    legend_pos="20%",
)

grid = Grid()
grid.add(bar, grid_bottom="60%", grid_left="60%")
grid.add(line, grid_bottom="60%", grid_right="60%")
grid.add(scatter, grid_top="60%", grid_left="60%")
grid.add(es, grid_top="60%", grid_right="60%")
grid.render()


# Grid not support word cloud




##########################    formatter #######################################
def label_formatter(params):
      return params.value + ' [Good!]'
  
{
      componentType: 'series',
      // 系列类型
      seriesType: string,
      // 系列在传入的 option.series 中的 index
      seriesIndex: number,
      // 系列名称
      seriesName: string,
      // 数据名，类目名
      name: string,
      // 数据在传入的 data 数组中的 index
      dataIndex: number,
      // 传入的原始数据项
      data: Object,
      // 传入的数据值
      value: number|Array,
      // 数据图形的颜色
      color: string,
  } 
