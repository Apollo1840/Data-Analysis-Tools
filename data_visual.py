# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:00:11 2018

@author: zouco
"""

import pandas as pd
import numpy as np
from pprint import pprint
import random


df=pd.read_csv('movie_metadata.csv')
df.dropna(inplace=True)

import seaborn as sns
sns.heatmap(df.corr(), annot=True)


##################### step 1 #########################
# know
df.info()

pd.set_option('display.max_columns', None)
df.head(12)
pd.reset_option('display.max_columns')


df['movie_facebook_likes'].dtype

df.describe()


def glimpse_data(df):
    with open('dt_describe.txt','a') as f:
        
        objects_num={'object':[], 'num':[]}
        objects_levels={'object':[], 'levels':[]}
        int_col = []
        float_col = []
        other_col = []
        
        for column_name in df.columns:
            if df[column_name].dtype=='O':
                objects_num['object'].append(column_name)
                objects_num['num'].append(len(set(df[column_name])))
                if len(set(df[column_name])) < 40:
                    objects_levels['object'].append(column_name)
                    objects_levels['levels'].append(set(df[column_name]))
            elif df[column_name].dtype == 'int64':     
                int_col.append(column_name)
            elif df[column_name].dtype == 'float64':     
                float_col.append(column_name)
            else:
                other_col.append(column_name)
        
            
        df_on = pd.DataFrame(objects_num)
        df_on = df_on.sort_values(by='object')
        int_col = sorted(int_col)
        float_col = sorted(float_col)
        
        f.write('\n--name of object columns:\n\n')
        f.write(df_on.__str__())
        f.write('\n\n')
        f.write(objects_levels.__str__())
        f.write('\n\n')
        
        f.write('\n--name of int columns:\n\n')
        for i in int_col:
            f.write(i+'\n')
        f.write('\n')
        
        f.write('\n--name of float columns:\n\n')
        for i in float_col:
            f.write(i+'\n')
        f.write('\n')
        
        f.write('\n--name of other columns:\n\n')
        for i in other_col:
            f.write(i+'\n')
        f.write('\n')
        
        f.write('\n------------------------------------------------------------------\n')
        f.write(df.describe().__str__())
        f.write('\n')

glimpse_data(df)
#----------------------------------------------------------------------------




















##################### step 2 #########################

import pandas_extend as pde
from pandas_extend import da_opt as dp

def draw_first_barchart():
    
    df['profits'] = [(i-j)/1000000 for i,j in zip(df.gross,df.budget)]
    value_column = 'profits'
    
    mc_column = 'genres'
    mc_sep = '|'
    
    cut_column = 'title_year'
    year_labels=["old","80's","90's","00's","recent"]
    bins = [1926,1979,1989,1999,2011,2016]
    df[cut_column+'_level']=pd.cut(df[cut_column], bins=bins, labels=year_labels)
    
    df_mc_column = pde.multi_content_column(df[mc_column], sep=mc_sep)
    x_axis = list(df_mc_column.classes.keys())
    
    # process data
    value = []
    for i in year_labels:
        dfi = df.loc[df[cut_column+'_level'] == i,:]
        dfi_mc_column = pde.multi_content_column(dfi[mc_column], sep=mc_sep)
        v=[]
        for i in x_axis:
            v.append(round(dp.mean(dfi.loc[dfi_mc_column.contains(i),value_column]),2))    
        value.append(v)
    
    dfv= pd.DataFrame(np.array(value).T)
    dfv[mc_column]=x_axis
    dfv=dfv.sort_values(by=4,ascending=False)
    
    # draw chart       
    from pyecharts import Bar
    chart_title = 'Profits of movie from different genres over years'
    
    bar =  Bar(chart_title, title_pos='center')
    for i in range(len(value)):
        bar.add(year_labels[i], dfv[mc_column], dfv[i], mark_line=['average'], mark_point=['max'],
                is_more_utils=True, is_datazoom_show=True, datazoom_type='both', 
                 legend_top='7%', yaxis_name=value_column, yaxis_name_gap=40)
    
    bar.render(chart_title + '.html')


#----------------------------------------------------------------------------
# bar3d
    
    # process data
    value = []
    for i in year_labels:
        dfi = df.loc[df[cut_column+'_level'] == i,:]
        dfi_mc_column = pde.multi_content_column(dfi[mc_column], sep=mc_sep)
        v=[]
        for i in x_axis:
            v.append(round(dp.mean(dfi.loc[dfi_mc_column.contains(i),value_column]),2))   
        value.append(v)
     
    dfv= pd.DataFrame(np.array(value).T)
    dfv[mc_column]=x_axis
    dfv=dfv.sort_values(by=4,ascending=False)
    
    value=[[] for _ in range(dfv.shape[1]-1)]
    for i in range(dfv.shape[0]):
        for j in range(dfv.shape[1]-1):
            value[j].append([i,j,dfv.iloc[i,j]])
    dfv
    pprint(value[0])


    from pyecharts import Bar3D
    
    title = 'Profits of movie from different genres over years'
    bar3d = Bar3D(title, title_pos='center')
    
    for j in range(dfv.shape[1]-1):
        pprint(value[j])
        bar3d.add(year_labels[j], dfv[mc_column], year_labels, value[j],
                  is_legend_show=True, legend_top='7%',
                  zaxis3d_name=value_column, grid3d_opacity=1,  xaxis3d_interval=0, 
                  grid3d_width=200, grid3d_depth=60, 
                  grid3d_shading='realistic', is_grid3d_rotate=True
                  )
    print('rendering')
    bar3d.render(title+'(Bar3D).html')




df.columns
dfr=pde.r_table(df.content_rating)
print(dfr)
dfr.level[:5]






##################### step 3 #########################
# draw some heatmap    
from copy import copy




# 00 value column            
df['profits'] = [(i-j)/1000000 for i,j in zip(df.gross,df.budget)]
value_column = 'profits'

# 01 attr column 1
cut_column = 'facenumber_in_poster'
x_attr = cut_column+'_level'
df[x_attr] = list(pd.cut(df[cut_column], bins=[0,1,2,4,8,50], labels=["small","medium-small","medium","medium-large","large"]))

# 02 attr column 2
mc_column = 'genres'
mc_sep = '|'
df_mc_column = pde.multi_content_column(df[mc_column], sep=mc_sep)
df['diversity'] = df_mc_column.list_levels


dfv=pde.pivot_table_df(df, x_attr, 'diversity', value_column, aggfunc=dp.mean)

SORT_ORDER = {"small": 0, "medium-small": 1, "medium": 2, 'medium-large':3,'large':4}
dfv.index = [SORT_ORDER[i] for i in dfv[x_attr]]
dfv = dfv.sort_index()


x_axis = copy(dfv[x_attr])
del dfv[x_attr]
data = pde.df_to_hm_data(dfv)

from pyecharts import HeatMap
title = 'Profits of movie with different face number on poster and diversity'
heatmap = HeatMap(title, title_pos='center')
heatmap.add('', x_axis, dfv.columns, data, 
                xaxis_interval = 0,
                xaxis_label_textsize=10,
                xaxis_name_gap=30,
                yaxis_label_textsize=10,
                yaxis_name_gap=30,
                yaxis_rotate = 0,
                is_visualmap=True,
                visual_range=[np.nanmin([x[2] for x in data]), np.nanmax([x[2] for x in data])],
            visual_text_color="#000", visual_orient='horizontal')
heatmap.render(title+'.html')



##################### step 4 #########################
# box-diagram

# 01 attr column 1

y_attr = 'content_rating'
dfr=pde.r_table(df[y_attr])
print(dfr.level[:5])
x_axis = ['G','PG','PG-13','R','Not Rated']

x_axis2 = ["bad","below average","average","good","very good"]

cut_column = 'imdb_score'
x_attr = cut_column+'_level'
df = pde.quantile_cut_column(df,cut_column, labels=x_axis2)

data = []
for i in x_axis2:
    v=[]
    for j in x_axis:      
        dfv = pde.drop_outliers(df.loc[(df[x_attr]==i) & (df[y_attr]==j),:], ['profits'])  # hack
        v.append(list(dfv['profits']))
    data.append(v)

from pyecharts import Boxplot
title = 'Profits of movie with different rating and score'
boxplot = Boxplot(title, title_pos='center')

p = 0
for i in x_axis2:
    boxplot.add(i, x_axis, boxplot.prepare_data(data[p]), legend_top='7%')
    p += 1

boxplot.render(title+'.html')




#############################################################################
################   smart heatmap  ###########################################

from pyecharts import HeatMap
from pyecharts import Page

def ref_list_maker(df, attr1, attr2):
        # df is a grouped table
        x_axis = list(set(df[attr1]))
        y_axis = list(set(df[attr2]))
        ref_list=[]
        for i in x_axis:
            y_axis_2 = list(set(df.loc[df[attr1]==i, attr2]))
            if y_axis_2:
                v=np.ma.median([ float(df.loc[(df[attr1]==i) & (df[attr2]==j), df.columns[-1]]) for j in y_axis_2])
            else:
                v=np.nan
            ref_list.append(v)
        return ref_list
    
def custom_sort(l, order_type=0, ref_list=[0]):
    if order_type==0:
        l_2 = l
        
    elif order_type==1:
        l_2 = sorted(l)
        
    elif order_type==2:
        SORT_ORDER = {"small": 0, "medium-small": 1, "medium": 2, 'medium-large':3,'large':4}
        l_2 = sorted(l, key=lambda val: SORT_ORDER[val])
        
    elif order_type==3:
        l_2 = [i for j,i in sorted(zip(ref_list,l))]
        
    return l_2 

def ordered_attr(df, attr1, attr2, order_type):
    x_axis = list(set(df[attr1]))    
    if order_type==3:
        ref_list = ref_list_maker(df, attr1, attr2)
        x_axis = custom_sort(list(set(df[attr1])), 3, ref_list=ref_list)
    else:
        x_axis = custom_sort(list(set(df[attr1])), order_type)
    return x_axis

def heatmap_2attr(df, attr1, attr2, func=len, value=None, order_type=[0,0]):
    if value==None:
        df2=df.groupby([attr1, attr2])[attr1].agg([func])
    else:
        df2=df.groupby([attr1, attr2])[value].agg([func])
    
    df2=df2.reset_index()
    
    x_axis = ordered_attr(df2, attr1, attr2, order_type[0])
    y_axis = ordered_attr(df2, attr2, attr1, order_type[1])
    data = [[x_axis.index(i), y_axis.index(j), df2.loc[(df2[attr1]==i) & (df2[attr2]==j), func.__name__]] 
            for i in x_axis for j in y_axis]
    
    heatmap = HeatMap()
    heatmap.add("热力图直角坐标系", x_axis, y_axis, data, 
                xaxis_name=attr1,
                yaxis_name=attr2,
                xaxis_interval = 0,
                xaxis_label_textsize=10,
                xaxis_name_gap=30,
                yaxis_label_textsize=10,
                yaxis_name_gap=30,
                yaxis_rotate = 90,
                is_visualmap=True,
                visual_range=[min(df2[func.__name__]), max(df2[func.__name__])],
            visual_text_color="#000", visual_orient='horizontal')
    return heatmap 

def heatmap_2attr_p(df, attr1, attr2, func=len, value=None, order_type=[0,0]):
    # heatmap and data amount inside the sticks
    
    if value==None:
        df2=df.groupby([attr1, attr2])[attr1].agg([func])
    else:
        df2=df.groupby([attr1, attr2])[value].agg([func])
    
    df2=df2.reset_index()
    
    x_axis = ordered_attr(df2, attr1, attr2, order_type[0])
    y_axis = ordered_attr(df2, attr2, attr1, order_type[1])
    data = [[x_axis.index(i), y_axis.index(j), df2.loc[(df2[attr1]==i) & (df2[attr2]==j), func.__name__]] 
            for i in x_axis for j in y_axis]
    
    p=Page()
    heatmap = HeatMap()
    heatmap.add("热力图直角坐标系", x_axis, y_axis, data, 
                xaxis_name=attr1,
                yaxis_name=attr2,
                xaxis_interval = 0,
                xaxis_label_textsize=10,
                xaxis_name_gap=30,
                yaxis_label_textsize=10,
                yaxis_name_gap=30,
                yaxis_rotate = 90,
                is_visualmap=True,
                visual_range=[min(df2[func.__name__]), max(df2[func.__name__])],
            visual_text_color="#000", visual_orient='horizontal')
    p.add(heatmap)
    
    df2=df.groupby([attr1, attr2])[attr1].agg([len])
    df2=df2.reset_index()
    data = [[x_axis.index(i), y_axis.index(j), df2.loc[(df2[attr1]==i) & (df2[attr2]==j), len.__name__]] 
            for i in x_axis for j in y_axis]
    heatmap = HeatMap()
    heatmap.add("热力图直角坐标系", x_axis, y_axis, data, 
                xaxis_name=attr1,
                yaxis_name=attr2,
                xaxis_interval = 0,
                xaxis_label_textsize=10,
                xaxis_name_gap=30,
                yaxis_label_textsize=10,
                yaxis_name_gap=30,
                yaxis_rotate = 90,
                is_label_show=True,
                label_pos='inside',
                is_visualmap=True,
                visual_range=[min(df2[len.__name__]), max(df2[len.__name__])],
            visual_text_color="#000", visual_orient='horizontal')
    p.add(heatmap)
    
    return p
    
p=heatmap_2attr_p(df, 'actor_1_facebook_likes_level','facenumber_in_poster_level', func=np.mean, value='gross', order_type=[2,2])
p.render()


h2=heatmap_2attr(df, 'actor_1_facebook_likes_level','facenumber_in_poster_level',func=np.mean, value='gross', order_type=[3,3])
   

heatmap_2attr(df, 'actor_1_facebook_likes_level','facenumber_in_poster_level',func=np.mean, value='gross', order_type=[3,3])
p=heatmap_2attr_p(df, 'actor_1_facebook_likes_level','facenumber_in_poster',func=np.ma.median, value='gross', order_type=[2,1])
p.render()




def random_cut_df(df):
    float_c = pde.choose_columns(df, 'float64')
    float_c.remove('gross')
    attr = random.sample(float_c, 2)
    print(attr)
    df2 = pde.quantile_cut_column(df, attr[0])
    df2 = pde.quantile_cut_column(df, attr[1])
    p=heatmap_2attr_p(df2, attr[0]+'_level', attr[1]+'_level', func=np.mean, value='gross', order_type=[2,2])
    return p

def make_some_heatmaps(df):
    for i in range(10): 
        p=random_cut_df(df)
        p.render('page_{}.html'.format(i))



##################################### step 4 ##############################################
# scatter

from pyecharts import Scatter

attr1='budget'
df = pde.quantile_cut_column(df, attr1)
g_axis = ["small","medium-small","medium","medium-large","large"]

bubble = Scatter()
for i in g_axis:
    dfv = df.loc[df[attr1+'_level']==i,:]
    vx = dfv['director_facebook_likes']
    vy = dfv['cast_total_facebook_likes']
    vz = dfv['imdb_score']
    print(vx,vy,vz)

    bubble.add(i, list(vx), list(vy), extra_data=list(vz), is_visualmap=True, visual_type='size',
               visual_range=[np.nanmin(df['imdb_score']), np.nanmax(df['imdb_score'])])
    
bubble.render()


def RGB(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

from pyecharts import Scatter3D

attr1='budget'
df = pde.quantile_cut_column(df, attr1)
g_axis = ["small","medium-small","medium","medium-large","large"]

title = 'Scatter of different movie from different budget level'
bubble = Scatter3D(title, title_pos='center')
for i in g_axis:
    dfv = df.loc[df[attr1+'_level']==i,:]
    attrs = ['director_facebook_likes','cast_total_facebook_likes','profits']
    print(dfv.shape)
    dfv = pde.drop_outliers(dfv[attrs], attrs)

    bubble.add(i, dfv.values, is_visualmap=True,
                   xaxis3d_name=attrs[0],
                   yaxis3d_name=attrs[1],
                   zaxis3d_name=attrs[2],
                   legend_top='7%',
                   # is_grid3d_rotate=True,
               visual_range=[np.nanmin(dfv[attrs[2]]), np.nanmax(dfv[attrs[2]])], symbol_size=5, label_color=[RGB(100,100,100)])

bubble.render(title + '.html')


#############################################################################
dfv=df.loc[:,['imdb_score','profits','actor_1_facebook_likes','cast_total_facebook_likes','director_facebook_likes','diversity','num_user_for_reviews']]
sns.heatmap(dfv.corr(), annot=True)



################################# step 5 ##################################################
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale



df['cast_FL_except_a1']=[(i-j) for i,j in zip(df['cast_total_facebook_likes'],df['actor_1_facebook_likes'])]

attrs = ['actor_1_facebook_likes','cast_FL_except_a1','director_facebook_likes']
dfv = pde.drop_outliers(df, attrs, level=6)
print(dfv.shape)

data = dfv[attrs].values

data = scale(data)
n_c = 6
model1 = KMeans(n_c)
model1.fit(data)

cluster_dic={0:'good_cast', 1: 'famous_actor_1', 2:'unpopular', 3:'extrodinary_cast', 4:'famous_director', 5: 'normal', np.nan:'extra'}
dfv['fl_cluster'] = [cluster_dic[i] for i in model1.labels_]


title = 'Clustering of facebook likes'
bubble2 = Scatter3D(title, title_pos='center')
for i in range(n_c):
    data = dfv.loc[model1.labels_ == i,attrs].values
    bubble2.add(cluster_dic[i],data, symbol_size=3,
                xaxis3d_name=attrs[0],
                yaxis3d_name=attrs[1],
                zaxis3d_name=attrs[2],
                legend_top='7%'
                )
bubble2.render(title+'.html')

attrs=['budget','gross','imdb_score']
dfv2 = dfv.groupby('fl_cluster')['budget','gross','imdb_score'].agg(dp.mean)

title = 'Scatter of different types of movie'
bubble2 = Scatter(title, title_pos='center')
for i in range(n_c):
    print(dfv2.loc[cluster_dic[i],:])
    bubble2.add(cluster_dic[i], 
                
                x_axis=[dfv2.loc[cluster_dic[i],attrs[0]]], 
                y_axis=[dfv2.loc[cluster_dic[i],attrs[1]]], 
                extra_data=[dfv2.loc[cluster_dic[i],attrs[2]]], 
                xaxis_name=attrs[0],
                yaxis_name=attrs[1],
                
                is_visualmap=True, visual_type='size', visual_orient='horizontal',
                visual_range=[np.nanmin(dfv2[attrs[2]]), np.nanmax(dfv2[attrs[2]])],
                legend_top='7%')
    
bubble2.render(title+'.html')



######################################################################################
g_axis = ["bad","below average","average","good","very good"]

cut_column = 'imdb_score'
x_attr = cut_column+'_level'
df = pde.quantile_cut_column(df,cut_column, labels=g_axis)

dfv = df.groupby(x_attr)['director_facebook_likes', 'actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','cast_FL_except_a1'].agg(dp.mean)




def pos_assign(n):
    return [[(i+1)*float(100/(n+1)),50] for i in range(n)]

from pyecharts import Pie
title = 'Pie chart of FB likes of different levels of movie'
pie =  Pie(title, title_pos='center',width=800)
positions = pos_assign(len(g_axis))
p=0
for i in g_axis:
    print(dfv.loc[i,:])
    pie.add(i, dfv.columns, dfv.loc[i,:], center=positions[p], radius = [15, 30], is_legend_show=True, legend_top='7%')
    p+=1

pie.render(title+'.html')


#################################################################################
attrs=['profits','imdb_score']
dfv=pde.drop_outliers(df, attrs, level=5)

title = 'Relationship between profits and imdb_scores'
s=Scatter(title, title_pos='center')
s.add('',dfv.profits, dfv.imdb_score, extra_data=list(dfv.imdb_score), 
      is_visualmap=True, visual_range=[np.nanmin(dfv.imdb_score), np.nanmax(dfv.imdb_score)])
s.render(title+'.html')


##################################################################################
from pyecharts import Sankey

cut_column = 'imdb_score'
attr1 = cut_column+'_level'
labels_1 = ["bad","below average","average","good","very good"]
df = pde.quantile_cut_column(df,cut_column, labels=labels_1)


column_name='profits'
attr2 = column_name+ '_level'
bins = [-13000, -10, 0, 10, 30, 600]
labels_2 = ['tragedy', 'lose', 'working', 'big sale', 'money tree']
df[column_name + '_level']=list(pd.cut(df[column_name], bins=bins, labels=labels_2))

dfv=df.groupby([attr1, attr2])[attr1].agg([len])

dfv=dfv.reset_index()
print(dfv)

nodes = []
for i in labels_1:
        nodes.append({'name': attr1+'_'+str(i)})
for i in labels_2:
        nodes.append({'name': attr2+'_'+str(i)})


links = []
for i in range(dfv.shape[0]):
     links.append({
                    'source':attr1+'_'+str(dfv.loc[i,attr1]),
                    'target':attr2+'_'+str(dfv.loc[i,attr2]),
                    'value':dfv.loc[i,'len'],
                    })
print(nodes)
print(links)

title = 'Sankey relation between imdb score and profits'
sankey = Sankey(title, title_pos='center', width=1200, height=600)
sankey.add("sankey", nodes, links, line_opacity=0.7,
           line_curve=0.5, line_color='source', is_legend_show=False,
           is_label_show=True, label_pos='right', is_random=True)
sankey.render(title+ '.html')

#####################################################################################
attr = 'title_year'
bins = [1900]
bins.extend(range(1971,2017,5))
bins
labels = ['<1975']
labels.extend([str(i)+'-'+str(i+5) for i in range(1971, 2016, 5)])
labels
df[attr+'_level']=pd.cut(df['title_year'], bins=bins, labels=labels)
df['gross(billion)']=[i/1000000000 for i in df['gross']]
dfv=df.groupby(attr+'_level')['gross(billion)'].agg(sum)
dfv

from pyecharts import Line

title = 'Gross improvement among countries'
l=Line(title, title_pos='center')

l.add('all', x_axis=list(dfv.index), y_axis=dfv.values, xaxis_interval=0, is_fill=True,
        area_opacity = 0.3, legend_top='7%')

dfv2=pde.r_table(df['country'])

for i in dfv2.level[:5]:
    dfv=df.loc[df['country']==i,:].groupby(attr+'_level')['gross(billion)'].agg(sum)
    l.add(i, x_axis=list(dfv.index), y_axis=dfv.values, xaxis_interval=0, is_fill=True,
        area_opacity = 0.3,legend_top='7%')
l.render(title+'.html')





###############################################################################
from pyecharts import Graph

import itertools

dfv=df.loc[df['title_year'] > 2013, :]

connection = ['actor_'+str(i+1)+'_name' for i in range(3)]

# generatre nodes and links
nodes = []

links = pd.DataFrame({'source': [],'target': []})
for i in range(dfv.shape[0]):
    for k,l in itertools.combinations(connection, 2):
        if {'name': dfv.iloc[i][k]} not in nodes:
            nodes.append({'name': dfv.iloc[i][k]})
        links.loc[links.shape[0]+1] = [dfv.iloc[i][k], dfv.iloc[i][l]]

dfv2 = links.groupby(['source','target']).agg(len)
dfv2 = dfv2.reset_index()
links_v = []
for i in range(dfv2.shape[0]):
    links_v.append({'source': dfv2.iloc[i]['source'], 'target': dfv2.iloc[i]['target'], 'value': dfv2.iloc[i][0]})

print(len(nodes))
print(len(set([links_v[i]['source'] for i in range(len(links_v))] + [links_v[i]['target'] for i in range(len(links_v))])))

print(nodes[:10])
print(links_v[:10])

print('rendering....')

title='Connections between actors'
g=Graph(title, title_pos='center')
g.add('', nodes, links_v, 
      graph_edge_length=[10, 100],
      graph_gravity=0.1
      )
g.render(title+'.html')

#-----------------------------------------------------------------------
from pyecharts import Graph

import itertools

dfv=df.loc[df['title_year'] > 2011, :]

connection = ['actor_'+str(i+1)+'_name' for i in range(3)]
fl = ['actor_'+str(i+1)+'_facebook_likes' for i in range(3)]

df_fl = pd.DataFrame({'actor': [],'FL': []})
for i in range(3):
    df0 = pd.DataFrame({'actor': dfv.loc[:, connection[i]], 'FL': dfv.loc[:, fl[i]]})
    df_fl = pd.concat([df_fl, df0], axis=0)

df_fl=df_fl.drop_duplicates('actor')
# hack : how do you deal with the duplicate name problem

df_fl=pde.quantile_cut_column(df_fl, 'FL', labels= [0,1,2,3,4])
# print(df_fl)

# generatre nodes and links
# nodes = [{'name': df_fl.iloc[i]['actor']} for i in range(df_fl.shape[0])]
nodes = [{'name': df_fl.iloc[i]['actor'], 'category': df_fl.iloc[i]['FL_level']} for i in range(df_fl.shape[0])]


links = pd.DataFrame({'source': [],'target': []})
for i in range(dfv.shape[0]):
    for k,l in itertools.combinations(connection, 2):
        links.loc[links.shape[0]+1] = [dfv.iloc[i][k], dfv.iloc[i][l]]

dfv2 = links.groupby(['source','target']).agg(len)
dfv2 = dfv2.reset_index()
links_v = []
for i in range(dfv2.shape[0]):
    links_v.append({'source': dfv2.iloc[i]['source'], 'target': dfv2.iloc[i]['target'], 'value': dfv2.iloc[i][0]})



# names = set([links_v[i]['source'] for i in range(len(links_v))] + [links_v[i]['target'] for i in range(len(links_v))])
# print(names)
# print(len(names))
# print(len(nodes))
# print(len(links_v))



print('rendering....')


title='Connections between actors'
g=Graph(title, title_pos='center')
g.add('', nodes, links_v, categories = [nodes[i]['category'] for i in range(len(nodes))],
      graph_edge_length=[10, 100],
      graph_gravity = 0.1,
      graph_repulsion = 200,
      is_legend_show = True,
      is_label_show = True,
      label_color = ['#313695','#4575b4','#fee090','#f46d43', '#a50026']
      )
g.render(title+'.html')

################################################################################
cut_column = 'imdb_score'
attr1 = cut_column+'_level'
labels_1 = ["bad","below average","average","good","very good"]
df = pde.quantile_cut_column(df,cut_column, labels=labels_1)

df_mcc = pde.multi_content_column(df.loc[df[attr1]=='very good','plot_keywords'],'|')
wl = df_mcc.list_classes('freq')[:50]


from pyecharts import WordCloud

wc = WordCloud()
wc.add("", [i for i,j in wl], [j for i,j in wl], word_size_range=[15, 100],
              shape='diamond', is_random=True)
wc.render()


