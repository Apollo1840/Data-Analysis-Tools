# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

%matplotlib qt


'''
    1   line and dots
    2   basic adjustments
    3   basic operations
    4   more charts
    5 animation
'''




# --------------------------------------------------------------------------
# 1 line and dots
x = np.random.randn(20)*10 + 20
x = np.sort(x)
x = x.reshape(10,2)
print(x)
plt.plot(x[:,0], x[:,1], 'b')
plt.plot(x[:,0], x[:,1], 'ro')
plt.show()









# --------------------------------------------------------------------------
# 2 basic adjustments

# 2.1 more types and add labels
plt.plot(x[:,0], x[:,1], 'b--', label='line')
plt.plot(x[:,0], x[:,1], 'r^', label='dots')
plt.legend()
plt.show()

# 2.2 subplots
plt.subplot(121)
plt.plot(x[:,0], x[:,1], label='line')
plt.subplot(122)
plt.plot(x[:,0], x[:,1], label='dots')
plt.show()


# axis
# use this to display coefficients
plt.plot(np.random.randn(1,100).T, 'o', label='first trial')
plt.plot(np.random.randn(1,100).T, '^', label='second trial')
plt.xticks(range(10), [i for i in 'abcdefghlj'],rotation=90)
plt.show()
    
# use this to show the scale of features
plt.plot(np.random.randn(100,10).min(axis=0), 'v')
plt.plot(np.random.randn(100,10).max(axis=0), '^')
plt.xticks(range(10), [i for i in 'abcdefghlj'],rotation=90)
plt.yscale('log')
plt.show()





# --------------------------------------------------------------------------
# 3 basic operations
plt.style.use('ggplot')
plt.plot([1,2,3,5,2,4,1,2.7])
plt.savefig('matplot_example.jpg')
plt.show()





# --------------------------------------------------------------------------
# 4 more charts

# 4.1 matshow
plt.matshow([[True,False,True,True,False]], cmap='gray_r')
plt.matshow([[1,2,3],[4,5,6]], cmap='gray_r')
plt.colorbar()
plt.show()


# 4.2 barchart
plt.subplot(121)
plt.bar(range(10), np.random.randn(10))
plt.subplot(122)
plt.barh(range(10), np.random.randn(10))
plt.show()






# --------------------------------------------------------------------------
# 5 animation
import matplotlib.animation as ani


'''
    Here introduces how to draw animation by matplotlib through ani.FuncAnimation
    1) define a function return graph objects in a list 
    2) 

'''

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True)

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani_1 = ani.FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True, interval=100)  # 100 means 100ms
plt.show()


# more

def update1(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    return plt.plot(xdata, ydata, 'ro')

def update2(frame)
    ln, = plt.plot(xdata, ydata, 'ro')
    ln2, = plt.plot(xdata, [-i for i in ydata], 'bo')
    return ln,ln2

ani_2 = ani.FuncAnimation(fig, update2, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True, interval=100)  # 100 means 100ms
plt.show()


# update graph
def animate(i):
    graph_data = open('example.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line)>1:
            x,y = line.split(',')
            xs.append(x)
            ys.append(y)
    
    ax1.clear()
    ax1.plot(xs,ys)
        
def dynamic_update_graph():
    fig = plt.figure() 
    ax1 = fig.add_subplot(1,1,1)
    ani_1 = ani.FuncAnimation(fig, animate, interval=1000)  # 1000 means 1000ms
    plt.show()


    
    
    

    
    



if __name__ == '__main__':

  
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# 柱状图 
plt.title(u"获救情况 (1为获救)") # 标题
plt.ylabel(u"人数")  

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') 
plt.title(u"按年龄看获救分布 (1为获救)")


plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")# plots an axis lable
plt.ylabel(u"密度") 
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")  
plt.show()  


 #然后我们再来看看各种舱级别情况下各性别的获救情况
fig=plt.figure()
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title(u"根据舱等级和性别的获救情况")

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
ax1.legend([u"女性/高级舱"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')

plt.show()
