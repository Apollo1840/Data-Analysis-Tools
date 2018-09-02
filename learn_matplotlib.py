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

    