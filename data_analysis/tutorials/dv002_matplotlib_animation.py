# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:26:35 2018

@author: zouco
"""



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
