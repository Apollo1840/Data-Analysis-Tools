# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:20:50 2018

@author: zouco
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# %matplotlib qt

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

def visualization_of_data():
    plt.matshow([[True,False,True,True,False]], cmap='gray_r')
    plt.matshow([[1,2,3],[4,5,6]], cmap='gray_r')
    plt.colorbar()
    plt.show()
    
def how_to_use_label():
    
    # use this to choose parameter
    plt.plot(range(10), np.random.rand(10), label='first trial')
    plt.plot(range(10), np.random.rand(10), label='second trial')
    plt.legend()
    plt.show()

def xaxis_type():
    
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
    


def well_known_charts(): 
    
    # bar chart
    plt.subplot(121)
    plt.bar(range(10), np.random.randn(10))
    plt.subplot(122)
    plt.barh(range(10), np.random.randn(10))
    plt.show()
    
if __name__ == '__main__':
    plt.style.use('ggplot')
    plt.plot([1,2,3,5,2,4,1,2.7])
    plt.savefig('matplot_example')
    plt.show()
    
    plt.plot([12,3,2,4,9,12],'o')
    plt.xticks(range(6), ['a','b','c','d','e','f'], rotation = 90)
    plt.show()
    
    