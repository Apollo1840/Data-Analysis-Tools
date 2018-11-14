# -*- coding: utf-8 -*-



import matplotlib.pyplot as plt

%matplotlib qt


'''
    1   line and dots
    2   basic adjustments
    3   basic operations
    4   more charts
    5 animation
'''

# ---------------------------------------------------------------------------
# 0 data
import numpy as np
x = np.random.randn(20)*10 + 20
x = np.sort(x)
x = x.reshape(10,2)
print(x)

import pandas as pd
df = pd.read_csv('Tutorials\\matplotlib_data0.csv')
df.head()


# --------------------------------------------------------------------------
# 1 line and dots
plt.plot(x[:,0], x[:,1], 'b')
plt.plot(x[:,0], x[:,1], 'ro')
plt.xlabel('x value')
plt.ylim([0,60])
plt.title('line and dots')
plt.grid(axis='y') 
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
plt.plot(x[:,0], x[:,1], 'b--')
plt.subplot(122)
plt.plot(x[:,0], x[:,1], 'r^')
plt.show()


# make a fig object
fig=plt.figure()

ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122, sharey=ax1)

ax1.plot(x[:,0], x[:,1], 'b--')
ax1.set_xlabel('ax1')

ax2.plot(x[:,0], x[:,1]*(-5), 'r^')
ax2.set_xlabel('ax2')

fig.tight_layout() 
plt.show()

# also there is subplot2grid, which can make beautiful dashboard
ax1=plt.subplot2grid((6,1),(0,0),rowspan=1)
ax2=plt.subplot2grid((6,1),(1,0),rowspan=4)
ax3=plt.subplot2grid((6,1),(5,0),rowspan=1)

ax1.plot(x[:,0], x[:,1], 'b--')
ax2.plot(x[:,0], x[:,1], 'b--')
ax3.plot(x[:,0], x[:,1], 'b--')

plt.show()





# axis
# use this to display coefficients
plt.plot(np.random.randn(1,10).T, 'o', label='first trial')
plt.plot(np.random.randn(1,10).T, '^', label='second trial')
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



