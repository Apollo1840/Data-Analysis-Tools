# -*- coding: utf-8 -*-

'''
    overview:

    1   line and dots
    2   basic adjustments
    3   basic operations
'''

import numpy as np
import matplotlib.pyplot as plt

# %matplotlib qt

# ---------------------------------------------------------------------------
# 0 prepare data to visualize
x = np.random.randn(20)*10 + 20
x = np.sort(x)
x = x.reshape(10, 2)
print(x)

# --------------------------------------------------------------------------
# 1 line and dots

# dim=1

x1 = np.random.randn(20, 1)*10
x2 = np.random.randn(20, 1)

plt.plot(x1, 'o', label='first trial')
plt.plot(x2, '^', label='second trial')
plt.xticks(range(10), [i for i in 'abcdefghlj'],rotation=90)
plt.yscale('log')
plt.show()

# dim=2

print(x)
plt.plot(x[:, 0], x[:, 1], 'b')
plt.plot(x[:, 0], x[:, 1], 'ro')

plt.xlabel('x value')
plt.ylim([0, 60])
plt.title('line and dots')
plt.grid(axis='y')
plt.show()


# --------------------------------------------------------------------------
# 2 basic adjustments

# 2.1 more types and add labels

plt.plot(x[:, 0], x[:, 1], 'b--', label='origin')
plt.plot(x[:, 1], x[:, 0], 'r^',  label='xy_reverse')
plt.legend()
plt.show()

# 2.2 subplots

plt.subplot(121)
plt.plot(x[:, 0], x[:, 1], 'b--')
plt.subplot(122)
plt.plot(x[:, 0], x[:, 1], 'r^')
plt.show()

# what happens behind: fig and axis
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, sharey=ax1)

ax1.plot(x[:, 0], x[:, 1], 'b--')
ax1.set_xlabel('hello')

ax2.plot(x[:, 1], x[:, 0]*(-5), 'r^')
ax2.set_xlabel('world')

fig.tight_layout() 
plt.show()

# also there is subplot2grid, which can make beautiful dashboard

ax1 = plt.subplot2grid((6, 2), (0, 0), rowspan=1)
ax2 = plt.subplot2grid((6, 2), (1, 0), rowspan=4)
ax3 = plt.subplot2grid((6, 2), (5, 1), rowspan=1)

ax1.plot(x[:, 0], x[:, 1], 'b--')
ax2.plot(x[:, 0], x[:, 1], 'b--')
ax3.plot(x[:, 0], x[:, 1], 'b--')

plt.show()


# --------------------------------------------------------------------------
# 3 basic operations
plt.style.use('ggplot')
plt.plot([1, 2, 3, 5, 2, 4, 1, 2.7])
plt.savefig('matplot_example.jpg')
plt.show()


