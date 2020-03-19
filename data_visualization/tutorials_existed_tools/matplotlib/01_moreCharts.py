# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib qt

# ---------------------------------------------------------------------------
# 0 prepare data to visualize
x = np.random.randn(20)*10 + 20
x = np.sort(x)
x = x.reshape(10, 2)
print(x)

df = pd.read_csv('matplotlib_data0.csv')
df.head()

# --------------------------------------------------------------------------
# 1 matshow
plt.matshow([[True,False,True,True,False]], cmap='gray_r')
plt.matshow([[1,2,3],[4,5,6]], cmap='gray_r')
plt.colorbar()
plt.show()

# 2 barchart
plt.subplot(121)
plt.bar(range(10), np.random.randn(10))
plt.subplot(122)
plt.barh(range(10), np.random.randn(10))
plt.show()
