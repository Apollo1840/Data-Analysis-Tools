import matplotlib.pyplot as plt
import numpy as np

# %matplotlib qt

data = np.reshape(np.random.randn(90), (30,3))
print(data)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for x in data:
    ax.scatter(x[0], x[1], x[2], marker="o")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 0], marker="o")
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

# ----------------------------------------------
# 3D mesh


# Make data.
x = np.arange(1000)
y = np.arange(50)
X, Y = np.meshgrid(x, y)

# lambda (x, y)
Z = np.array(range(50000)).reshape(1000, 50)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap="magma",
                       linewidth=0,
                       antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# Make data.
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(x, y)

# lambda (x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap="magma",
                       linewidth=0,
                       antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()