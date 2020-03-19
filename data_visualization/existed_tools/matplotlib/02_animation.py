# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:26:35 2018

@author: zouco
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# ----------------------------------------------
# basics
xdata, ydata = [], []


def animate(frame):
    # frame will varies from 0 to inf as integer
    xdata.append(frame)
    ydata.append(np.sin(frame))

    plt.plot(xdata, ydata)  # this plot will plt on old fig


fig = plt.figure()
the_animation = ani.FuncAnimation(fig, animate, interval=1000)  # 1000 means 1000ms

plt.show()

# --------------------------------------------------------------
# basics2: using axis instead
xdata, ydata = [], []


def animate(frame):
    # frame will varies from 0 to inf as integer
    xdata.append(frame)
    ydata.append(np.sin(frame))

    ax.clear()
    ax.plot(xdata, ydata)

    ax2.clear()
    ax2.plot(xdata, [(-1) * y for y in ydata])


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
the_animation = ani.FuncAnimation(fig, animate, interval=1000)  # 1000 means 1000ms

plt.show()

# advance:

# --------------------------------------------------------------
# use .set_data() to update axis

xdata, ydata = [], []


def init():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    lines, = plt.plot([], [], 'ro', animated=True)
    return fig, lines


def animate(frame):
    xdata.append(frame * np.pi / 50)
    ydata.append(np.sin(frame * np.pi / 50))
    lines.set_data(xdata, ydata)


fig, lines = init()
the_animation = ani.FuncAnimation(fig,
                                  animate,
                                  interval=100)  # 100 means 100ms

plt.show()

# ----------------------------------------------------------
# customize frames

xdata, ydata = [], []


def init():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    lines, = plt.plot([], [], 'ro', animated=True)
    return fig, lines


def animate(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    lines.set_data(xdata, ydata)


fig, lines = init()
the_animation = ani.FuncAnimation(fig,
                                  animate,
                                  frames=np.linspace(0, 2 * np.pi, 128),
                                  interval=100)  # 100 means 100ms

plt.show()

# ------------------------------------------------------
# use blit


# blit: The blit keyword is an important one:
# this tells the animation to only re-draw the pieces of the plot which have changed.
# The time saved with blit=True means that the animations display much more quickly.

xdata, ydata = [], []


def init():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    return fig, ax


def animate(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    lines, = ax.plot(xdata, ydata, 'bo')
    return lines,


fig, ax = init()
ani_2 = ani.FuncAnimation(fig,
                          animate,
                          frames=np.linspace(0, 2 * np.pi, 128),
                          blit=True,
                          interval=100)  # 100 means 100ms

plt.show()

# plot 2 lines

xdata, ydata = [], []


def init():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    return fig, ax


def animate(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))

    ax.plot(xdata, ydata, 'ro')
    lines, = ax.plot(xdata, [-i for i in ydata], 'bo')
    return lines,


fig, ax = init()
ani_2 = ani.FuncAnimation(fig,
                          animate,
                          frames=np.linspace(0, 2 * np.pi, 128),
                          blit=True,
                          interval=100)  # 100 means 100ms

plt.show()

# plot with .set_data()

xdata, ydata = [], []


def init():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    lines, = plt.plot([], [], 'ro', animated=True)
    return fig, lines


def animate(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    lines.set_data(xdata, ydata)
    return lines,


fig, lines = init()
the_animation = ani.FuncAnimation(fig,
                                  animate,
                                  frames=np.linspace(0, 2 * np.pi, 128),
                                  interval=100,  # 100 means 100ms
                                  blit=True)

plt.show()
