import matplotlib.pyplot as plt
import numpy as np

# %matplotlib tk


class Cursor:
    def __init__(self, ax):
        self.ax = ax

        self.lx = ax.axhline(color='k', alpha=0.2)  # the horiz line
        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line

        self.clicks = []
        self.scatter_clicks, = ax.plot([], [], linestyle="none", marker="o", color="r")

        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata

        # update the line positions and the text
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))

        self.ax.figure.canvas.draw()

    def mouse_click(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        self.clicks.append((x, y))

        # update the scatter
        if len(self.clicks)>=1:
            xs, ys = zip(*self.clicks)
            self.scatter_clicks.set_data(xs, ys)

        self.ax.figure.canvas.draw()


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cursor = Cursor(ax)
    fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
    fig.canvas.mpl_connect('button_press_event', cursor.mouse_click)
    plt.show()