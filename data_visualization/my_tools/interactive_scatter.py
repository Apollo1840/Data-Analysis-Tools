import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')


class MontainPoint:
    def __init__(self, x, y, hue=None, signal=None):
        self.x = x
        self.y = y
        self.hue = hue
        self.signal = signal


def mountainborn_scatter_simple(x, y, data, signals):
    def on_pick(event):
        # print(event.artist.obj.x)
        axs[1].cla()
        axs[1].plot(event.artist.obj.signal)
        plt.show()

    assert len(data[x]) == len(signals)

    x_values = list(data[x])
    y_values = list(data[y])
    points_objs = []
    for i in range(len(data[x])):
        points_objs.append(MontainPoint(x=x_values[i],
                                        y=y_values[i],
                                        signal=signals[i]))

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    for obj in points_objs:
        artist = axs[0].plot(obj.x, obj.y, 'ro', picker=5)[0]
        artist.obj = obj

    fig.canvas.callbacks.connect('pick_event', on_pick)

    plt.show()


def mountainborn_scatter(x, y, hue=None, data=None, signals=None):
    def on_pick(event):
        # print(event.artist.obj.x)
        axs[1].cla()
        axs[1].plot(event.artist.obj.signal)
        plt.show()

    if signals is not None:
        assert len(data[x]) == len(signals)

    if data is not None:
        x_values = list(data[x])
        y_values = list(data[y])

    else:
        x_values = x
        y_values = y

    if hue is None:
        hues = ["ro" for _ in range(len(x_values))]
    elif data is None:
        hues = hue
    else:
        hues = list(data[hue])

    if signals is not None:
        assert len(x_values) == len(signals)
    else:
        signals = [(x, y) for x, y in zip(x_values, y_values)]

    points_objs = []
    for i in range(len(x_values)):
        points_objs.append(
            MontainPoint(x=x_values[i],
                         y=y_values[i],
                         hue=hues[i],
                         signal=signals[i]))

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    for obj in points_objs:
        artist = axs[0].plot(obj.x, obj.y, obj.hue, picker=5)[0]
        # explanation of picker: https://matplotlib.org/3.1.1/gallery/event_handling/pick_event_demo.html
        artist.obj = obj

    fig.canvas.callbacks.connect('pick_event', on_pick)

    plt.show()


if __name__ == "__main__":
    # prepare data
    points = np.random.rand(10, 2)
    df = pd.DataFrame(points, columns=["x", "y"])

    mountainborn_scatter("x", "y", df, points)
