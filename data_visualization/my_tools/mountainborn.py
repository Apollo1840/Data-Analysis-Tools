import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')


class MontainPointObj:

    def __init__(self, x, y, hue=None, signal=None):
        self.x = x
        self.y = y
        self.hue = hue
        self.signal = signal

        # assigned externally
        self.info_dict = None

    def plot(self, axis):
        axis.cla()
        axis.plot(self.signal)
        plt.show()


class HeartbeatPointObj(MontainPointObj):

    def plot(self, axis):
        axis.cla()

        beat_ws = 500  # 500 from long_reference_beats
        beat_fs = 360
        axis.plot(list((np.arange(len(self.signal)) - beat_ws) * 1000 / beat_fs), self.signal, "-")
        axis.axvline(-90 * 1000 / beat_fs, color="r", ls=":")
        axis.axvline(90 * 1000 / beat_fs, color="r", ls=":")
        axis.set_title(
            "{} (pre_rr: {:.2f}, post_rr: {:.2f}, local_rr: {:.2f})".format(
                self.info_dict["threechar_label"],
                self.info_dict["pre_rr"],
                self.info_dict["post_rr"],
                self.info_dict["local_rr"])
        )
        plt.show()


def mountainborn_scatter_simple(x, y, data, signals):

    assert len(data[x]) == len(signals)

    x_values = list(data[x])
    y_values = list(data[y])
    mp_objs = []
    for i in range(len(data[x])):
        mp_objs.append(MontainPointObj(x=x_values[i],
                                       y=y_values[i],
                                       signal=signals[i]))

    return mountainborn_scatter(mp_objs)


def scatter(x, y, hue=None, data=None, signals=None):
    # todo: look seaborn scatter code to figure out how to implement hue

    # inputs handling

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

    # initialize moutain point objects
    mp_objs = []
    for i in range(len(x_values)):
        mp_obj = MontainPointObj(x=x_values[i],
                                 y=y_values[i],
                                 hue=hues[i],
                                 signal=signals[i])
        mp_objs.append(mp_obj)

    return mountainborn_scatter(mp_objs)


def mountainborn_scatter(mp_objs):

    # plot the first plot with mp_objs
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    for mp_obj in mp_objs:
        artist = axs[0].plot(mp_obj.x, mp_obj.y, mp_obj.hue, picker=5)[0]
        # explanation of picker: https://matplotlib.org/3.1.1/gallery/event_handling/pick_event_demo.html
        artist.obj = mp_obj

    fig.canvas.callbacks.connect('pick_event', lambda event: event.artist.obj.plot(axs[1]))

    plt.show()

    return axs


if __name__ == "__main__":
    # prepare data
    points = np.random.rand(10, 2)
    df = pd.DataFrame(points, columns=["x", "y"])

    scatter("x", "y", df, points)
