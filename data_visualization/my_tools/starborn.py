"""

extension of matplotlib, in seaborn API style.

mustly in interactive field

functionality:
    - scatter the interactive points

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')


class StarPoint:

    def __init__(self, x, y, hue=None, signal=None):
        self.x = x
        self.y = y
        self.hue = hue
        self.signal = signal

        # assigned externally
        self.info_dict = None

    def plot(self, axis):
        global fig
        axis.cla()
        axis.plot(self.signal)
        fig.canvas.draw_idle()


# An example of how to extend StarPoint
class HeartbeatPointObj(StarPoint):

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


def iscatter_simple(x, y, data, signals):
    """
    
    :param x: str
    :param y: str
    :param data: pd.DataFrame
    :param signals: list[list], list of signals row(x, y) present.
    """
    assert len(data[x]) == len(signals)

    x_values = list(data[x])
    y_values = list(data[y])
    star_points = []
    for i in range(len(data[x])):
        star_points.append(StarPoint(x=x_values[i],
                                     y=y_values[i],
                                     signal=signals[i]))

    return _scatter(star_points)


def iscatter(x, y, hue=None, data=None, signals=None):
    """

    :param x: str
    :param y: str
    :param data: pd.DataFrame
    :param signals: list[list], list of signals row(x, y) present.
    """

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
    star_points = []
    for i in range(len(x_values)):
        starp = StarPoint(x=x_values[i],
                          y=y_values[i],
                          hue=hues[i],
                          signal=signals[i])
        star_points.append(starp)

    return _scatter(star_points)


def _scatter(star_points):
    """
    
    :param: star_objs: List[StarPoint]
    """
    global fig

    # plot the first plot with star_points
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    for starp in star_points:
        artist = axs[0].plot(starp.x, starp.y, starp.hue, picker=True)[0]
        artist.set_pickradius(10)
        # explanation of picker: https://matplotlib.org/3.1.1/gallery/event_handling/pick_event_demo.html
        artist.obj = starp

    fig.canvas.mpl_connect('pick_event', lambda event: event.artist.obj.plot(axs[1]))

    plt.show()

    return axs


if __name__ == "__main__":
    # prepare data
    points = np.random.rand(10, 2)
    df = pd.DataFrame(points, columns=["x", "y"])

    iscatter("x", "y", data=df, signals=points)
