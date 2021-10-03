"""

extension of matplotlib, in seaborn API style.

mustly in interactive field

functionality:
    - scatter the interactive points

"""
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.plot_with_axis(axis)
        fig.canvas.draw_idle()

    def plot_with_axis(self, axis):
        axis.plot(self.signal)

    @classmethod
    def from_item(cls, item, x, y, func_plot):
        star = StarPoint(x=getattr(item, x), y=getattr(item, y))
        star.plot_with_axis = lambda axis: func_plot(item, axis)
        return star


# Two examples of how to extend StarPoint
class TitleStarPoint(StarPoint):

    def plot_with_axis(self, axis):
        axis.plot(self.signal)
        axis.set_title(self.info_dict["title"])


class HeartbeatStarPoint(StarPoint):

    def plot_with_axis(self, axis):
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


def iscatter(x, y, hue=None, data=None, signals=None, custom_star=StarPoint, info_dicts=None):
    """


    :param x: str
    :param y: str
    :param hue: str: only support categorical data.
    :param data: pd.DataFrame
    :param signals: list[list], list of signals row(x, y) present.
    :param custom_star: StarPoint,
    :param info_dicts: List[Dict]
    """

    # inputs handling

    if data is not None:
        x_values = list(data[x])
        y_values = list(data[y])

    else:
        x_values = x
        y_values = y

    if hue is None:
        hues = ["blue" for _ in range(len(x_values))]
    elif data is None:
        hues = hue
    else:
        hues = _parse_hue(data[hue])

    if signals is not None:
        assert len(x_values) == len(signals)
    else:
        signals = [(x, y) for x, y in zip(x_values, y_values)]

    # initialize star point objects
    star_points = []
    for i in range(len(x_values)):
        starp = custom_star(x=x_values[i],
                            y=y_values[i],
                            hue=hues[i],
                            signal=signals[i])
        if info_dicts is not None:
            starp.info_dict = info_dicts[i]
        star_points.append(starp)

    return _scatter(star_points)


def iscatter_items(items, x, y, func_plot, hue=None):
    """

    :param items:
    :param x: str
    :param y: str
    :param hue: str
    :param func_plot: lambda obj, axis: ....
    :return:
    """

    star_points = []
    for i in range(len(items)):
        starp = StarPoint.from_item(items[i], x, y, func_plot=func_plot)
        star_points.append(starp)

    if hue:
        hues = _parse_hue([getattr(item, hue) for item in items])
        for i, starp in enumerate(star_points):
            starp.hue = hues[i]

    return _scatter(star_points)


def _scatter(star_points):
    """

    :param: star_objs: List[StarPoint]
    """
    global fig

    # plot the first plot with star_points
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    for starp in star_points:
        artist = axs[0].plot(starp.x, starp.y, "o", color=starp.hue, picker=True)[0]
        artist.set_pickradius(10)
        # explanation of picker: https://matplotlib.org/3.1.1/gallery/event_handling/pick_event_demo.html
        artist.obj = starp

    fig.canvas.mpl_connect('pick_event', lambda event: event.artist.obj.plot(axs[1]))

    plt.show()

    return axs


def _parse_hue(data_hue, legend=True):
    """

    # todo: look seaborn scatter code to figure out how to implement hue

    :param data_hue: List[str]
    :return:
    """

    levels = list(set(data_hue))
    palette = _palette_from_levels(levels, legend=legend)
    hues = [palette.get(hi) for hi in data_hue]
    return hues


def _palette_from_levels(levels, legend=True):
    n_level = len(levels)
    palette = sns.color_palette("husl", n_level)
    palette = dict(zip(levels, palette))

    # plot legend
    if legend:
        plt.figure(figsize=(3, 1))
        for i in range(n_level):
            plt.plot(0, i, "o", color=palette[levels[i]])
            plt.text(0.01, i, str(levels[i]),
                     horizontalalignment='left',
                     size='medium',
                     color='black',
                     weight='semibold')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.show()

    return palette


class DummyObject:

    def __init__(self):
        self.a = np.random.random()
        self.b = np.random.random()
        self.c = np.random.random(5)
        self.d = np.random.randint(5)

    @staticmethod
    def plot_dummy(obj, axis):
        axis.plot(obj.c)
        axis.set_title(obj.d)


if __name__ == "__main__":
    # example code

    # prepare data
    points = np.random.rand(10, 3)
    df = pd.DataFrame(points[:, :2], columns=["x", "y"])
    labels = ["A", "B", "C"]
    df["labels"] = [labels[i] for i in np.random.choice(range(3), len(df), replace=True)]
    # df

    # plot it
    iscatter("x", "y", data=df, signals=points)

    # plot it
    iscatter("x", "y", hue="labels", data=df, signals=points)

    # plot it
    info_dicts = [{"title": label} for label in df["labels"]]
    iscatter("x", "y", hue="labels", data=df, signals=points,
             custom_star=TitleStarPoint, info_dicts=info_dicts)

    items = [DummyObject() for _ in range(10)]
    iscatter_items(items, "a", "b", hue="d", func_plot=DummyObject.plot_dummy)
