"""

extension of matplotlib, in seaborn API style.

"""

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolor

# very good learn example
color_map = mplcolor.LinearSegmentedColormap(
    "my_map",
    {
        "red": [(0, 1.0, 1.0),
                (1.0, .5, .5)],
        "green": [(0, 0.5, 0.5),
                  (1.0, 0, 0)],
        "blue": [(0, 0.50, 0.5),
                 (1.0, 0, 0)]
    }
)


def barplot_colorbar(x, y, color, data):
    """
    plot the bar chart, whos bar color is some source of information with

    :params: x: str
    :params: y: str
    :params: color: str
    :params: data: pd.DataFrame

    """

    data_normalizer = mplcolor.Normalize()

    color_map = mplcolor.LinearSegmentedColormap.from_list("my_map", ["g", "r"])
    """
    color_map = mplcolor.LinearSegmentedColormap(
        "my_map",
        segmentdata={'red': [(0.0, 0.0, 0.0),
                             (0.5, 1.0, 1.0),
                             (1.0, 1.0, 1.0)],

                     'green': [(0.0, 0.0, 0.0),
                               (0.25, 0.0, 0.0),
                               (0.75, 1.0, 1.0),
                               (1.0, 1.0, 1.0)],

                     'blue': [(0.0, 0.0, 0.0),
                              (0.5, 0.0, 0.0),
                              (1.0, 1.0, 1.0)]}
    )
    
    """

    x_value = data[x]
    y_value = data[y]
    c_value = data[color]

    plt.bar(list(range(len(x_value))),
            y_value,
            align="center",
            color=color_map(data_normalizer(c_value)))

    plt.xticks(list(range(len(x_value))), x_value)
    plt.show()


def barhplot_stacked(x, y, hue, data, sort_by_x=True, ys=None, hues=None, show=True,
                     color_palette=None):
    """

    > df_test = pd.DataFrame({
            "x": [20, 35, 30, 35, 27],
            "y": ['G1', 'G1', 'G2', 'G2', 'G3'],
            "hue": ["stage1", "stage2", "stage1", "stage2","stage1"]
        })

    > barhplot_stacked("x", "y", "hue", df_test)


    """

    if sort_by_x:
        df_sort = data.groupby(y)[x].apply(sum)
        df_sort = df_sort.reset_index()
        df_sort = df_sort.sort_values(by=x, ascending=True)
        ys = df_sort[y]
    elif ys is None:
        ys = sorted(set(data[y]))

    if hues is None:
        hues = sorted(set(data[hue]))

    last_xs = [0 for _ in range(len(ys))]
    for huei in hues:
        xs = [sum(data.loc[(data[y] == yi) & (data[hue] == huei), x]) for yi in ys]
        if sum(xs) > 0:
            if color_palette is not None:
                plt.barh(ys, xs, left=last_xs, label=huei, color=next(color_palette))
            else:
                plt.barh(ys, xs, left=last_xs, label=huei)
        last_xs = [last_xs[i] + xs[i] for i in range(len(ys))]

    plt.ylabel(y)
    plt.legend()

    if show:
        plt.show()
