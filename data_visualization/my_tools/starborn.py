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
