import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar


def np_random_normal(num, scale=0.5):
    return np.array([min(abs(np.random.normal(scale=scale)) / 3, 1) for _ in range(num)])


def pie_heatmap(data, row_names, col_names):
    """
    row will be the theta,
    col wiil be the r,
    """

    def meshgrid_for_polar(n_sections, n_layers):
        return np.meshgrid(np.linspace(0, 2 * np.pi, n_sections), np.arange(n_layers))

    # produce polar plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(6, 6))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # plot data
    theta, r = meshgrid_for_polar(n_sections=len(row_names) + 1, n_layers=len(col_names) + 1)

    print(len(theta))
    print(len(data))

    ax.pcolormesh(theta, r, data, cmap="YlGnBu")

    # set ticklabels
    pos, step = np.linspace(0, 2 * np.pi, len(row_names), endpoint=False, retstep=True)
    pos += step / 2.
    ax.set_xticks(pos)
    ax.set_xticklabels(row_names)

    ax.set_yticks(np.arange(len(col_names)))
    ax.set_yticklabels(col_names)
    plt.show()


if __name__ == "__main__":
    # generate the table with timestamps
    np.random.seed(1)

    # noise
    times = pd.Series(pd.to_datetime("Nov 1 '14 at 0:42") +
                      pd.to_timedelta(np.random.rand(int(1e3)) * 60 * 24 * 40, unit='m'))

    times = times.append(pd.Series(pd.to_datetime("Dec 16 '20 at 0:42") +
                                   pd.to_timedelta(np_random_normal(int(1e5)) * 60 * 24 * 40, unit='m')))

    times = times.append(pd.Series(pd.to_datetime("Dec 19 '20 at 0:42") +
                                   pd.to_timedelta(np_random_normal(int(1e4), scale=0.1) * 60 * 24 * 40, unit='m')))

    # generate counts of each (weekday, hour)
    data = pd.crosstab(times.dt.weekday,
                       times.dt.hour.apply(lambda x: '{:02d}:00'.format(x))).fillna(0)
    data.index = [calendar.day_name[i][0:3] for i in data.index]
    data = data.T

    print(data[:100])


    # plot the pie heatmap
    row_names = data.index
    col_names = data.columns
    data = data.T.values

    pie_heatmap(data, row_names, col_names)
