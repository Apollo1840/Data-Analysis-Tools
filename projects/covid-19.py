import pandas as pd
import os
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw

from timewheel import pie_heatmap_df, pie_heatmap

plt.style.use('seaborn')

os.chdir(os.path.dirname(os.getcwd()))
print(os.getcwd())

PATH_TO_DB = "./datasets/covid"
PATH_TO_DATA = os.path.join(PATH_TO_DB, "covid_de.csv")

DATE = "date"
CASES = "cases"
DAYS_AFTER = "days_after"
DATETIME = "datetime"
PERIOD_ID = "period_id"
PERIOD_ID_DAY = "period_id_day"


def calc_dist_metric(df_date, period):
    df_date[PERIOD_ID] = [day // period for day in df_date[DAYS_AFTER]]
    df_date[PERIOD_ID_DAY] = [day % period for day in df_date[DAYS_AFTER]]

    # calculate the dist_metric
    df_timewheel_t = df_date.pivot_table(values=CASES, index=[PERIOD_ID], columns=[PERIOD_ID_DAY], aggfunc='sum')
    df_timewheel_t = df_timewheel_t.dropna()
    data_t = np.array(df_timewheel_t.values)
    data_t /= np.sum(data_t, axis=1)[:, np.newaxis]

    dist_matrix = np.zeros((data_t.shape[0], data_t.shape[0]))
    for i in range(data_t.shape[0]):
        for j in range(data_t.shape[0]):
            dist, _ = fastdtw(data_t[i, :], data_t[j, :])
            dist_matrix[i, j] = dist

    # print(dist_matrix)
    dist_metric = np.median(dist_matrix)

    return dist_metric


if __name__ == "__main__":
    df = pd.read_csv(PATH_TO_DATA)
    print(df.info())
    print(df.head())

    df_date = df.groupby(DATE)[CASES].apply(sum)
    df_date = df_date.reset_index()
    sns.barplot(x=DATE, y=CASES, data=df_date)
    # todo: filter by start_date

    df_date[DATETIME] = df_date[DATE].apply(lambda date_obj: datetime.strptime(str(date_obj), "%Y-%m-%d"))
    df_date = df_date.sort_values(DATETIME)
    df_date[DAYS_AFTER] = df_date[DATETIME] - pd.to_datetime([df_date[DATETIME][0] for _ in range(len(df_date))])
    df_date[DAYS_AFTER] = df_date[DAYS_AFTER].dt.days
    print(df_date.head())

    period = 7
    df_date[PERIOD_ID] = [day // period for day in df_date[DAYS_AFTER]]
    df_date[PERIOD_ID_DAY] = [day % period for day in df_date[DAYS_AFTER]]

    # plot the charts
    df_timewheel = df_date.pivot_table(values=CASES, index=[PERIOD_ID_DAY], columns=[PERIOD_ID], aggfunc='sum')
    print(df_timewheel.head())

    pie_heatmap_df(df_timewheel)

    data = np.array(df_timewheel.values)
    data /= np.sum(data, axis=0)[np.newaxis, :]
    pie_heatmap(data, df_timewheel.index, df_timewheel.columns)
    # todo: plot xlabels with jump

    # calculate the dist_metric
    dist_metric = calc_dist_metric(df_date, period=period)

    periods = [3, 5, 6, 7, 8, 10, 13, 14, 15, 20, 25, 28, 29, 30, 60]
    metrics = [calc_dist_metric(df_date, period) for period in periods]

    plt.plot(periods, metrics, ".-")
    plt.xticks(np.arange(0, max(periods) + 1, 7))
    plt.xlabel("length of the period (days)")
    plt.ylabel("fastdtw in-matrix distance metric")
    plt.show()
