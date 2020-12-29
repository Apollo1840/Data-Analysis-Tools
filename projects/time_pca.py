import pandas as pd
import os
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
plt.style.use('seaborn')

os.chdir(os.path.dirname(os.getcwd()))
print(os.getcwd())

PATH_TO_DB = "./datasets/power_usage"
PATH_TO_DATA_PU = os.path.join(PATH_TO_DB, "power_usage_2016_to_2020.csv")
PATH_TO_DATA_TH = os.path.join(PATH_TO_DB, "weather_2016_2020_daily.csv")

START_DATE = "StartDate"
DATE = "Date"
VALUE = "Value (kWh)"

CASES = "cases"
DAYS_AFTER = "days_after"
DATETIME = "datetime"
PERIOD_ID = "period_id"
PERIOD_ID_DAY = "period_id_day"


def average_smooth(sig, level=10):
    new_sig = [np.mean(sig[max((i - level//2), 0):(i + level//2)]) for i in range(len(sig)-level//2)]
    return new_sig


if __name__ == '__main__':
    df_pu = pd.read_csv(PATH_TO_DATA_PU)
    df_th = pd.read_csv(PATH_TO_DATA_TH)

    df_pu[DATETIME] = df_pu[START_DATE].apply(lambda date_obj: datetime.strptime(str(date_obj), "%Y-%m-%d %H:%M:%S"))
    df_pu[DATE] = df_pu[DATETIME].dt.strftime("%Y-%m-%d")

    df_pu_date = df_pu.groupby(by=DATE)[VALUE].apply(sum)
    df_pu_date = df_pu_date.reset_index()

    df = df_pu_date.merge(df_th, on=DATE, how="inner")

    df.head()

    feature_columns = [c for c in df.columns if c.endswith("_avg")]

    # plot multiple signals
    plt.figure()

    for f in feature_columns:
        plt.plot(df[f], label=f, alpha=0.8)

    plt.legend(loc='upper left')
    plt.ylim((-10, 150))
    plt.xlabel('Date (Days after 2016-01-06')
    plt.ylabel('Stack of time series data')
    plt.show()

    # apply PCA
    data = np.array(df[feature_columns].values)
    data = Normalizer().fit_transform(data)
    pca_transformer = PCA(n_components=1, random_state=0)
    principle_feature = pca_transformer.fit_transform(data)
    principle_feature = list(principle_feature.ravel())

    plt.plot(principle_feature, label="principle_feature", alpha=0.6, color="b")
    plt.legend(loc="upper left")
    plt.xlabel('Date (Days after 2016-01-06')
    plt.show()

    fig, ax1 = plt.subplots()

    ax1.plot(principle_feature, label="principle_feature", alpha=0.4, color="b")
    ax1.plot(average_smooth(principle_feature), color="b")
    plt.legend(loc="upper left")

    ax2 = ax1.twinx()

    ax2.plot(df[VALUE], label=VALUE, alpha=0.4, color="r")
    ax2.plot(average_smooth(df[VALUE]), color="r")
    plt.legend(loc="upper right")

    plt.grid(None)
    plt.xlabel('Date (Days after 2016-01-06')
    plt.show()

    # sns.barplot(x=list(range(len(principle_feature))), y=principle_feature, hue=df[VALUE])
    # plt.bar(x=range(len(principle_feature)), height=principle_feature, color=df[VALUE])












