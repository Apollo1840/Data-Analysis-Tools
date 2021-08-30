import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from rent_entities import gepflegte2, mobschon


def scatterplot_text(x, y, text, data, *args, **kwargs):
    sns.scatterplot(x, y, data=data, *args, **kwargs)
    for line in range(0, len(data)):
        plt.text(data[x][line] + 5,
                 data[y][line],
                 data[text][line],
                 horizontalalignment='left',
                 size='medium',
                 color='black')
        print(
            "{}: \n\tpro:{}\n\tcon:{}\n".format(
                data["id"][line],
                data["pro"][line],
                data["con"][line])
        )

    plt.plot([1000, np.mean(data[x]) * 1.2],
             [35, np.mean(data[y]) * 1.2], "g--")
    plt.grid() 
    plt.show()


def app2df(list_app_ents):
    data = {}
    for key in list_app_ents[0]._serialize().keys():
        data[key] = []

    for ae in list_app_ents:
        ae_dict = ae._serialize()
        for key, value in ae_dict.items():
            data[key].append(value)

    return pd.DataFrame(data)


if __name__ == "__main__":
    oppotunities = [gepflegte2,
                    mobschon
                    ]

    df = app2df(oppotunities)

    df = df.sort_values(["desire_score"], ascending=False)
    df = df.reset_index(drop=True)

    df = df.sort_values(["size"], ascending=False)
    df = df.reset_index(drop=True)
    plt.figure(figsize=(10, 10))
    scatterplot_text("price", "size", "id", hue="label", size="desire_score", data=df)
