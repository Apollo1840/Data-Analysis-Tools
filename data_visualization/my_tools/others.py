import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# change correlation matrix to parallel_coordinates-style heatmap
v = []
a = np.reshape(range(9), (3, 3))
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        v.append([i, j, a[i, j]])

df = pd.DataFrame(v)
df.columns = ["x", "y", "att"]
print(df)

fig = px.parallel_coordinates(df, color="att", dimensions=["x", "y"])
fig.show()


def parallel_corr(cor_matrix):
    """
    every row in cor_matrix sums up to 1: np.sum(att[0], axis=1) == 1

    """
    v = []
    # print(cor_matrix)
    for i in range(cor_matrix.shape[0]):
        max_, min_, med_ = np.max(cor_matrix[i]), np.min(cor_matrix[i]), np.median(cor_matrix[i])
        # scales = []
        for j in range(cor_matrix.shape[1]):
            scale = (cor_matrix[i, j] - min_) / (max_ - min_)
            scale = scale if scale > med_ else scale / 2
            v.append([str(i), str(j), scale])
            # scales.append(scale)
            # print(cor_matrix[i, j].numpy())
        # plt.hist(scales)
        # plt.show()

    # max_, min_ = np.max([vi[2] for vi in v]), np.min([vi[2] for vi in v])
    # scales = [(vi[2]-min_)/(max_-min_) for vi in v]

    # plt.hist(scales)
    # plt.show()

    # median = np.median(scales)
    # scales = [s if s>median else s/2 for s in scales]

    # plt.hist(scales)
    # plt.show()

    # unit_sigmoid = lambda x: 1 / (1 + np.exp(-x * np.exp(1)))
    # scales = [unit_sigmoid(s) for s in scales]
    # base = np.exp(max([vi[2] for vi in v]))
    # scales = [np.exp(vi[2]) / base for vi in v]

    for vi in v:
        plt.plot([vi[0], vi[1]], ["0", "1"], alpha=vi[2], linewidth=vi[2], c=(vi[2], 0, 0))

    plt.xticks([])
    plt.yticks([])
    plt.show()


def permutation_show(perm1, perm2):
    assert len(set(perm1).intersection(set(perm2))) == len(set(perm1))

    for i in range(len(perm1)):
        p1 = perm1[i]
        p2 = perm2.index(p1)
        plt.plot([i, p2], ["0", "1"])
        plt.text(p2, "1", p1)

    plt.xticks(range(len(perm1)), perm1)
    plt.yticks([])
    plt.show()

