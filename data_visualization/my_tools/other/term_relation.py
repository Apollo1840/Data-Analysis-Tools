import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


# Term relation
def parallel_corr_plotly(df):
    # change correlation matrix to parallel_coordinates-style heatmap
    fig = px.parallel_coordinates(df, color="att", dimensions=["x", "y"])
    fig.show()


# Term relation
def parallel_corr(cor_matrix):
    """
    every row in cor_matrix sums up to 1: np.sum(att[0], axis=1) == 1

    """

    stack_cor = stacklise_scale(cor_matrix)

    for cori in stack_cor:
        # cori: i, j, correlation
        plt.plot([cori[0], cori[1]], ["0", "1"], alpha=cori[2], linewidth=cori[2], c=(cori[2], 0, 0))

    plt.xticks([])
    plt.yticks([])
    plt.show()


# Term relation: helper function
def stacklise(cor_matrix):
    v = []
    for i in range(cor_matrix.shape[0]):
        for j in range(cor_matrix.shape[1]):
            v.append([str(i), str(j), cor_matrix[i, j]])
    return v


# Term relation: helper function
def stacklise_scale(cor_matrix):
    v = []

    for i in range(cor_matrix.shape[0]):
        # for the maxmin normalize
        max_, min_, med_ = np.max(cor_matrix[i]), np.min(cor_matrix[i]), np.median(cor_matrix[i])
        for j in range(cor_matrix.shape[1]):
            # maxmin normalize
            scale = (cor_matrix[i, j] - min_) / (max_ - min_)
            # cut the lower half
            scale = scale if scale > med_ else scale / 2

            v.append([str(i), str(j), scale])

    return v


# Term relation: helper function
def permutation_show(perm1, perm2):
    """
    ["A", "B", "C"]
    ["A", "C", "B"]

    :param: perm1: List[str]
    """

    assert len(set(perm1).intersection(set(perm2))) == len(set(perm1))

    for i in range(len(perm1)):
        p1 = perm1[i]
        p2 = perm2.index(p1)
        plt.plot([i, p2], ["0", "1"])
        plt.text(p2, "1", p1)

    plt.xticks(range(len(perm1)), perm1)
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    v = []
    a = np.reshape(range(9), (3, 3))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            v.append([i, j, a[i, j]])

    df = pd.DataFrame(v)
    df.columns = ["x", "y", "att"]
