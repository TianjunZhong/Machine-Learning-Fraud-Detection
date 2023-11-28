import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA


def standardize_col(df: pd.DataFrame, columns: list):
    '''
    standardize specified columns of a dataframe
    '''
    for column_name in columns:
        col = df[column_name]
        mean = col.mean()
        std = col.std()
        col = (col - mean) / std
        df[column_name] = col


def find_n_components(x: pd.DataFrame, explained_varince: float):
    d = x.shape[1]
    pca = PCA(n_components=d)
    pca.fit(x)

    exp_var = pca.explained_variance_ratio_
    exp_var_sum = 0
    for n, ratio in enumerate(exp_var):
        exp_var_sum += ratio
        # print(exp_var_sum)
        if exp_var_sum >= explained_varince:
            return n + 1


def main():
    # loading dataset
    data = pd.read_csv("creditcard.csv")

    # y = data["Class"]
    # x = data.drop(columns="Class")

    """
    feature extraction

    The semantic meanings of variables V1 through V28 are not provided, but we assume they 
    are somewhat meaningful. "Amount" is clear, leaving only "Time", which measures the 
    "Number of seconds elapsed between this transaction and the first transaction in the 
    dataset", lacking semantic sense. Feature extraction is required for "Time."
    """
    time_col = data["Time"]

    # extract time of day
    seconds_in_a_day = 24 * 60 * 60
    time_of_day = time_col % seconds_in_a_day

    # place the new feature into dataset
    data["Time of Day"] = time_of_day
    data.drop(columns="Time", inplace=True)

    """
    feature scaling

    Thanks to the kaggle dataset, V1 through V28 are already standardized, leaving only
    "Amount" and "Time of Day" to be standardized.
    """
    standardize_col(data, ["Amount", "Time of Day"])

    """
    feature selection
    """
    corr_matrix = data.corr(method="pearson")
    # sn.heatmap(data=corr_matrix, xticklabels=True, yticklabels=True, cmap="PiYG", vmin=-1, vmax=1)
    # plt.show()

    corr_class = corr_matrix["Class"]
    print(corr_class.to_string(), end="\n\n")

    selected_feat = corr_class.abs().sort_values(ascending=False)
    selected_feat = selected_feat[selected_feat >= 0.1]
    selected_feat.drop(labels="Class", inplace=True)
    selected_feat = list(selected_feat.index)
    print(selected_feat, end="\n\n")

    x_selected = data[selected_feat]

    """
    PCA

    # Due the large number of features (30), it's inconvenient and complicated to find 
    # correlations or information entropies among the features and then select appropriate
    # features. Instead, we adopt PCA to reduce the dimension.
    """
    n = find_n_components(x_selected, 0.9)
    print(f"Number of principal components to capture 90% variation is {n}", end="\n\n")
    pca = PCA(n_components=n)
    x_new = pd.DataFrame(pca.fit_transform(x_selected))
    print("The PCA has following components:")
    print(pca.components_)

    # output new data
    x_new.to_csv("x.csv", index=False)
    data["Class"].to_csv("y.csv", index=False)





if __name__ == "__main__":
    main()