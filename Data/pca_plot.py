import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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


def main():
    # loading dataset
    data = pd.read_csv("creditcard.csv")

    time_col = data["Time"]

    # extract time of day
    seconds_in_a_day = 24 * 60 * 60
    time_of_day = time_col % seconds_in_a_day

    # place the new feature into dataset
    data["Time of Day"] = time_of_day
    data.drop(columns="Time", inplace=True)

    standardize_col(data, ["Amount", "Time of Day"])

    corr_matrix = data.corr(method="pearson")
    # sn.heatmap(data=corr_matrix, xticklabels=True, yticklabels=True, cmap="PiYG", vmin=-1, vmax=1)
    # plt.show()

    corr_class = corr_matrix["Class"]
    print(corr_class.to_string(), end="\n\n")

    # select the most correlated features
    selected_feat = corr_class.abs().sort_values(ascending=False)
    selected_feat = selected_feat[selected_feat >= 0.1]
    selected_feat.drop(labels="Class", inplace=True)
    selected_feat = list(selected_feat.index)
    print("The following features are selected:")
    print(selected_feat, end="\n\n")

    x_selected = data[selected_feat]

    d = x_selected.shape[1]
    pca = PCA(n_components=d)
    pca.fit(x_selected)

    exp_var = pca.explained_variance_ratio_
    exp_var_sum = 0
    explained_var = []
    for ratio in exp_var:
        exp_var_sum += ratio
        explained_var.append(exp_var_sum)

    plt.plot(list(range(1, d + 1)), explained_var)
    plt.plot(9, explained_var[8], marker="o", label=f"9, {explained_var[8]:.3f}")
    plt.title("Explained Data Variance vs. Number of Principal Components")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Explained Variance")
    plt.legend()
    plt.xticks(list(range(1, 12)))
    plt.show()
    


if __name__ == "__main__":
    main()