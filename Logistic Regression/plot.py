import pandas as pd
import matplotlib.pyplot as plt


def main():
    cv_results = pd.read_csv("cv_results_class_weight.csv")
    # cv_results = pd.read_csv("cv_results_class_weight_smote.csv")
    key_info = cv_results[["param_class_weight", "mean_test_score"]]
    plot_data = {"class_1_weight": [], "F1_score": []}
    best_weight = 0
    best_score = 0

    for i in range(0, len(key_info)):
        row = key_info.loc[i]

        class_1_weight = row["param_class_weight"]
        class_1_weight = float(class_1_weight.split(":")[-1].strip().strip("}"))
        F1_score = row["mean_test_score"]
        if F1_score > best_score:
            best_score = F1_score
            best_weight = class_1_weight

        plot_data["class_1_weight"].append(class_1_weight)
        plot_data["F1_score"].append(F1_score)

    plt.plot(plot_data["class_1_weight"], plot_data["F1_score"])
    plt.plot(best_weight, best_score, marker="o", label=f"{best_weight}, {best_score:.3f}")
    plt.xlabel("Class 1 Weight")
    plt.ylabel("F1 Score")
    plt.title("Class Weight Tunning with Original Data")
    # plt.title("Class Weight Tunning with Smote Data")
    plt.legend()
    plt.xticks([i * 0.1 for i in range(0, 11)])
    plt.show()

    # cv_results = pd.read_csv("cv_results_penalty.csv")
    # penalty = cv_results["param_penalty"]
    # F1_score = cv_results["mean_test_score"]
    # print(penalty)
    # print(F1_score)
    # plt.bar(penalty, F1_score)
    # plt.ylabel("F1 Score")
    # plt.title("F1 Score of Different Penalty Types")
    # plt.show()


if __name__ == "__main__":
    main()