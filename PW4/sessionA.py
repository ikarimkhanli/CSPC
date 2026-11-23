import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.inspection import permutation_importance


def adjusted_r2(r2, n, p):
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    df = pd.read_csv("data/concrete.csv")
    print("=== A2: CLEANING AND SUMMARY ===")
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing values per column:\n", df.isna().sum())

    df_clean = df.drop_duplicates().reset_index(drop=True)
    print("Rows before cleaning:", len(df), "| Rows after cleaning:", len(df_clean))
    print("\nDescribe cleaned dataset:\n", df_clean.describe())

    feature_cols = [
        "cement",
        "slag",
        "ash",
        "water",
        "superplastic",
        "coarse_agg",
        "fine_agg",
        "age",
    ]
    X = df_clean[feature_cols]
    y = df_clean["strength"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(df_clean["cement"], df_clean["strength"], alpha=0.5)
    axes[0].set_xlabel("Cement (kg/m^3)")
    axes[0].set_ylabel("Compressive Strength (MPa)")
    axes[0].set_title("Strength vs Cement")

    axes[1].scatter(df_clean["age"], df_clean["strength"], alpha=0.5)
    axes[1].set_xlabel("Age (days)")
    axes[1].set_ylabel("Compressive Strength (MPa)")
    axes[1].set_title("Strength vs Age")
    plt.tight_layout()
    plt.savefig("results/figures/A2_strength_relationships.png", dpi=150)
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\n=== A3: SPLIT SIZES ===")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(X_train, y_train)

    y_train_pred = lin_reg.predict(X_train)
    y_test_pred = lin_reg.predict(X_test)

    n_train, p_lin = X_train.shape
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_adj_r2 = adjusted_r2(train_r2, n_train, p_lin)

    lin_metrics = pd.DataFrame(
        [
            {
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                "test_mae": mean_absolute_error(y_test, y_test_pred),
                "train_r2": train_r2,
                "train_adj_r2": train_adj_r2,
                "test_r2": test_r2,
            }
        ]
    )
    print("\n=== A4: LINEAR REGRESSION METRICS ===")
    print("Intercept:", lin_reg.intercept_)
    for name, coef in zip(feature_cols, lin_reg.coef_):
        print(f"  {name}: {coef}")
    print("\n", lin_metrics.to_string(index=False))

    residuals_test = y_test - y_test_pred
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(X_test["age"], residuals_test, alpha=0.6)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Age (days, log scale)")
    ax.set_ylabel("Residual (true - predicted, MPa)")
    ax.set_title("Linear Regression Test Residuals vs Age")
    plt.tight_layout()
    plt.savefig("results/figures/A4_residuals_vs_age.png", dpi=150)
    plt.close()

    poly_rows = []
    for deg in [1, 2, 3, 4]:
        poly_pipe = make_pipeline(
            PolynomialFeatures(degree=deg, include_bias=False),
            StandardScaler(),
            LinearRegression(),
        )
        poly_pipe.fit(X_train, y_train)
        p_deg = poly_pipe[0].n_output_features_

        tr_pred = poly_pipe.predict(X_train)
        te_pred = poly_pipe.predict(X_test)

        tr_r2 = r2_score(y_train, tr_pred)
        te_r2 = r2_score(y_test, te_pred)
        tr_adj_r2 = adjusted_r2(tr_r2, n_train, p_deg)

        poly_rows.append(
            {
                "degree": deg,
                "n_features": p_deg,
                "train_rmse": np.sqrt(mean_squared_error(y_train, tr_pred)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, te_pred)),
                "train_r2": tr_r2,
                "train_adj_r2": tr_adj_r2,
                "test_r2": te_r2,
            }
        )

    poly_df = pd.DataFrame(poly_rows)
    print("\n=== A5: POLYNOMIAL REGRESSION TABLE ===")
    print(poly_df.to_string(index=False))

    y_class = (df_clean["strength"] >= 40.0).astype(int)
    print("\n=== A6: CLASSIFICATION CLASS PROPORTIONS ===")
    print(y_class.value_counts(normalize=True))

    Xc_train, Xc_test, c_train, c_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    k_values = [1, 3, 5, 7, 15, 25, 51, 101]
    knn_rows = []
    for k in k_values:
        knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        knn_pipe.fit(Xc_train, c_train)
        knn_rows.append(
            {
                "K": k,
                "train_acc": knn_pipe.score(Xc_train, c_train),
                "test_acc": knn_pipe.score(Xc_test, c_test),
            }
        )
    knn_df = pd.DataFrame(knn_rows)
    print("\n=== A6: KNN K-SWEEP ===")
    print(knn_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(knn_df["K"], knn_df["train_acc"], marker="o", label="Train accuracy")
    ax.plot(knn_df["K"], knn_df["test_acc"], marker="s", label="Test accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("K (number of neighbors, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("KNN Accuracy vs K")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/A6_knn_k_sweep.png", dpi=150)
    plt.close()

    chosen_k = 7
    knn_scaled = make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=chosen_k)
    ).fit(Xc_train, c_train)
    knn_unscaled = KNeighborsClassifier(n_neighbors=chosen_k).fit(Xc_train, c_train)
    print(
        f"\nAt K={chosen_k}: scaled test acc = {knn_scaled.score(Xc_test, c_test):.4f} "
        f"vs unscaled test acc = {knn_unscaled.score(Xc_test, c_test):.4f}"
    )

    split_scores = []
    for rs in [0, 1, 2, 3, 4]:
        X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
            X, y_class, test_size=0.2, random_state=rs, stratify=y_class
        )
        m = make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=chosen_k)
        ).fit(X_tr_s, y_tr_s)
        split_scores.append(m.score(X_te_s, y_te_s))
    print(
        "5-split KNN test accuracies:",
        [round(s, 4) for s in split_scores],
        "| min:",
        round(min(split_scores), 4),
        "| max:",
        round(max(split_scores), 4),
        "| std:",
        round(float(np.std(split_scores)), 4),
    )

    depths = [1, 2, 3, 4, 5, 6, 8, None]
    tree_rows = []
    for d in depths:
        clf = DecisionTreeClassifier(
            criterion="entropy", max_depth=d, min_samples_leaf=1, random_state=0
        )
        clf.fit(Xc_train, c_train)
        tree_rows.append(
            {
                "max_depth": str(d),
                "n_leaves": clf.get_n_leaves(),
                "train_acc": clf.score(Xc_train, c_train),
                "test_acc": clf.score(Xc_test, c_test),
            }
        )
    tree_df = pd.DataFrame(tree_rows)
    print("\n=== A6: DECISION TREE DEPTH SWEEP ===")
    print(tree_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_pos = np.arange(len(tree_df))
    ax.plot(x_pos, tree_df["train_acc"], marker="o", label="Train accuracy")
    ax.plot(x_pos, tree_df["test_acc"], marker="s", label="Test accuracy")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tree_df["max_depth"])
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Decision Tree Accuracy vs max_depth")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/A6_tree_depth_sweep.png", dpi=150)
    plt.close()

    chosen_depth = 4
    chosen_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=chosen_depth,
        min_samples_leaf=1,
        random_state=0,
    ).fit(Xc_train, c_train)
    print(f"\nTop 2 levels of Decision Tree (max_depth={chosen_depth}):")
    print(export_text(chosen_tree, feature_names=feature_cols, max_depth=2))

    def eval_classifier(name, model, X_eval, y_eval, threshold=0.5):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_eval)[:, 1]
            y_pred = (proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_eval)
        tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
        return {
            "model": name,
            "threshold": threshold,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "accuracy": accuracy_score(y_eval, y_pred),
            "precision": precision_score(y_eval, y_pred, zero_division=0),
            "recall": recall_score(y_eval, y_pred, zero_division=0),
            "f1": f1_score(y_eval, y_pred, zero_division=0),
        }

    models_summary = pd.DataFrame(
        [
            eval_classifier(f"KNN(K={chosen_k})", knn_scaled, Xc_test, c_test, 0.5),
            eval_classifier(
                f"DecisionTree(depth={chosen_depth})", chosen_tree, Xc_test, c_test, 0.5
            ),
        ]
    )
    print("\n=== A7: MODEL COMPARISON (THRESHOLD = 0.5) ===")
    print(models_summary.to_string(index=False))

    print("\n=== A7: THRESHOLD SWEEP FOR DECISION TREE ===")
    for t in [0.3, 0.5, 0.7, 0.9]:
        proba_test = chosen_tree.predict_proba(Xc_test)[:, 1]
        y_pred_t = (proba_test >= t).astype(int)
        cm = confusion_matrix(c_test, y_pred_t)
        prec = precision_score(c_test, y_pred_t, zero_division=0)
        rec = recall_score(c_test, y_pred_t, zero_division=0)
        print(f"Threshold {t:.1f} -> prec={prec:.4f}, rec={rec:.4f}\nCM:\n{cm}")

    chosen_threshold = 0.7
    final_metrics = pd.DataFrame(
        [
            eval_classifier(f"KNN(K={chosen_k})", knn_scaled, Xc_test, c_test, 0.5),
            eval_classifier(
                f"DecisionTree(depth={chosen_depth}, t={chosen_threshold})",
                chosen_tree,
                Xc_test,
                c_test,
                chosen_threshold,
            ),
        ]
    )
    final_metrics.to_csv("results/A_metrics.csv", index=False)
    print("\nSaved results/A_metrics.csv")

    std_pipe = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
    std_coefs = std_pipe[-1].coef_

    perm_knn = permutation_importance(
        knn_scaled, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0
    )
    perm_tree = permutation_importance(
        chosen_tree, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0
    )

    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "raw_coef": lin_reg.coef_,
            "std_coef": std_coefs,
            "tree_importance": chosen_tree.feature_importances_,
            "perm_importance_knn": perm_knn.importances_mean,
            "perm_importance_tree": perm_tree.importances_mean,
        }
    )
    print("\n=== A8: FEATURE IMPORTANCE TABLE ===")
    print(imp_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].barh(imp_df["feature"], imp_df["std_coef"])
    axes[0].set_title("Standardized Coefs (Linear)")

    axes[1].barh(imp_df["feature"], imp_df["tree_importance"])
    axes[1].set_title("Tree Feature Importances")

    axes[2].barh(imp_df["feature"], imp_df["perm_importance_knn"])
    axes[2].set_title("Permutation Importance (KNN, F1)")
    plt.tight_layout()
    plt.savefig("results/figures/A8_feature_importances.png", dpi=150)
    plt.close()

    X_log_age = X.copy()
    X_log_age["age"] = np.log(X_log_age["age"])
    X_tr_la, X_te_la, y_tr_la, y_te_la = train_test_split(
        X_log_age, y, test_size=0.2, random_state=42
    )
    m_la = LinearRegression().fit(X_tr_la, y_tr_la)
    r2_la = r2_score(y_te_la, m_la.predict(X_te_la))

    X_wb = X_log_age.copy()
    binder = X_wb["cement"] + X_wb["slag"] + X_wb["ash"]
    X_wb["water"] = X_wb["water"] / binder
    X_tr_wb, X_te_wb, y_tr_wb, y_te_wb = train_test_split(
        X_wb, y, test_size=0.2, random_state=42
    )
    m_wb = LinearRegression().fit(X_tr_wb, y_tr_wb)
    r2_wb = r2_score(y_te_wb, m_wb.predict(X_te_wb))

    print("\n=== BONUS: DOMAIN FEATURES ===")
    print(f"log(age) only -> p={X_wb.shape[1]}, test R2={r2_la:.4f}")
    print(f"log(age) + w/b -> p={X_wb.shape[1]}, test R2={r2_wb:.4f}")


if __name__ == "__main__":
    main()