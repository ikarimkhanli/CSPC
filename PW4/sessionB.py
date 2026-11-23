import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline


def agreement_score(y_true, labels):
    ct = pd.crosstab(y_true, labels)
    return ct.max(axis=0).sum() / len(y_true), ct


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    wine = load_wine(as_frame=True)
    X = wine.data
    cultivar = wine.target

    print("=== B1: WINE DATASET DESCRIPTION ===")
    print(wine.DESCR)
    print("\n=== B1: FEATURE SUMMARY ===")
    desc = X.describe()
    print(desc)
    ranges = desc.loc["max"] - desc.loc["min"]
    print("\nFeature ranges (max - min) sorted:\n", ranges.sort_values(ascending=False))
    print("\nFeature std sorted:\n", desc.loc["std"].sort_values(ascending=False))

    pca_raw = PCA(n_components=None)
    pca_raw.fit(X)
    pc1_loadings = pd.Series(pca_raw.components_[0], index=X.columns).sort_values(
        key=abs, ascending=False
    )
    print("\n=== B2: RAW PCA ===")
    print("PC1 explained variance ratio:", pca_raw.explained_variance_ratio_[0])
    print("PC1 loadings sorted by magnitude:\n", pc1_loadings)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(X)
    print("\n=== B3: SCALED DATA CHECK ===")
    print("Mean after scaling (max abs):", np.max(np.abs(Z.mean(axis=0))))
    print("Std after scaling:", np.round(Z.std(axis=0), 4))

    pca_scaled = PCA(n_components=None)
    Z_pca = pca_scaled.fit_transform(Z)
    evr = pca_scaled.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    scree_df = pd.DataFrame(
        {
            "PC": [f"PC{i+1}" for i in range(len(evr))],
            "explained_variance_ratio": evr,
            "cumulative_variance": cum_evr,
        }
    )
    print("\n=== B3: SCALED PCA VARIANCE ===")
    print(scree_df.to_string(index=False))
    n_90 = int(np.argmax(cum_evr >= 0.90) + 1)
    print(f"Components needed for >= 90% variance: {n_90}")
    print(f"Variance kept by PC1 + PC2: {cum_evr[1]:.4f}")

    loadings_df = pd.DataFrame(
        {
            "PC1": pca_scaled.components_[0],
            "PC2": pca_scaled.components_[1],
        },
        index=X.columns,
    )
    print("\nPC1 loadings sorted:\n", loadings_df["PC1"].sort_values())
    print("\nPC2 loadings sorted:\n", loadings_df["PC2"].sort_values())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_idx = np.arange(1, len(evr) + 1)
    ax.bar(x_idx, evr, alpha=0.7, label="Individual EVR")
    ax.plot(x_idx, cum_evr, marker="o", color="tab:orange", label="Cumulative EVR")
    ax.axhline(0.90, color="red", linestyle="--", label="0.90 threshold")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"PC{i}" for i in x_idx])
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Scree Plot (Scaled Wine Data)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figures/B3_scree_plot.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(Z_pca[:, 0], Z_pca[:, 1], alpha=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Wines on PC1-PC2 Plane (Unlabeled)")
    plt.tight_layout()
    plt.savefig("results/figures/B3_pc1_pc2_unlabeled.png", dpi=150)
    plt.close()

    print("\n=== B4(a): INERTIA ACROSS RANDOM SEEDS (k=3, n_init=1) ===")
    for rs in range(8):
        km_tmp = KMeans(
            n_clusters=3, init="k-means++", n_init=1, max_iter=300, random_state=rs
        ).fit(Z)
        print(f"random_state={rs} -> inertia={km_tmp.inertia_:.2f}")

    inertias = []
    k_range = list(range(1, 11))
    for k in k_range:
        km_k = KMeans(
            n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=0
        ).fit(Z)
        inertias.append(km_k.inertia_)

    print("\n=== B4(b): ELBOW CURVE (k=1..10, n_init=10, random_state=0) ===")
    for k, iner in zip(k_range, inertias):
        print(f"k={k:2d} -> inertia={iner:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_range, inertias, marker="o")
    ax.set_xticks(k_range)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Curve on Scaled Wine Data")
    plt.tight_layout()
    plt.savefig("results/figures/B4_elbow_curve.png", dpi=150)
    plt.close()

    chosen_k = 3
    km = KMeans(
        n_clusters=chosen_k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=0,
    ).fit(Z)

    print("\n=== B5: CLUSTER SIZES AND CENTERS ===")
    print("Cluster sizes (np.bincount):", np.bincount(km.labels_))
    centers_orig = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=X.columns
    )
    print("\nCluster centers in original units:\n", centers_orig.round(2).to_string())

    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter = ax.scatter(Z_pca[:, 0], Z_pca[:, 1], c=km.labels_, cmap="tab10", alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Wines on PC1-PC2 Plane Colored by K-Means Cluster (k=3)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig("results/figures/B5_pc1_pc2_clusters.png", dpi=150)
    plt.close()

    agr_scaled, ct_scaled = agreement_score(cultivar, km.labels_)

    km_raw = KMeans(
        n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0
    ).fit(X)
    agr_raw, ct_raw = agreement_score(cultivar, km_raw.labels_)

    km_pca2 = KMeans(
        n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0
    ).fit(Z_pca[:, :2])
    agr_pca2, ct_pca2 = agreement_score(cultivar, km_pca2.labels_)

    print("\n=== B6: CROSSTABS ===")
    print("(i) Scaled Z:\n", ct_scaled)
    print("\n(ii) Raw X:\n", ct_raw)
    print("\n(iii) Scaled PC1-PC2:\n", ct_pca2)

    agreement_df = pd.DataFrame(
        [
            {"representation": "scaled_Z", "k": 3, "agreement": agr_scaled},
            {"representation": "raw_X", "k": 3, "agreement": agr_raw},
            {"representation": "scaled_PC1_PC2", "k": 3, "agreement": agr_pca2},
        ]
    )
    agreement_df.to_csv("results/B_agreement.csv", index=False)
    print("\n=== B6: AGREEMENT TABLE ===")
    print(agreement_df.to_string(index=False))

    X_tr_w, X_te_w, y_tr_w, y_te_w = train_test_split(
        X, cultivar, test_size=0.25, random_state=42, stratify=cultivar
    )
    sup_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(
        X_tr_w, y_tr_w
    )
    sup_pca_knn = make_pipeline(
        StandardScaler(), PCA(n_components=2), KNeighborsClassifier(n_neighbors=5)
    ).fit(X_tr_w, y_tr_w)

    print("\n=== BONUS: SUPERVISED WINE CLASSIFICATION ===")
    print(f"StandardScaler + KNN(5) test accuracy: {sup_knn.score(X_te_w, y_te_w):.4f}")
    print(
        f"StandardScaler + PCA(2) + KNN(5) test accuracy: "
        f"{sup_pca_knn.score(X_te_w, y_te_w):.4f}"
    )


if __name__ == "__main__":
    main()