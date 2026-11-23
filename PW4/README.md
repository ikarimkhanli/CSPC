# Practical Work 4: Supervised & Unsupervised ML

## Session A: Concrete Strength (`fetch_concrete.py`, `sessionA.py`)
- **Data & Cleaning (A1–A3):** Features (`cement`, `slag`, `ash`, `water`, `superplastic`, `coarse_agg`, `fine_agg`, `age` in `kg/m^3` and `day`), target `strength` (`MPa`). Duplicates are removed before splitting (`80/20`, `random_state=42`) to prevent train-test leakage (`data/concrete.csv`, `results/figures/A2_strength_relationships.png`).
- **Regression (A4–A5):** Linear regression train/test RMSE, MAE, R², adjusted R², and residuals vs `age` (`results/figures/A4_residuals_vs_age.png`). Polynomial degrees `1..4` (`p`, train/test RMSE, R², adjusted R²), selected degree rationale, and high-degree (`deg=4`) overfitting risk.
- **Classification (A6–A7):** Target `strength >= 40 MPa`. KNN (`K` sweep, scaling impact, 5-split stability; `results/figures/A6_knn_k_sweep.png`) and Decision Tree (`max_depth` sweep, leaf counts, root split meaning; `results/figures/A6_tree_depth_sweep.png`). Metrics (`results/A_metrics.csv`), safety impact of FP vs FN on site, and threshold (`0.3, 0.5, 0.7, 0.9`) trade-off.
- **Importance & Bonus (A8):** Feature importance table & plot (`results/figures/A8_feature_importances.png`), ranking differences across methods, causal caution, and `log(age)` / `w/b` comparison.

## Session B: Wine Chemistry (`sessionB.py`)
- **PCA & Scaling (B1–B3):** Wine dataset summary and feature scale differences. Raw PCA vs standardized PCA (`results/figures/B3_scree_plot.png`, `results/figures/B3_pc1_pc2_unlabeled.png`), explained variance, loadings, and why standardization is needed.
- **K-Means & Cultivars (B4–B6):** Seed sensitivity (`random_state=0..7`, `n_init=1` vs `n_init=10`), elbow curve (`k=1..10`, `results/figures/B4_elbow_curve.png`), cluster profiles & PC1–PC2 plot (`results/figures/B5_pc1_pc2_clusters.png`). Cultivar agreement (`results/B_agreement.csv`) across scaled, raw, and PC1–PC2 representations, plus supervised KNN bonus note.

## Latest Run & Automation Notes (B7–B8)
- **Pipeline status:** Latest `Snakefile` dry-run / execution result: `[add latest snakemake output / status here]`.
- **Reproducibility behavior:** `snakemake -n` (dry run), clean rebuild after removing `data/` and `results/`, no-op re-run when outputs are up to date, and selective re-run after changing a hyperparameter in `sessionA.py`.
-