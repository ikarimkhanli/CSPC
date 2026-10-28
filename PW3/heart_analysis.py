import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Load data and inspect structure
df = pd.read_csv("heart.csv")
print("Data Shape (Rows, Columns):", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Isolate continuous columns
continuous_cols = ["age", "chol", "trestbps", "thalach"]
print("\nStatistical Summary:")
print(df[continuous_cols].describe())

# 2. Visualize distributions (4-panel subplot grid)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
    axes[i].hist(df[col].dropna(), bins=20, edgecolor='black', alpha=0.7)
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("heart_distributions.png")
plt.close()
print("\nPlot saved successfully as 'heart_distributions.png'.")

# 3. Normality Test (Shapiro-Wilk)
print("\n--- Normality Tests (Shapiro-Wilk) ---")
for col in continuous_cols:
    stat_val, p_val = stats.shapiro(df[col].dropna())
    print(f"{col}: p-value = {p_val:.5f} ({'Not normal' if p_val < 0.05 else 'Normal'})")

# 4. Compare thalach for Healthy vs. Disease groups
healthy = df[df["target"] == 0]["thalach"]
disease = df[df["target"] == 1]["thalach"]

print(f"\nHealthy group - Mean thalach: {healthy.mean():.2f}, SEM: {stats.sem(healthy):.2f}")
print(f"Disease group - Mean thalach: {disease.mean():.2f}, SEM: {stats.sem(disease):.2f}")

# Two-sample Independent T-test
t_stat, p_val_ttest = stats.ttest_ind(healthy, disease)
print(f"T-test result: t-statistic = {t_stat:.2f}, p-value = {p_val_ttest:.5f}")

# 5. Age vs. Thalach Correlation
corr, p_corr = stats.pearsonr(df["age"], df["thalach"])
print(f"\nPearson correlation (Age vs. Thalach): {corr:.2f} (p-value: {p_corr:.5f})")

plt.figure()
plt.scatter(df["age"], df["thalach"], alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Thalach (Max Heart Rate)")
plt.title("Age vs Thalach")
plt.savefig("age_thalach_scatter.png")
plt.close()
print("Plot saved successfully as 'age_thalach_scatter.png'.")