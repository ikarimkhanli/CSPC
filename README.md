# CSPC -Computer Science for Physics and Chemistry
## PW1 --- Lab B
- **What the data showed:** The data points showed how the count values decrease over time in a decay process.
- **Comparison with analytical law:** The observed data points matched the analytical decay law very well on the plot.
- **Snakemake pipeline:** I made a Snakefile to run the script automatically.
g

## PW2 --- Lab A: Motion from Tracking Data

- **Mean Acceleration:** -8.58 m/s²
- **Noise Observation:** Taking a derivative compares nearby points, which makes the noise bigger. Because acceleration is calculated using two derivatives, it becomes much noisier than the position data.
- **Integration Results:** Integration works in the opposite way of differentiation by adding up values, which helps cancel out random noise. The position recovered by integration matches the original position very closely, with a maximum difference of about 0.78 meters.


# PW2 Lab B - Numerical Optimization and Chemical Applications

This repository contains the Python-based solutions for Laboratory B (PW2), focusing on numerical optimization methods, differential equations, and chemical applications using `numpy`, `scipy.optimize`, and `matplotlib`.

## Lab Structure

- **Part 2 (`warmup_STUDENT.py`):** Implementation and verification of gradient descent, Newton's method, and SLSQP for convex and non-convex functions.
- **Part 3 (`kinetics_STUDENT.py`):** Curve fitting for a first-order chemical reaction rate constant ($k$) using noisy experimental data (`kinetics.csv`) and optimization (`kinetics.png`).
- **Part 4 (`equilibrium_STUDENT.py`):** Solving chemical equilibrium equations for the $H_2 + I_2 \rightleftharpoons 2HI$ system using root-finding (`newton`) and constrained optimization (`SLSQP`) (`equilibrium.png`).
- **Part 5 (`titration_STUDENT.py`):** Determining the equivalence point of an acid-base titration curve (`titration.csv`) by calculating numerical gradients (`np.gradient`) (`titration.png`).

## Environment Setup

Ensure you have the required Conda environment (`cspc`) active with the necessary dependencies:
```bash
conda activate cspc
pip install numpy scipy matplotlib



# Practical Work 3 - Heart Disease Data Analysis

Hello! In this project, I performed statistical analysis on the Cleveland heart disease dataset (`heart.csv`) using Python.

## What I Did:
- **Data Preparation:** Downloaded and set up the `heart.csv` dataset inside the `PW3` directory.
- **Data Visualization:** Generated histograms to visualize the distributions of continuous variables like age, cholesterol, and max heart rate (`heart_distributions.png`).
- **Normality Testing:** Applied the Shapiro-Wilk test to check if the continuous features follow a normal distribution.
- **Group Comparison:** Used an independent T-test to compare the maximum heart rate (`thalach`) between healthy individuals and patients with heart disease.
- **Correlation Analysis:** Explored the relationship between age and maximum heart rate and created a scatter plot (`age_thalach_scatter.png`).

## Files Included:
- `heart.csv`: The dataset used for the analysis.
- `heart_analysis.py`: The Python script containing all calculations, statistical tests, and plot generations.
- `heart_distributions.png` & `age_thalach_scatter.png`: The generated visualization plots.
-