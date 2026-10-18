"""
Week 3 Task: Advanced Data Exploration and Visualization
Dataset: Iris flower dataset (built into scikit-learn; no download required)

Purpose:
- Perform advanced exploratory data analysis (EDA)
- Explore distributions, outliers, correlations, pairwise patterns, and group differences
- Run example hypothesis tests
- Save high-quality visualizations and a text summary for the report

Requirements:
    pip install pandas numpy matplotlib seaborn scipy scikit-learn

Run:
    python week3_advanced_eda.py

Outputs are saved to a folder named "week3_outputs" beside this script.
Note: Statistical tests demonstrate a workflow and should be interpreted in context.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, f_oneway, kruskal
from sklearn.datasets import load_iris

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = Path(__file__).resolve().parent / "week3_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["savefig.bbox"] = "tight"


def load_data() -> pd.DataFrame:
    """Load the Iris dataset and return a readable DataFrame."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df = df.drop(columns=["target"])
    df = df.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    })
    return df


def save_figure(filename: str) -> None:
    """Save the current figure and close it."""
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def make_visualizations(df: pd.DataFrame) -> None:
    """Create several complementary plots and save them as PNG files."""
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # 1. Distributions
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feature in zip(axes.flat, features):
        sns.histplot(data=df, x=feature, hue="species", kde=True,
                     element="step", stat="density", common_norm=False, ax=ax)
        ax.set_title(f"Distribution of {feature.replace('_', ' ').title()}")
    fig.suptitle("Feature Distributions by Species", fontsize=15, y=1.02)
    fig.tight_layout()
    save_figure("01_feature_distributions.png")

    # 2. Correlation heatmap
    corr = df[features].corr(method="pearson")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, linewidths=0.5)
    plt.title("Pearson Correlation Heatmap")
    save_figure("02_correlation_heatmap.png")

    # 3. Pairplot
    grid = sns.pairplot(df, vars=features, hue="species", corner=True,
                        diag_kind="hist", plot_kws={"alpha": 0.75, "s": 35})
    grid.fig.suptitle("Pairwise Relationships by Species", y=1.02)
    grid.savefig(OUTPUT_DIR / "03_pairwise_relationships.png", dpi=220,
                 bbox_inches="tight")
    plt.close("all")

    # 4. Box plots for outlier and group comparison
    long_df = df.melt(id_vars="species", value_vars=features,
                      var_name="feature", value_name="measurement")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=long_df, x="feature", y="measurement", hue="species")
    plt.title("Feature Distributions and Potential Outliers by Species")
    plt.xlabel("Feature")
    plt.ylabel("Measurement (cm)")
    plt.xticks(rotation=15)
    save_figure("04_boxplots.png")

    # 5. Scatter plot with regression line
    plt.figure(figsize=(8, 6))
    sns.lmplot(data=df, x="petal_length", y="petal_width", hue="species",
               height=5.5, aspect=1.2, scatter_kws={"alpha": 0.8, "s": 45},
               ci=None)
    plt.suptitle("Petal Length vs Petal Width by Species", y=1.02)
    plt.savefig(OUTPUT_DIR / "05_petal_relationship.png", dpi=220,
                bbox_inches="tight")
    plt.close("all")

    # 6. Mean feature values by species
    means = df.groupby("species")[features].mean()
    scaled = (means - means.min()) / (means.max() - means.min())
    plt.figure(figsize=(10, 5))
    sns.heatmap(scaled, annot=means.round(2), fmt=".2f", cmap="YlGnBu",
                cbar_kws={"label": "Relative scale within each feature"})
    plt.title("Mean Feature Measurements by Species\n(annotations show original cm values)")
    plt.xlabel("Feature")
    plt.ylabel("Species")
    save_figure("06_species_mean_heatmap.png")


def run_statistics(df: pd.DataFrame) -> dict:
    """Run correlation and group-difference tests; return results for reporting."""
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    results = {}

    # Pearson correlation: petal length vs petal width
    r, p = pearsonr(df["petal_length"], df["petal_width"])
    results["petal_correlation"] = (r, p)

    # One-way ANOVA tests whether mean differs among species for each feature.
    anova_results = {}
    kruskal_results = {}
    for feature in features:
        groups = [group[feature].to_numpy()
                  for _, group in df.groupby("species", observed=True)]
        f_stat, p_value = f_oneway(*groups)
        h_stat, h_p = kruskal(*groups)
        anova_results[feature] = (f_stat, p_value)
        kruskal_results[feature] = (h_stat, h_p)

    results["anova"] = anova_results
    results["kruskal"] = kruskal_results
    return results


def write_summary(df: pd.DataFrame, stats: dict) -> None:
    """Write key descriptive statistics and test results to a text file."""
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    lines = [
        "WEEK 3: ADVANCED DATA EXPLORATION AND VISUALIZATION",
        "=" * 58,
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Missing values in dataset: {int(df.isna().sum().sum())}",
        "",
        "DESCRIPTIVE STATISTICS",
        df[features].describe().round(3).to_string(),
        "",
        "SPECIES COUNTS",
        df["species"].value_counts().to_string(),
        "",
        "PEARSON CORRELATION: PETAL LENGTH vs PETAL WIDTH",
        f"r = {stats['petal_correlation'][0]:.4f}",
        f"p-value = {stats['petal_correlation'][1]:.6g}",
        "",
        "ONE-WAY ANOVA BY SPECIES (H0: all species means are equal)",
    ]
    for feature, (f_stat, p_value) in stats["anova"].items():
        lines.append(f"{feature}: F = {f_stat:.4f}, p-value = {p_value:.6g}")
    lines.extend([
        "",
        "KRUSKAL-WALLIS BY SPECIES (non-parametric comparison)",
    ])
    for feature, (h_stat, p_value) in stats["kruskal"].items():
        lines.append(f"{feature}: H = {h_stat:.4f}, p-value = {p_value:.6g}")
    lines.extend([
        "",
        "INTERPRETATION NOTES",
        "- A small p-value (commonly < 0.05) is evidence against the test's null hypothesis.",
        "- Statistical significance does not by itself establish practical importance or causation.",
        "- ANOVA assumptions include independent observations, approximately normal residuals, and similar group variances.",
        "- Kruskal-Wallis provides a rank-based alternative but does not identify which groups differ.",
        "- The Iris dataset is small and educational; avoid generalizing its patterns to all flowers.",
        "",
        "Visualizations saved as PNG files in this output folder.",
    ])
    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_data()
    print("Dataset preview:")
    print(df.head(), "\n")
    print("Dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDescriptive statistics:")
    print(df.describe(numeric_only=True).round(3))

    make_visualizations(df)
    stats = run_statistics(df)
    write_summary(df, stats)

    print("\nPearson correlation (petal length vs petal width):")
    print(f"r = {stats['petal_correlation'][0]:.4f}, "
          f"p = {stats['petal_correlation'][1]:.6g}")
    print("\nANOVA results:")
    for feature, (f_stat, p_value) in stats["anova"].items():
        print(f"{feature}: F={f_stat:.4f}, p={p_value:.6g}")

    print(f"\nFinished. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
