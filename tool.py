import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ============================================================
# TOOL 1 — DATA VISUALIZATION
# Generates plots based on LLM-suggested plot type & columns
# ============================================================

def visualize_data(df: pd.DataFrame, plot_type: str, columns: list, hue: str = None):
    """
    Generate a plot based on the LLM's suggestion.

    Args:
        df        : The user's uploaded DataFrame
        plot_type : e.g. 'histogram', 'scatter plot', 'box plot', 'bar plot', 'heatmap', 'pair plot'
        columns   : List of column names to use
        hue       : Optional categorical column for color grouping
    """
    plot_type = plot_type.lower().strip()
    plt.figure(figsize=(10, 6))

    try:
        if "histogram" in plot_type or "hist" in plot_type:
            sns.histplot(df[columns[0]], kde=True)
            plt.title(f"Histogram — {columns[0]}")

        elif "scatter" in plot_type:
            sns.scatterplot(data=df, x=columns[0], y=columns[1], hue=hue)
            plt.title(f"Scatter Plot — {columns[0]} vs {columns[1]}")

        elif "box" in plot_type:
            sns.boxplot(data=df, x=columns[0] if len(columns) > 1 else None,
                        y=columns[-1], hue=hue)
            plt.title(f"Box Plot — {columns[-1]}")

        elif "bar" in plot_type:
            sns.barplot(data=df, x=columns[0], y=columns[1], hue=hue)
            plt.title(f"Bar Plot — {columns[0]} vs {columns[1]}")

        elif "heatmap" in plot_type:
            numeric_df = df[columns] if columns else df.select_dtypes(include="number")
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")

        elif "pair" in plot_type:
            sns.pairplot(df[columns] if columns else df.select_dtypes(include="number"),
                         hue=hue)
            plt.suptitle("Pair Plot", y=1.02)

        elif "count" in plot_type:
            sns.countplot(data=df, x=columns[0], hue=hue)
            plt.title(f"Count Plot — {columns[0]}")

        elif "violin" in plot_type:
            sns.violinplot(data=df, x=columns[0] if len(columns) > 1 else None,
                           y=columns[-1], hue=hue)
            plt.title(f"Violin Plot — {columns[-1]}")

        else:
            print(f"⚠️  Plot type '{plot_type}' not recognized. Supported: histogram, scatter, box, bar, heatmap, pair, count, violin.")
            plt.close()
            return

        plt.tight_layout()
        plt.show()
        print(f"✅ Plot generated: {plot_type.title()} for {columns}")

    except Exception as e:
        plt.close()
        print(f"❌ Error generating plot: {e}")


# ============================================================
# TOOL 2 — DATA ENGINEERING (Fill Nulls + Detailed EDA)
# ============================================================

def data_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills null values and prints a detailed EDA report.

    - Numeric columns  → filled with median
    - Categorical cols → filled with mode

    Returns the cleaned DataFrame.
    """
    print("=" * 50)
    print("📋 DETAILED EDA REPORT")
    print("=" * 50)

    # Shape
    print(f"\n🔷 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Data types
    print("\n🔷 Data Types:")
    print(df.dtypes.to_string())

    # Null values before cleaning
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    print(f"\n🔷 Null Values Found: {len(null_cols)} column(s) with missing data")
    if not null_cols.empty:
        print(null_cols.to_string())

    # Fill nulls
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
                print(f"   ✅ Filled '{col}' (numeric) with median: {fill_val:.4f}")
            else:
                fill_val = df[col].mode()[0]
                df[col].fillna(fill_val, inplace=True)
                print(f"   ✅ Filled '{col}' (categorical) with mode: '{fill_val}'")

    # Descriptive statistics
    print("\n🔷 Descriptive Statistics (Numeric):")
    print(df.describe().to_string())

    # Categorical columns — value counts
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        print("\n🔷 Categorical Column Value Counts:")
        for col in cat_cols:
            print(f"\n  📌 {col}:")
            print(df[col].value_counts().to_string())

    # Duplicate rows
    dupes = df.duplicated().sum()
    print(f"\n🔷 Duplicate Rows: {dupes}")

    print("\n" + "=" * 50)
    print("✅ Data Engineering Complete — Cleaned DataFrame returned.")
    print("=" * 50)

    return df


# ============================================================
# TOOL 3 — SHOW DATASET HEAD
# ============================================================

def show_head(df: pd.DataFrame, n: int = 5):
    """
    Display the first n rows of the dataset.

    Args:
        df : The user's uploaded DataFrame
        n  : Number of rows to show (default: 5)
    """
    print(f"\n👀 First {n} rows of the dataset:")
    print("-" * 50)
    print(df.head(n).to_string())
    print("-" * 50)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# ============================================================
# TOOL 4 — CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df: pd.DataFrame, top_n: int = 5):
    """
    Generate a correlation heatmap and list the top N most correlated pairs.

    Args:
        df    : The user's uploaded DataFrame
        top_n : Number of top correlated pairs to display (default: 5)
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        print("⚠️  Need at least 2 numeric columns for correlation analysis.")
        return

    corr_matrix = numeric_df.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, square=True)
    plt.title("📊 Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Top N correlated pairs (excluding self-correlations)
    corr_pairs = (
        corr_matrix
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    corr_pairs["Abs Correlation"] = corr_pairs["Correlation"].abs()
    top_pairs = corr_pairs.sort_values("Abs Correlation", ascending=False).head(top_n)

    print(f"\n🔗 Top {top_n} Most Correlated Feature Pairs:")
    print("-" * 50)
    for _, row in top_pairs.iterrows():
        direction = "positive" if row["Correlation"] > 0 else "negative"
        print(f"  {row['Feature 1']} ↔ {row['Feature 2']}: {row['Correlation']:.4f} ({direction})")
    print("-" * 50)


# ============================================================
# TOOL 5 — OUTLIER DETECTION (IQR Method)
# ============================================================

def detect_outliers(df: pd.DataFrame, show_plots: bool = True) -> pd.DataFrame:
    """
    Detect outliers in all numeric columns using the IQR method.
    Optionally renders box plots for visual inspection.

    Args:
        df          : The user's uploaded DataFrame
        show_plots  : Whether to render box plots (default: True)

    Returns:
        DataFrame summarizing outlier counts per column.
    """
    numeric_df = df.select_dtypes(include="number")
    summary = []

    print("\n🎯 OUTLIER DETECTION REPORT (IQR Method)")
    print("=" * 50)

    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = (numeric_df[col] < lower) | (numeric_df[col] > upper)
        outlier_count = outlier_mask.sum()
        outlier_pct = (outlier_count / len(df)) * 100

        summary.append({
            "Column": col,
            "Outliers": outlier_count,
            "Outlier %": round(outlier_pct, 2),
            "Lower Bound": round(lower, 4),
            "Upper Bound": round(upper, 4)
        })

        flag = "⚠️ " if outlier_count > 0 else "✅"
        print(f"  {flag} {col}: {outlier_count} outliers ({outlier_pct:.1f}%)  |  Bounds: [{lower:.2f}, {upper:.2f}]")

    print("=" * 50)

    # Box plots for visual inspection
    if show_plots and len(numeric_df.columns) > 0:
        n_cols = min(3, len(numeric_df.columns))
        n_rows = (len(numeric_df.columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()

        for i, col in enumerate(numeric_df.columns):
            sns.boxplot(y=df[col], ax=axes[i], color="skyblue")
            axes[i].set_title(f"{col}")

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("🎯 Outlier Box Plots (IQR)", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    summary_df = pd.DataFrame(summary)
    return summary_df
