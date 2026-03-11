import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ============================================================
# TOOL 1 — DATA VISUALIZATION
# Generates plots based on LLM-suggested plot type & columns
# Returns a matplotlib Figure (caller decides how to display).
# ============================================================

def visualize_data(df: pd.DataFrame, plot_type: str, columns: list, hue: str = None):
    """
    Generate a plot based on the LLM's suggestion.

    Args:
        df        : The user's uploaded DataFrame
        plot_type : e.g. 'histogram', 'scatter plot', 'box plot', 'bar plot', 'heatmap', 'pair plot'
        columns   : List of column names to use
        hue       : Optional categorical column for color grouping

    Returns:
        (fig, message)  — matplotlib Figure + status message string
    """
    plot_type = plot_type.lower().strip()
    fig = None

    try:
        if "histogram" in plot_type or "hist" in plot_type:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[columns[0]], kde=True, ax=ax)
            ax.set_title(f"Histogram — {columns[0]}")

        elif "scatter" in plot_type:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=columns[0], y=columns[1], hue=hue, ax=ax)
            ax.set_title(f"Scatter Plot — {columns[0]} vs {columns[1]}")

        elif "box" in plot_type:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x=columns[0] if len(columns) > 1 else None,
                        y=columns[-1], hue=hue, ax=ax)
            ax.set_title(f"Box Plot — {columns[-1]}")

        elif "bar" in plot_type:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df, x=columns[0], y=columns[1], hue=hue, ax=ax)
            ax.set_title(f"Bar Plot — {columns[0]} vs {columns[1]}")

        elif "heatmap" in plot_type:
            fig, ax = plt.subplots(figsize=(10, 7))
            numeric_df = df[columns] if columns else df.select_dtypes(include="number")
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")

        elif "pair" in plot_type:
            subset = df[columns] if columns else df.select_dtypes(include="number")
            g = sns.pairplot(subset, hue=hue)
            g.figure.suptitle("Pair Plot", y=1.02)
            fig = g.figure

        elif "count" in plot_type:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df, x=columns[0], hue=hue, ax=ax)
            ax.set_title(f"Count Plot — {columns[0]}")

        elif "violin" in plot_type:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=df, x=columns[0] if len(columns) > 1 else None,
                           y=columns[-1], hue=hue, ax=ax)
            ax.set_title(f"Violin Plot — {columns[-1]}")

        else:
            return None, f"⚠️  Plot type '{plot_type}' not recognized. Supported: histogram, scatter, box, bar, heatmap, pair, count, violin."

        if fig is not None:
            fig.tight_layout()

        return fig, f"✅ Plot generated: {plot_type.title()} for {columns}"

    except Exception as e:
        if fig is not None:
            plt.close(fig)
        return None, f"❌ Error generating plot: {e}"


# ============================================================
# TOOL 2 — DATA ENGINEERING (Fill Nulls + Detailed EDA)
# Returns (cleaned_df, report_string)
# ============================================================

def data_engineering(df: pd.DataFrame):
    """
    Fills null values and builds a detailed EDA report.

    - Numeric columns  → filled with median
    - Categorical cols → filled with mode

    Returns:
        (cleaned_df, report_string)
    """
    lines = []
    lines.append("=" * 50)
    lines.append("📋 DETAILED EDA REPORT")
    lines.append("=" * 50)

    # Shape
    lines.append(f"\n🔷 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Data types
    lines.append("\n🔷 Data Types:")
    lines.append(df.dtypes.to_string())

    # Null values before cleaning
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    lines.append(f"\n🔷 Null Values Found: {len(null_cols)} column(s) with missing data")
    if not null_cols.empty:
        lines.append(null_cols.to_string())

    # Fill nulls
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
                lines.append(f"   ✅ Filled '{col}' (numeric) with median: {fill_val:.4f}")
            else:
                fill_val = df[col].mode()[0]
                df[col].fillna(fill_val, inplace=True)
                lines.append(f"   ✅ Filled '{col}' (categorical) with mode: '{fill_val}'")

    # Descriptive statistics
    lines.append("\n🔷 Descriptive Statistics (Numeric):")
    lines.append(df.describe().to_string())

    # Categorical columns — value counts
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        lines.append("\n🔷 Categorical Column Value Counts:")
        for col in cat_cols:
            lines.append(f"\n  📌 {col}:")
            lines.append(df[col].value_counts().to_string())

    # Duplicate rows
    dupes = df.duplicated().sum()
    lines.append(f"\n🔷 Duplicate Rows: {dupes}")

    lines.append("\n" + "=" * 50)
    lines.append("✅ Data Engineering Complete — Cleaned DataFrame returned.")
    lines.append("=" * 50)

    return df, "\n".join(lines)


# ============================================================
# TOOL 3 — SHOW DATASET HEAD
# Returns head DataFrame
# ============================================================

def show_head(df: pd.DataFrame, n: int = 5):
    """
    Return the first n rows of the dataset.

    Args:
        df : The user's uploaded DataFrame
        n  : Number of rows to show (default: 5)

    Returns:
        (head_df, info_string)
    """
    head_df = df.head(n)
    info = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"
    return head_df, info


# ============================================================
# TOOL 4 — CORRELATION ANALYSIS
# Returns (fig, top_pairs_df)
# ============================================================

def correlation_analysis(df: pd.DataFrame, top_n: int = 5):
    """
    Generate a correlation heatmap and list the top N most correlated pairs.

    Args:
        df    : The user's uploaded DataFrame
        top_n : Number of top correlated pairs to display (default: 5)

    Returns:
        (fig, top_pairs_df, report_string)
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None, None, "⚠️  Need at least 2 numeric columns for correlation analysis."

    corr_matrix = numeric_df.corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, square=True, ax=ax)
    ax.set_title("📊 Correlation Heatmap")
    fig.tight_layout()

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

    lines = []
    lines.append(f"\n🔗 Top {top_n} Most Correlated Feature Pairs:")
    lines.append("-" * 50)
    for _, row in top_pairs.iterrows():
        direction = "positive" if row["Correlation"] > 0 else "negative"
        lines.append(f"  {row['Feature 1']} ↔ {row['Feature 2']}: {row['Correlation']:.4f} ({direction})")
    lines.append("-" * 50)

    return fig, top_pairs, "\n".join(lines)


# ============================================================
# TOOL 5 — OUTLIER DETECTION (IQR Method)
# Returns (fig, summary_df, report_string)
# ============================================================

def detect_outliers(df: pd.DataFrame, show_plots: bool = True):
    """
    Detect outliers in all numeric columns using the IQR method.
    Optionally renders box plots for visual inspection.

    Args:
        df          : The user's uploaded DataFrame
        show_plots  : Whether to render box plots (default: True)

    Returns:
        (fig_or_None, summary_df, report_string)
    """
    numeric_df = df.select_dtypes(include="number")
    summary = []
    lines = []

    lines.append("\n🎯 OUTLIER DETECTION REPORT (IQR Method)")
    lines.append("=" * 50)

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
        lines.append(f"  {flag} {col}: {outlier_count} outliers ({outlier_pct:.1f}%)  |  Bounds: [{lower:.2f}, {upper:.2f}]")

    lines.append("=" * 50)

    # Box plots for visual inspection
    fig = None
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

        fig.suptitle("🎯 Outlier Box Plots (IQR)", fontsize=14, y=1.02)
        fig.tight_layout()

    summary_df = pd.DataFrame(summary)
    return fig, summary_df, "\n".join(lines)


# ============================================================
# TOOL 6 — DATASET STATISTICS
# Returns report_string
# ============================================================

def dataset_stats(df: pd.DataFrame):
    """
    Build a comprehensive statistical overview of the dataset.
    Covers shape, data types, numeric stats, categorical stats,
    memory usage, and unique value counts per column.

    Returns:
        report_string
    """
    lines = []

    lines.append("\n" + "=" * 60)
    lines.append("📊 DATASET STATISTICS REPORT")
    lines.append("=" * 60)

    # Shape
    lines.append(f"\n📐 Shape       : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

    # Memory usage
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_kb = mem_bytes / 1024
    lines.append(f"💾 Memory Usage: {mem_kb:.2f} KB ({mem_bytes:,} bytes)")

    # Data types summary
    dtype_counts = df.dtypes.value_counts()
    lines.append(f"\n🔠 Data Types Breakdown:")
    for dtype, count in dtype_counts.items():
        lines.append(f"   {str(dtype):<15} → {count} column(s)")

    # Per-column overview table
    lines.append(f"\n📋 Column Overview:")
    lines.append(f"  {'Column':<20} {'Dtype':<12} {'Non-Null':<10} {'Nulls':<8} {'Unique':<8}")
    lines.append("  " + "-" * 62)
    for col in df.columns:
        non_null = df[col].notna().sum()
        nulls    = df[col].isna().sum()
        unique   = df[col].nunique()
        dtype    = str(df[col].dtype)
        lines.append(f"  {col:<20} {dtype:<12} {non_null:<10} {nulls:<8} {unique:<8}")

    # Numeric statistics
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        lines.append(f"\n🔢 Numeric Column Statistics:")
        stats = numeric_df.describe().T
        stats["skewness"] = numeric_df.skew()
        stats["kurtosis"] = numeric_df.kurt()
        lines.append(stats.to_string())

    # Categorical statistics
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        lines.append(f"\n🔤 Categorical Column Statistics:")
        lines.append(f"  {'Column':<20} {'Unique Values':<15} {'Top Value':<20} {'Top Freq'}")
        lines.append("  " + "-" * 65)
        for col in cat_cols:
            unique_count = df[col].nunique()
            top_val      = df[col].value_counts().idxmax() if not df[col].dropna().empty else "N/A"
            top_freq     = df[col].value_counts().max()    if not df[col].dropna().empty else 0
            lines.append(f"  {col:<20} {unique_count:<15} {str(top_val):<20} {top_freq}")

    # Duplicate rows
    dupes = df.duplicated().sum()
    if len(df) > 0:
        lines.append(f"\n🔁 Duplicate Rows: {dupes} ({(dupes / len(df) * 100):.2f}%)")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


# ============================================================
# TOOL 7 — MISSING VALUE HANDLER
# Returns (cleaned_df, report_string, before_summary_df)
# ============================================================

def handle_missing_values(df: pd.DataFrame):
    """
    Detect missing values column by column, explain the handling
    strategy chosen for each, apply it, and show a before/after summary.

    Strategy logic:
    - Numeric   → if skewed (|skew| > 0.5): fill with MEDIAN
                  else: fill with MEAN
    - Categorical / Object / Bool → fill with MODE (most frequent value)
    - Datetime  → fill with FORWARD FILL (ffill)
    - If >50% of rows are null → DROP the column entirely

    Returns:
        (cleaned_df, report_string)
    """
    lines = []

    lines.append("\n" + "=" * 60)
    lines.append("🧹 MISSING VALUE HANDLER")
    lines.append("=" * 60)

    null_counts = df.isnull().sum()
    missing_cols = null_counts[null_counts > 0]

    if missing_cols.empty:
        lines.append("\n✅ No missing values detected! Dataset is already clean.")
        lines.append("=" * 60)
        return df, "\n".join(lines)

    total_rows = len(df)
    lines.append(f"\n🔍 Found {len(missing_cols)} column(s) with missing values:\n")

    before_summary = []
    actions_taken  = []

    for col in missing_cols.index:
        null_count = missing_cols[col]
        null_pct   = (null_count / total_rows) * 100
        dtype      = df[col].dtype

        before_summary.append({
            "Column":   col,
            "Dtype":    str(dtype),
            "Nulls":    null_count,
            "Null %":   f"{null_pct:.1f}%"
        })

        lines.append(f"  📌 Column : '{col}'")
        lines.append(f"     Dtype  : {dtype}")
        lines.append(f"     Missing: {null_count} / {total_rows} rows  ({null_pct:.1f}%)")

        # ── Strategy Decision ──────────────────────────────────
        if null_pct > 50:
            strategy = "DROP COLUMN"
            reason   = f"{null_pct:.1f}% missing — too much data lost to fill reliably"
            df.drop(columns=[col], inplace=True)

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            strategy = "FORWARD FILL (ffill)"
            reason   = "Datetime — forward fill preserves time-series continuity"
            df[col].ffill(inplace=True)
            df[col].bfill(inplace=True)  # catch leading NaTs

        elif pd.api.types.is_numeric_dtype(dtype):
            skewness = df[col].skew()
            if abs(skewness) > 0.5:
                fill_val = df[col].median()
                strategy = f"MEDIAN  ({fill_val:.4f})"
                reason   = f"Skewness = {skewness:.2f} — median is robust to skewed/outlier-heavy data"
            else:
                fill_val = df[col].mean()
                strategy = f"MEAN    ({fill_val:.4f})"
                reason   = f"Skewness = {skewness:.2f} — data is roughly symmetric, mean is appropriate"
            df[col].fillna(fill_val, inplace=True)

        else:
            # Categorical / object / bool
            fill_val = df[col].mode()[0] if not df[col].dropna().empty else "Unknown"
            strategy = f"MODE    ('{fill_val}')"
            reason   = "Categorical — mode (most frequent value) is the safest imputation"
            df[col].fillna(fill_val, inplace=True)

        lines.append(f"     Strategy: {strategy}")
        lines.append(f"     Why     : {reason}\n")

        actions_taken.append({
            "Column":   col,
            "Strategy": strategy,
            "Reason":   reason
        })

    # ── Before / After Summary Table ──────────────────────────
    lines.append("-" * 60)
    lines.append("📊 BEFORE / AFTER SUMMARY")
    lines.append("-" * 60)
    lines.append(f"  {'Column':<20} {'Before (Nulls)':<15} {'After (Nulls)'}")
    lines.append("  " + "-" * 45)
    for item in before_summary:
        col = item["Column"]
        before_nulls = item["Nulls"]
        if col in df.columns:
            after_nulls = df[col].isnull().sum()
            status = "✅ Clean" if after_nulls == 0 else f"⚠️  {after_nulls} remain"
        else:
            status = "🗑️  Dropped"
        lines.append(f"  {col:<20} {str(before_nulls):<15} {status}")

    remaining = df.isnull().sum().sum()
    lines.append(f"\n{'✅ All missing values handled!' if remaining == 0 else f'⚠️  {remaining} missing values still remain.'}")
    lines.append("=" * 60)

    return df, "\n".join(lines)
