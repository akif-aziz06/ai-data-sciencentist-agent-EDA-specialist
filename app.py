import streamlit as st
import pandas as pd
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent import load_uploaded_dataset, _build_metadata
from tool import (
    show_head,
    dataset_stats,
    handle_missing_values,
    data_engineering,
    correlation_analysis,
    detect_outliers,
    visualize_data,
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI EDA Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium look ───────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        color: #a0aec0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 8px;
    }

    /* Card-like containers */
    .stExpander {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 12px;
        padding: 16px;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Success/info/warning boxes */
    .stAlert {
        border-radius: 10px;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 10px;
    }

    /* Header styling */
    h1, h2, h3 {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Code blocks */
    code {
        background-color: rgba(102, 126, 234, 0.15);
        border-radius: 4px;
        padding: 2px 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar — file upload ─────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI EDA Agent")
    st.markdown("**Upload any CSV/XLSX dataset** and get a full automated EDA powered by GPT-4o-mini.")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload Dataset (max 10 MB)",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV and Excel files up to 10 MB.",
    )

    st.markdown("---")
    st.markdown(
        "**Built with** 🐍 Python  •  📊 Seaborn  •  🤖 LangChain  •  🎈 Streamlit"
    )


# ── Load data into session state ──────────────────────────────
if uploaded_file is not None:
    # Only load if new file or first load
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_file_id") != file_id:
        with st.spinner("Loading dataset..."):
            metadata, df = load_uploaded_dataset(uploaded_file)
            st.session_state["df_original"] = df.copy()
            st.session_state["df"] = df
            st.session_state["metadata"] = metadata
            st.session_state["_file_id"] = file_id
            # Reset processed flags
            for key in ["missing_done", "engineering_done", "llm_response"]:
                st.session_state.pop(key, None)


# ── Main content ──────────────────────────────────────────────
if "df" not in st.session_state:
    st.markdown("# 🤖 AI Data Scientist — EDA Specialist")
    st.markdown("### Upload a dataset from the sidebar to get started.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📊 7 EDA Tools")
        st.markdown("Statistics, missing values, engineering, outliers, correlation, and more.")
    with col2:
        st.markdown("#### 🤖 AI-Powered Plots")
        st.markdown("GPT-4o-mini analyzes your data and recommends the best visualizations.")
    with col3:
        st.markdown("#### 📂 Any Dataset")
        st.markdown("Upload CSV or XLSX up to 10 MB. Works on unseen data.")

    st.stop()


# ── Tabs ──────────────────────────────────────────────────────
df = st.session_state["df"]

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📂 Preview",
    "📊 Statistics",
    "🧹 Missing Values",
    "🔧 Engineering",
    "🎯 Outliers",
    "🔗 Correlation",
    "🤖 AI Visualizations",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Dataset Preview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("## 📂 Dataset Preview")

    head_df, info = show_head(df, n=10)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]}")
    mem_kb = df.memory_usage(deep=True).sum() / 1024
    col3.metric("Memory", f"{mem_kb:.1f} KB")

    st.markdown("### First 10 Rows")
    st.dataframe(head_df, use_container_width=True)

    with st.expander("📋 Column Data Types"):
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null": df.notna().sum().values,
            "Nulls": df.isna().sum().values,
            "Unique": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Dataset Statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("## 📊 Dataset Statistics")

    report = dataset_stats(df)
    st.code(report, language="text")

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        with st.expander("🔢 Numeric Statistics Table", expanded=True):
            stats = numeric_df.describe().T
            stats["skewness"] = numeric_df.skew()
            stats["kurtosis"] = numeric_df.kurt()
            st.dataframe(stats, use_container_width=True)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        with st.expander("🔤 Categorical Statistics", expanded=True):
            cat_data = []
            for col in cat_cols:
                cat_data.append({
                    "Column": col,
                    "Unique": df[col].nunique(),
                    "Top Value": df[col].value_counts().idxmax() if not df[col].dropna().empty else "N/A",
                    "Top Freq": df[col].value_counts().max() if not df[col].dropna().empty else 0,
                })
            st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Missing Values
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("## 🧹 Missing Value Handler")

    null_total = df.isnull().sum().sum()
    if null_total == 0 and not st.session_state.get("missing_done"):
        st.success("✅ No missing values detected! Dataset is already clean.")
    else:
        if not st.session_state.get("missing_done"):
            st.warning(f"⚠️ Found **{null_total}** missing values across the dataset.")

            if st.button("🧹 Handle Missing Values", type="primary"):
                with st.spinner("Applying smart imputation strategies..."):
                    cleaned_df, report = handle_missing_values(df.copy())
                    st.session_state["df"] = cleaned_df
                    st.session_state["missing_done"] = True
                    st.session_state["missing_report"] = report
                    st.rerun()
        else:
            st.success("✅ Missing values have been handled!")

    if st.session_state.get("missing_report"):
        st.code(st.session_state["missing_report"], language="text")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Data Engineering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("## 🔧 Data Engineering & EDA")

    if not st.session_state.get("engineering_done"):
        if st.button("🔧 Run Data Engineering", type="primary"):
            with st.spinner("Running data engineering pipeline..."):
                cleaned_df, report = data_engineering(df.copy())
                st.session_state["df"] = cleaned_df
                st.session_state["engineering_done"] = True
                st.session_state["engineering_report"] = report
                st.rerun()
    else:
        st.success("✅ Data Engineering complete!")

    if st.session_state.get("engineering_report"):
        st.code(st.session_state["engineering_report"], language="text")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — Outlier Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("## 🎯 Outlier Detection (IQR Method)")

    df_current = st.session_state["df"]
    fig, summary_df, report = detect_outliers(df_current)

    st.code(report, language="text")

    if summary_df is not None and not summary_df.empty:
        st.markdown("### Summary Table")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if fig is not None:
        st.markdown("### Box Plots")
        st.pyplot(fig)
        plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — Correlation Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab6:
    st.markdown("## 🔗 Correlation Analysis")

    df_current = st.session_state["df"]
    fig, top_pairs, report = correlation_analysis(df_current)

    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)

    st.code(report, language="text")

    if top_pairs is not None and not top_pairs.empty:
        st.markdown("### Top Correlated Pairs")
        st.dataframe(top_pairs, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 7 — AI Visualizations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab7:
    st.markdown("## 🤖 AI-Powered Visualizations")
    st.markdown("GPT-4o-mini analyzes your dataset and recommends the best plots.")

    if st.button("🚀 Generate AI Visualizations", type="primary"):
        # Load env & build chain
        load_dotenv(dotenv_path=".evv")

        llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1500)

        ai_prompt = ChatPromptTemplate.from_messages([
            ('system', """You are an expert Data Scientist and EDA Specialist.

A user has uploaded an unknown dataset. I will provide you its metadata: column names, data types, a summary (df.info()), and a preview (df.head()).

Your job is to analyze this metadata and recommend the most insightful plots for 3 types of EDA analysis.
For each category, recommend 1 or 2 plots — pick the count that best suits the dataset (don't force 2 if 1 is enough).

You MUST respond with ONLY valid JSON (no markdown, no code fences, no extra text).
Use this exact structure:

{{
  "univariate": [
    {{"plot_type": "<plot type>", "columns": ["<column_name>"], "hue": null, "reason": "<1-sentence reason>"}}
  ],
  "bivariate": [
    {{"plot_type": "<plot type>", "columns": ["<col1>", "<col2>"], "hue": "<optional_col_or_null>", "reason": "<1-sentence reason>"}}
  ],
  "multivariate": [
    {{"plot_type": "<plot type>", "columns": ["<col1>", "<col2>", "<col3>"], "hue": "<optional_col_or_null>", "reason": "<1-sentence reason>"}}
  ]
}}

Rules:
- Use generic plot types: histogram, scatter plot, box plot, bar plot, heatmap, pair plot, count plot, violin plot
- Always use EXACT column names from the provided metadata
- "hue" must be a categorical column name or null
- "columns" must be a list of exact column name strings
- Each category ("univariate", "bivariate", "multivariate") must have 1 or 2 plot objects
- Base all recommendations strictly on the provided metadata — do NOT invent columns"""),
            ('human', "{dataset_info}")
        ])

        ai_chain = ai_prompt | llm_model | StrOutputParser()

        # Rebuild metadata from current (possibly cleaned) dataframe
        df_current = st.session_state["df"]
        current_metadata = _build_metadata(df_current)

        with st.spinner("🤖 Asking GPT-4o-mini for the best plots..."):
            try:
                response = ai_chain.invoke({"dataset_info": current_metadata})
                st.session_state["llm_response"] = response
            except Exception as e:
                st.error(f"❌ LLM call failed: {e}")
                st.stop()

    # ── Render recommendations ────────────────────────────────
    if st.session_state.get("llm_response"):
        response = st.session_state["llm_response"]

        with st.expander("🔍 Raw LLM Response", expanded=False):
            st.code(response, language="json")

        try:
            recommendations = json.loads(response)
            df_current = st.session_state["df"]

            for category in ["univariate", "bivariate", "multivariate"]:
                plots = recommendations.get(category, [])
                if not plots:
                    continue

                emoji = {"univariate": "🔹", "bivariate": "🔸", "multivariate": "🔶"}[category]
                st.markdown(f"### {emoji} {category.title()} Analysis ({len(plots)} plot{'s' if len(plots) > 1 else ''})")

                for i, plot in enumerate(plots):
                    plot_type = plot.get("plot_type", "histogram")
                    columns = plot.get("columns", [])
                    hue = plot.get("hue")
                    reason = plot.get("reason", "")

                    if hue == "null" or hue == "None" or hue is None:
                        hue = None

                    st.markdown(f"**{plot_type.title()}** — Columns: `{', '.join(columns)}`"
                                + (f" — Hue: `{hue}`" if hue else ""))
                    st.caption(f"💡 {reason}")

                    fig, msg = visualize_data(df_current, plot_type, columns, hue=hue)
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning(msg)

            st.success("✅ All AI-recommended visualizations generated!")

        except json.JSONDecodeError:
            st.error("❌ Failed to parse LLM response as JSON. Try again — the LLM occasionally adds extra text.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
