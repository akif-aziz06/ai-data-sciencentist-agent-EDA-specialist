import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent import load_dataset
from tool import (
    show_head,
    dataset_stats,
    handle_missing_values,
    data_engineering,
    correlation_analysis,
    detect_outliers,
    visualize_data,
)

load_dotenv(dotenv_path=".evv")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1500)


prompt = ChatPromptTemplate.from_messages([
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

chain = prompt | llm | StrOutputParser()


# ══════════════════════════════════════════════════════════════
#                    PERFECT EDA PIPELINE
# ══════════════════════════════════════════════════════════════

# ── Step 1: Load dataset (metadata string + raw DataFrame)
metadata, df = load_dataset()

# ── Step 2: Preview the dataset
print("\n" + "=" * 50)
print("STEP 1 — Dataset Preview")
print("=" * 50)
show_head(df)

# ── Step 3: Dataset Statistics
print("\n" + "=" * 50)
print("STEP 2 — Dataset Statistics")
print("=" * 50)
dataset_stats(df)

# ── Step 4: Missing Value Handler (smart strategy: skew-based median/mean, mode, ffill, or drop)
print("\n" + "=" * 50)
print("STEP 3 — Missing Value Handler")
print("=" * 50)
df = handle_missing_values(df)

# ── Step 5: Data Engineering — detailed EDA on clean data
print("\n" + "=" * 50)
print("STEP 4 — Data Engineering & EDA")
print("=" * 50)
df = data_engineering(df)

# ── Step 6: Outlier Detection on clean data
print("\n" + "=" * 50)
print("STEP 5 — Outlier Detection")
print("=" * 50)
detect_outliers(df)

# ── Step 7: Correlation Analysis on clean data
print("\n" + "=" * 50)
print("STEP 6 — Correlation Analysis")
print("=" * 50)
correlation_analysis(df)

# ── Step 8: Ask LLM for structured plot recommendations (JSON)
print("\n" + "=" * 50)
print("STEP 7 — Asking GPT-4o-mini for Plot Advice...")
print("=" * 50)
response = chain.invoke({"dataset_info": metadata})
print(response)

# ── Step 9: Auto-Visualize — parse LLM JSON and generate all recommended plots
print("\n" + "=" * 50)
print("STEP 8 — Auto-Generating Recommended Visualizations")
print("=" * 50)

try:
    recommendations = json.loads(response)

    for category in ["univariate", "bivariate", "multivariate"]:
        plots = recommendations.get(category, [])
        if not plots:
            print(f"⚠️  No {category} plots recommended.")
            continue

        print(f"\n📊 Generating {category.upper()} plots ({len(plots)} recommended)...\n")

        for i, plot in enumerate(plots, 1):
            plot_type = plot.get("plot_type", "histogram")
            columns = plot.get("columns", [])
            hue = plot.get("hue")
            reason = plot.get("reason", "")

            # Convert null string to None
            if hue == "null" or hue == "None":
                hue = None

            print(f"  🎨 [{category.title()} #{i}] {plot_type.title()} — Columns: {columns}" +
                  (f" — Hue: {hue}" if hue else ""))
            print(f"     Reason: {reason}")

            visualize_data(df, plot_type, columns, hue=hue)

    print("\n" + "=" * 50)
    print("✅ All visualizations generated successfully!")
    print("=" * 50)

except json.JSONDecodeError as e:
    print(f"\n❌ Failed to parse LLM response as JSON: {e}")
    print("Raw response was:")
    print(response)
    print("\nTip: Re-run the script — the LLM occasionally adds extra text around the JSON.")
except Exception as e:
    print(f"\n❌ Error during auto-visualization: {e}")