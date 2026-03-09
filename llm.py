import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent import load_dataset
from tool import show_head, data_engineering, correlation_analysis, detect_outliers, visualize_data

load_dotenv(dotenv_path=".evv")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=800)


prompt = ChatPromptTemplate.from_messages([
    ('system', """You are an expert Data Scientist and EDA Specialist.

A user has uploaded an unknown dataset. I will provide you its metadata: column names, data types, a summary (df.info()), and a preview (df.head()).

Your job is to analyze this metadata and recommend the most insightful plots for 3 types of EDA analysis.
Do NOT assume any specific library — recommend generic plot types (e.g., histogram, scatter plot, box plot, heatmap, pair plot).
Always reference the actual column names from the provided metadata.

Respond in this exact structured format:

## 🔹 Univariate Analysis (1 variable at a time)
- Plot: <plot type>
- Columns: <exact column name(s) from the dataset>
- Reason: <1-sentence reason based on the column's data type and distribution>

## 🔸 Bivariate Analysis (relationship between 2 variables)
- Plot: <plot type>
- Columns: <exact column name(s) from the dataset>
- Reason: <1-sentence reason explaining what relationship this reveals>

## 🔶 Multivariate Analysis (patterns across 3+ variables)
- Plot: <plot type>
- Columns: <exact column name(s) from the dataset>
- Reason: <1-sentence reason explaining what multi-variable insight this uncovers>

Base all recommendations strictly on the provided metadata. Do not invent or assume any columns."""),
    ('human', "{dataset_info}")
])

chain = prompt | llm | StrOutputParser()


# ── Step 1: Load dataset (metadata string + raw DataFrame)
metadata, df = load_dataset()

# ── Step 2: Preview the dataset
print("\n" + "=" * 50)
print("STEP 1 — Dataset Preview")
print("=" * 50)
show_head(df)

# ── Step 3: Data Engineering — clean nulls + detailed EDA
print("\n" + "=" * 50)
print("STEP 2 — Data Engineering & EDA")
print("=" * 50)
df = data_engineering(df)

# ── Step 4: Correlation Analysis
print("\n" + "=" * 50)
print("STEP 3 — Correlation Analysis")
print("=" * 50)
correlation_analysis(df)

# ── Step 5: Outlier Detection
print("\n" + "=" * 50)
print("STEP 4 — Outlier Detection")
print("=" * 50)
detect_outliers(df)

# ── Step 6: Ask LLM for plot recommendations
print("\n" + "=" * 50)
print("STEP 5 — Asking GPT-4o-mini for Plot Advice...")
print("=" * 50)
response = chain.invoke({"dataset_info": metadata})
print(response)

# ── Step 7: Visualize (manual — use after reading LLM suggestions above)
# Example: visualize_data(df, "scatter plot", ["total_bill", "tip"], hue="sex")
# Uncomment and adjust columns/plot type based on LLM output above