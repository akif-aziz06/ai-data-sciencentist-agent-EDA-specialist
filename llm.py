import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent import load_dataset

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


metadata = load_dataset()

print("Asking GPT-4o-mini for graph advice...")
print("-" * 20)

response = chain.invoke({"dataset_info": metadata})

print(response)