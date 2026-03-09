import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path=".evv")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=384)


prompt = ChatPromptTemplate.from_messages([
    ('system', "You are an expert Data Scientist and EDA Specialist. I will give you dataset metadata. Your job is to reply with the best Seaborn plot type (e.g., 'scatterplot', 'boxplot', 'barplot') to find insights, and give a 1-sentence reason why."),
    ('human', "{dataset_info}")
])

chain = prompt | llm | StrOutputParser()


dummy_metadata = "Columns: ['Total_Bill' (float), 'Tip' (float), 'Day_of_Week' (category)]"

print("Asking GPT-4o-mini for graph advice...")
print("-" * 20)

response = chain.invoke({"dataset_info": dummy_metadata})

print(response)