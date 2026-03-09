import seaborn as sns
import pandas as pd

def load_dataset():
    df = sns.load_dataset('tips') 
    metadata_dict = df.dtypes.astype(str).to_dict()
    dataset_info = f"Dataset Columns and Types: {metadata_dict}"
    return dataset_info

print("Here is what we will send to the LLM:")
print("-" * 20)
print(load_dataset())