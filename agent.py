import seaborn as sns
import pandas as pd
import io

def load_dataset():
    df = sns.load_dataset('tips')

    # Column dtypes
    metadata_dict = df.dtypes.astype(str).to_dict()
    dataset_info = f"Dataset Columns and Types: {metadata_dict}"

    # df.info() — must be captured since it prints to stdout
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()

    # df.head()
    df_head = df.head().to_string()

    full_metadata = (
        f"{dataset_info}\n\n"
        f"--- df.info() ---\n{df_info}\n"
        f"--- df.head() ---\n{df_head}"
    )

    # Return both metadata string AND the DataFrame
    return full_metadata, df

print("Here is what we will send to the LLM:")
print("-" * 20)
metadata, _ = load_dataset()
print(metadata)
