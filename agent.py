import seaborn as sns
import pandas as pd
import io


def load_dataset():
    """Load the built-in seaborn 'tips' dataset (CLI demo fallback)."""
    df = sns.load_dataset('tips')
    metadata = _build_metadata(df)
    return metadata, df


def load_uploaded_dataset(uploaded_file):
    """
    Load a user-uploaded CSV or XLSX file.

    Args:
        uploaded_file : A file-like object (e.g. Streamlit UploadedFile)

    Returns:
        (metadata_string, DataFrame)
    """
    filename = getattr(uploaded_file, "name", "uploaded")

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    metadata = _build_metadata(df)
    return metadata, df


def _build_metadata(df: pd.DataFrame) -> str:
    """Build a metadata string from a DataFrame for the LLM."""
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

    return full_metadata


# ── CLI test (only runs when executed directly) ──
if __name__ == "__main__":
    print("Here is what we will send to the LLM:")
    print("-" * 20)
    metadata, _ = load_dataset()
    print(metadata)
