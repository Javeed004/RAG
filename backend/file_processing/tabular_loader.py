import pandas as pd
from langchain_core.documents import Document


def _find_header_row(file_path, reader, max_scan=15, **read_kwargs):
    """
    Real spreadsheets often have title rows / blank rows above the
    actual header (like this one). Scan the first few rows and treat
    the first row that has the same number of filled cells as the
    row right after it as the header — a title row is usually sparser
    than the header+data rows that follow it.
    """
    raw = reader(file_path, header=None, nrows=max_scan, **read_kwargs)

    for i in range(len(raw) - 1):
        filled = raw.iloc[i].notna().sum()
        next_filled = raw.iloc[i + 1].notna().sum()
        if filled >= 2 and filled == next_filled:
            return i
    return 0


def _dataframe_to_documents(df, file_path):
    documents = []
    for row_index, row in df.iterrows():
        row_text = "\n".join(
            f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])
        )
        if not row_text.strip():
            continue
        documents.append(
            Document(
                page_content=row_text,
                metadata={"source": file_path.name, "page": row_index},
            )
        )
    return documents


def load_csv(file_path):
    header_row = _find_header_row(file_path, pd.read_csv)
    df = pd.read_csv(file_path, header=header_row)
    df = df.dropna(axis=1, how="all")
    return _dataframe_to_documents(df, file_path)


def load_excel(file_path):
    header_row = _find_header_row(file_path, pd.read_excel)
    df = pd.read_excel(file_path, header=header_row)
    df = df.dropna(axis=1, how="all")
    return _dataframe_to_documents(df, file_path)