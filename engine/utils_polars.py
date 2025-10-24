import io
import re
import os
import polars as pl

SUPPORTED_EXTS = {".csv", ".xlsx", ".xls", ".parquet"}

# ---------------------------------------------------------------------
# Read file into Polars DataFrame
# ---------------------------------------------------------------------
def _read_csv_bytes(file_bytes: bytes) -> pl.DataFrame:
    """Read CSV content efficiently using Polars."""
    return pl.read_csv(io.BytesIO(file_bytes), infer_schema_length=10000)


def _read_excel_bytes(file_bytes: bytes) -> pl.DataFrame:
    """Read Excel file into Polars (using pandas as fallback)."""
    import pandas as pd

    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet = xls.sheet_names[0]
    pdf = pd.read_excel(xls, sheet_name=sheet)
    return pl.from_pandas(pdf)


def read_any_to_polars(file_bytes: bytes, filename: str) -> pl.DataFrame:
    """Universal reader for CSV/XLSX/XLS/Parquet."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        return _read_csv_bytes(file_bytes)
    if ext in {".xlsx", ".xls"}:
        return _read_excel_bytes(file_bytes)
    if ext == ".parquet":
        return pl.read_parquet(io.BytesIO(file_bytes))
    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------
# Write output as Parquet
# ---------------------------------------------------------------------
def write_parquet(df: pl.DataFrame, path: str) -> str:
    """Save Polars DataFrame as Parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.write_parquet(path)
    return path


# ---------------------------------------------------------------------
# Ensure expected schema
# ---------------------------------------------------------------------
def ensure_columns(
    df: pl.DataFrame, call_id_col: str, ts_col: str, speaker_col: str, text_col: str
) -> pl.DataFrame:
    """Standardize column names for transcripts."""
    out = df.rename(
        {
            call_id_col: "call_id",
            ts_col: "timestamp",
            speaker_col: "speaker",
            text_col: "text",
        }
    )
    return out.with_columns(
        [
            pl.col("call_id").cast(pl.Utf8),
            pl.col("speaker").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8),
        ]
    )


# ---------------------------------------------------------------------
# Parse raw transcript column (if entire conversation is one string)
# ---------------------------------------------------------------------
TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\]\s*(?P<speaker>[A-Za-z ]+):\s*(?P<text>.*)$"
)


def explode_raw_transcript_column(
    df: pl.DataFrame, raw_col: str, call_id_col: str = None
) -> pl.DataFrame:
    """
    Split single-column transcripts into multiple utterance rows.
    Expected pattern: [HH:MM:SS SPEAKER]: text
    """
    rows = []
    for rec in df.to_dicts():
        call_id = rec.get(call_id_col, "CALL_1")
        for ln in str(rec.get(raw_col, "")).splitlines():
            m = TRANSCRIPT_LINE_RE.match(ln)
            if m:
                h, mnt, s = int(m.group("h")), int(m.group("m")), int(m.group("s"))
                rows.append(
                    {
                        "call_id": call_id,
                        "timestamp": h * 3600 + mnt * 60 + s,
                        "speaker": m.group("speaker").strip().upper(),
                        "text": m.group("text").strip(),
                    }
                )
    return pl.from_dicts(rows)
