#loading dataset
from __future__ import annotations
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_table(source, name: str | None = None) -> pd.DataFrame:
    """Load csv/tsv/xlsx/json/parquet from a path or an uploaded file-like object."""
    if isinstance(source, (str, Path)):
        name = name or str(source)
        data = Path(source).read_bytes()
    else:  # e.g. Streamlit UploadedFile
        name = name or getattr(source, "name", "upload.csv")
        data = source.read()
    ext = Path(name).suffix.lower()

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(data))
    elif ext == ".json":
        try:
            df = pd.read_json(io.BytesIO(data))
        except ValueError:
            df = pd.DataFrame(json.loads(data))
    elif ext == ".parquet":
        df = pd.read_parquet(io.BytesIO(data))
    else:  # csv / tsv / txt: sniff the delimiter, tolerate odd encodings
        last_err = None
        df = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=enc)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if df is None:
            raise ValueError(f"Could not parse {name}: {last_err}")

    df.columns = [str(c).strip() for c in df.columns]
    return coerce_types(df)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse columns that look like dates; leave everything else alone."""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if s.dtype == object or str(s.dtype).startswith("str"):
            sample = s.dropna().astype(str).head(50)
            if len(sample) == 0:
                continue
            # only try dates when values look date-like (contain - / : and a digit)
            looks = sample.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{4}-\d{2}", regex=True).mean()
            if looks > 0.8:
                parsed = pd.to_datetime(s, errors="coerce")
                if parsed.notna().mean() > 0.9:
                    df[col] = parsed
    return df


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        nun = s.nunique(dropna=True)
        if nun == len(s.dropna()) and nun > 20 and pd.api.types.is_integer_dtype(s):
            return "numeric (id-like)"
        return "numeric"
    nun = s.nunique(dropna=True)
    if nun > 0.5 * max(len(s.dropna()), 1) and nun > 30:
        return "text (high-cardinality)"
    return "categorical"


def clean(obj):
    """Make anything JSON-serialisable (numpy scalars, NaN, Timestamps...)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, pd.Period)):
        return str(obj)
    if obj is pd.NaT:
        return None
    return obj


def table(df: pd.DataFrame, max_rows: int = 30) -> dict:
    shown = df.head(max_rows)
    return {
        "columns": [str(c) for c in shown.columns],
        "rows": clean(shown.astype(object).where(shown.notna(), None).values.tolist()),
        "n_rows": int(len(df)),
        "truncated": bool(len(df) > max_rows),
    }
