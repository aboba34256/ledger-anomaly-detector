"""Load and standardise a general-ledger / journal-entry export."""
from __future__ import annotations

import numpy as np
import pandas as pd

OPTIONAL_TEXT_COLUMNS = ["account", "vendor", "description", "posted_by", "approved_by"]


def normalize_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy with standard column names.

    Required: ``amount`` and either ``timestamp`` or ``date``.
    Optional: entry_id, account, vendor, description, posted_by, approved_by.
    """
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]

    if "amount" not in d.columns:
        raise ValueError("Ledger needs an 'amount' column.")
    if "timestamp" in d.columns:
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    elif "date" in d.columns:
        d["timestamp"] = pd.to_datetime(d["date"], errors="coerce")
    else:
        raise ValueError("Ledger needs a 'date' or 'timestamp' column.")

    cleaned = d["amount"].astype(str).str.replace(r"[$,\s]", "", regex=True)
    d["amount"] = pd.to_numeric(cleaned, errors="coerce")
    d = d.dropna(subset=["timestamp", "amount"]).reset_index(drop=True)

    if "entry_id" not in d.columns:
        d["entry_id"] = [f"ROW-{i + 1:06d}" for i in range(len(d))]
    d["entry_id"] = d["entry_id"].astype(str)
    if d["entry_id"].duplicated().any():  # keep IDs unique so results join cleanly
        n = d.groupby("entry_id").cumcount() + 1
        d["entry_id"] = d["entry_id"] + "#" + n.astype(str)

    for col in OPTIONAL_TEXT_COLUMNS:
        if col not in d.columns:
            d[col] = np.nan
    d["date"] = d["timestamp"].dt.strftime("%Y-%m-%d")
    return d
