"""Combine rule hits into a per-entry risk score and tier."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .detectors import RULE_WEIGHTS, run_all_detectors
from .ledger import normalize_ledger


def score_entries(ledger: pd.DataFrame, flags: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per flagged entry. Score = 1 - prod(1 - weight), so extra independent
    red flags raise risk with diminishing returns and the score stays in [0, 1)."""
    cols = ["rules", "n_rules", "reasons", "risk_score", "risk_tier"]
    if flags.empty:
        return pd.DataFrame(columns=list(ledger.columns) + cols)
    f = flags.drop_duplicates(["entry_id", "rule"]).assign(weight=lambda x: x["rule"].map(RULE_WEIGHTS))
    agg = f.groupby("entry_id").agg(
        rules=("rule", lambda s: ", ".join(sorted(s))),
        n_rules=("rule", "nunique"),
        reasons=("detail", " | ".join),
        risk_score=("weight", lambda w: float(1 - np.prod(1 - w.to_numpy()))),
    ).reset_index()
    agg["risk_tier"] = np.select(
        [agg["risk_score"] >= cfg.high_threshold, agg["risk_score"] >= cfg.medium_threshold],
        ["High", "Medium"], default="Low")
    out = ledger.merge(agg, on="entry_id", how="inner")
    out["_abs"] = out["amount"].abs()
    return out.sort_values(["risk_score", "_abs"], ascending=False).drop(columns="_abs").reset_index(drop=True)


def analyze(df: pd.DataFrame, cfg: Config | None = None):
    """Full pipeline: returns (clean ledger, raw rule flags, scored flagged entries)."""
    cfg = cfg or Config()
    ledger = normalize_ledger(df)
    flags = run_all_detectors(ledger, cfg)
    return ledger, flags, score_entries(ledger, flags, cfg)
