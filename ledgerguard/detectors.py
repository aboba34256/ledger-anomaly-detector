"""Rule-based anomaly detectors.

Every detector takes a normalised ledger plus a Config and returns a DataFrame
with columns ``entry_id, rule, detail``. ``detail`` is a plain-English reason
that is later turned into an audit-style note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

EXPECTED_BENFORD = np.log10(1 + 1 / np.arange(1, 10))

RULE_LABELS = {
    "duplicate": "Possible duplicate payment",
    "round_number": "Round-dollar amount",
    "off_hours": "Weekend / off-hours posting",
    "threshold": "Just below approval limit",
    "outlier": "Unusually large for account",
    "self_approved": "Posted and approved by same user",
    "missing_approval": "No approver above limit",
    "benford_vendor": "Vendor fails Benford's Law",
}

RULE_WEIGHTS = {
    "duplicate": 0.60,
    "self_approved": 0.55,
    "outlier": 0.50,
    "missing_approval": 0.50,
    "threshold": 0.40,
    "round_number": 0.35,
    "off_hours": 0.35,
    "benford_vendor": 0.30,
}

_EMPTY = ["entry_id", "rule", "detail"]


def _flags(ids, rule: str, details) -> pd.DataFrame:
    return pd.DataFrame({"entry_id": list(ids), "rule": rule, "detail": list(details)}, columns=_EMPTY)


# --------------------------------------------------------------------------- #
# Benford's Law helpers
# --------------------------------------------------------------------------- #
def first_digits(amounts) -> np.ndarray:
    a = np.abs(np.asarray(amounts, dtype=float))
    a = a[a > 0]
    digits = np.floor(a / 10 ** np.floor(np.log10(a))).astype(int)
    return np.clip(digits, 1, 9)


def benford_table(amounts, cfg: Config | None = None):
    """Observed vs expected first-digit distribution plus conformity statistics."""
    cfg = cfg or Config()
    digits = first_digits(amounts)
    n = len(digits)
    counts = np.array([(digits == d).sum() for d in range(1, 10)])
    observed = counts / n if n else np.zeros(9)
    table = pd.DataFrame(
        {"digit": range(1, 10), "expected": EXPECTED_BENFORD, "observed": observed, "count": counts}
    )
    chi2 = float(((counts - n * EXPECTED_BENFORD) ** 2 / (n * EXPECTED_BENFORD)).sum()) if n else 0.0
    mad = float(np.abs(observed - EXPECTED_BENFORD).mean()) if n else 0.0
    # Nigrini's mean-absolute-deviation conformity bands for first digits
    if mad < 0.006:
        label = "Close conformity"
    elif mad < 0.012:
        label = "Acceptable conformity"
    elif mad < 0.015:
        label = "Marginal conformity"
    else:
        label = "Nonconformity"
    return table, {"n": n, "chi2": chi2, "mad": mad, "conformity": label, "fails": chi2 > cfg.benford_chi2}


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def detect_duplicates(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.assign(amt=df["amount"].abs().round(2))
    key = "vendor" if d["vendor"].notna().any() else "description"
    d = d[d[key].notna() & (d["amt"] > 0)].sort_values([key, "amt", "timestamp"])
    grp = d.groupby([key, "amt"])
    d["prev_ts"] = grp["timestamp"].shift()
    d["prev_id"] = grp["entry_id"].shift()
    gap = (d["timestamp"] - d["prev_ts"]).dt.total_seconds() / 86_400
    keep = gap.le(cfg.duplicate_window_days) & d["prev_ts"].notna()
    hit, gap = d[keep], gap[keep]
    details = [
        f"Same {key} and amount (${r.amt:,.2f}) as {r.prev_id}, posted {g:.0f} day(s) earlier"
        for r, g in zip(hit.itertuples(), gap)
    ]
    return _flags(hit["entry_id"], "duplicate", details)


def detect_round_numbers(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    amt = df["amount"].abs()
    cents = (amt * 100).round().astype("int64")
    hit = df[(amt >= cfg.round_min) & (cents % 100_000 == 0)]
    return _flags(hit["entry_id"], "round_number",
                  [f"Exact round-dollar amount ${a:,.0f}" for a in hit["amount"].abs()])


def detect_off_hours(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ts = df["timestamp"]
    has_time = bool((ts.dt.normalize() != ts).any())
    weekend = ts.dt.dayofweek >= 5
    if has_time:
        night = (ts.dt.hour < cfg.business_start) | (ts.dt.hour >= cfg.business_end)
    else:
        night = pd.Series(False, index=df.index)
    hit = df[weekend | night]
    fmt = "%A %H:%M" if has_time else "%A"
    return _flags(hit["entry_id"], "off_hours",
                  [f"Posted on {t.strftime(fmt)}, outside normal business hours" for t in hit["timestamp"]])


def detect_threshold(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    amt = df["amount"].abs()
    lo = cfg.approval_limit * (1 - cfg.threshold_band)
    hit = df[(amt >= lo) & (amt < cfg.approval_limit)]
    return _flags(hit["entry_id"], "threshold",
                  [f"${a:,.2f} sits just under the ${cfg.approval_limit:,.0f} approval limit"
                   for a in hit["amount"].abs()])


def detect_outliers(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.assign(amt=df["amount"].abs(), grp=df["account"].fillna("(no account)"))
    d = d[d["amt"] > 0]
    logs = np.log(d["amt"])
    g = logs.groupby(d["grp"])
    med = g.transform("median")
    mad = (logs - med).abs().groupby(d["grp"]).transform("median") * 1.4826
    z = (logs - med) / mad.where(mad > 0)
    hit = d[(z > cfg.outlier_z) & (g.transform("size") >= cfg.outlier_min_n)]
    details = [
        f"Amount ${r.amt:,.0f} is {z[i]:.1f} robust SDs above the typical {r.grp} posting "
        f"(median ${np.exp(med[i]):,.0f})"
        for i, r in zip(hit.index, hit.itertuples())
    ]
    return _flags(hit["entry_id"], "outlier", details)


def detect_control_breaks(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = []
    both = df["posted_by"].notna() & df["approved_by"].notna()
    same = df[both & (df["posted_by"] == df["approved_by"])]
    out.append(_flags(same["entry_id"], "self_approved",
                      [f"Posted and approved by the same user ({u}), a segregation-of-duties conflict"
                       for u in same["posted_by"]]))
    if df["approved_by"].notna().any():
        miss = df[df["approved_by"].isna() & (df["amount"].abs() >= cfg.approval_limit)]
        out.append(_flags(miss["entry_id"], "missing_approval",
                          [f"${a:,.0f} entry at or above the approval limit has no approver recorded"
                           for a in miss["amount"].abs()]))
    return pd.concat(out, ignore_index=True)


def detect_benford_vendors(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not df["vendor"].notna().any():
        return _flags([], "benford_vendor", [])
    out = []
    for vendor, g in df.dropna(subset=["vendor"]).groupby("vendor"):
        g = g[g["amount"].abs() > 0]
        if len(g) < cfg.benford_min_n:
            continue
        table, stats = benford_table(g["amount"], cfg)
        if not stats["fails"]:
            continue
        tb = table.set_index("digit")
        bad = tb.index[(tb["observed"] - tb["expected"]) >= cfg.benford_excess]
        for eid, dig in zip(g["entry_id"], first_digits(g["amount"])):
            if dig in bad:
                obs, exp = tb.at[dig, "observed"], tb.at[dig, "expected"]
                out.append((eid, f"Vendor '{vendor}' fails Benford's Law (chi2={stats['chi2']:.0f}, "
                                 f"n={stats['n']}); leading digit {dig} appears {obs:.0%} of the time "
                                 f"vs {exp:.0%} expected"))
    return _flags([o[0] for o in out], "benford_vendor", [o[1] for o in out])


DETECTORS = [
    detect_duplicates, detect_round_numbers, detect_off_hours, detect_threshold,
    detect_outliers, detect_control_breaks, detect_benford_vendors,
]


def run_all_detectors(ledger: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or Config()
    parts = [fn(ledger, cfg) for fn in DETECTORS]
    return pd.concat(parts, ignore_index=True).drop_duplicates(["entry_id", "rule"])
