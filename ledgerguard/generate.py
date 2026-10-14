"""Synthetic general-ledger generator with *seeded* irregularities.

Because we plant the irregularities ourselves, we know the ground truth and can
measure how many the detector actually catches (see evaluation.py).
Never use real company data in a public portfolio project.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ACCOUNTS = {  # account: (mu, sigma) of log-amount
    "Office Supplies": (5.0, 1.3),
    "Software & Subscriptions": (6.0, 1.3),
    "Professional Fees": (7.4, 1.2),
    "Travel & Meals": (5.8, 1.2),
    "Utilities": (6.2, 1.2),
    "Marketing": (7.0, 1.3),
    "Equipment": (7.5, 1.3),
    "Repairs & Maintenance": (6.3, 1.3),
}
ACCOUNT_LIST = list(ACCOUNTS)
_PREFIX = ["Northwind", "Summit", "Bluebird", "Keystone", "Harbor", "Redwood", "Pioneer", "Atlas",
           "Cobalt", "Evergreen", "Lakeside", "Meridian", "Granite", "Sterling", "Vertex", "Orchard",
           "Beacon", "Falcon", "Juniper", "Crescent"]
VENDORS = [f"{p} {s}" for p in _PREFIX for s in ("Supply Co", "Services LLC")]
POSTERS = ["asmith", "bjones", "cnguyen", "dpatel"]
APPROVERS = ["mgr_lee", "mgr_garcia", "ctrl_kim"]
FRAUD_VENDOR = "Apex Consulting Group LLC"
BUSINESS_DAYS = pd.bdate_range("2025-01-01", "2025-12-31")

_SHARES = {"duplicate": 0.30, "round_number": 0.15, "off_hours": 0.15,
           "threshold": 0.15, "outlier": 0.10, "self_approved": 0.15}


def _clean_rows(n: int, rng: np.random.Generator) -> pd.DataFrame:
    vi = rng.integers(0, len(VENDORS), n)
    accts = np.array([ACCOUNT_LIST[i % len(ACCOUNT_LIST)] for i in vi])
    mu = np.array([ACCOUNTS[a][0] for a in accts])
    sg = np.array([ACCOUNTS[a][1] for a in accts])
    amount = np.maximum(np.round(np.exp(rng.normal(mu, sg)), 2), 1.0)
    days = BUSINESS_DAYS[rng.integers(0, len(BUSINESS_DAYS), n)]
    secs = rng.integers(8 * 3600, 18 * 3600, n)
    inv = rng.integers(10_000, 99_999, n)
    return pd.DataFrame({
        "timestamp": days + pd.to_timedelta(secs, unit="s"),
        "account": accts,
        "vendor": np.array(VENDORS)[vi],
        "description": [f"INV-{i} {a}" for i, a in zip(inv, accts)],
        "amount": amount,
        "posted_by": rng.choice(POSTERS, n),
        "approved_by": rng.choice(APPROVERS, n),
        "injected_type": "clean",
    })


def _mutate(df: pd.DataFrame, rows: np.ndarray, kind: str, rng: np.random.Generator) -> None:
    if kind == "round_number":
        df.loc[rows, "amount"] = rng.choice([5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 50000.0], len(rows))
    elif kind == "threshold":
        df.loc[rows, "amount"] = np.round(rng.uniform(9500, 9999.99, len(rows)), 2)
    elif kind == "self_approved":
        df.loc[rows, "approved_by"] = df.loc[rows, "posted_by"]
    elif kind == "outlier":
        for r in rows:
            mu, sg = ACCOUNTS[df.at[r, "account"]]
            df.at[r, "amount"] = round(float(np.exp(mu + sg * rng.uniform(3.8, 6.0))), 2)
    elif kind == "off_hours":
        for r in rows:
            day = df.at[r, "timestamp"].normalize()
            if rng.random() < 0.5:  # weekend, normal hours
                day += pd.Timedelta(days=int((5 - day.dayofweek) % 7 + rng.integers(0, 2)))
                hour = int(rng.integers(9, 17))
            else:                   # weekday, middle of the night
                hour = int(rng.choice([0, 1, 2, 3, 4, 5, 21, 22, 23]))
            df.at[r, "timestamp"] = day + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
    df.loc[rows, "injected_type"] = kind


def generate_ledger(n: int = 10_000, seed: int = 42, fraud_rate: float = 0.03):
    """Return (ledger, truth). ``truth`` maps entry_id -> injected_type ('clean' if none)."""
    rng = np.random.default_rng(seed)
    n_inj = int(n * fraud_rate)
    n_benford = max(30, int(n_inj * 0.15)) if n_inj else 0
    counts = {k: int((n_inj - n_benford) * v) for k, v in _SHARES.items()} if n_inj else dict.fromkeys(_SHARES, 0)
    n_dup = counts["duplicate"]

    base = _clean_rows(n - n_dup - n_benford, rng)
    n_mut = sum(v for k, v in counts.items() if k != "duplicate")
    picks = rng.choice(len(base), size=n_mut + n_dup, replace=False)
    mut_idx, dup_src = picks[:n_mut], picks[n_mut:]

    pos = 0
    for kind, k in counts.items():
        if kind == "duplicate":
            continue
        _mutate(base, mut_idx[pos:pos + k], kind, rng)
        pos += k

    dups = base.loc[dup_src].copy()
    shifts = rng.integers(1, 6, len(dups))
    dups["timestamp"] = pd.to_datetime(
        pd.Series([t + pd.offsets.BDay(int(s)) for t, s in zip(dups["timestamp"], shifts)], index=dups.index))
    dups["injected_type"] = "duplicate"

    fab = _clean_rows(n_benford, rng)
    lead = rng.choice([7, 8], n_benford)
    mag = rng.choice([100, 1000], n_benford)
    fab["vendor"], fab["account"] = FRAUD_VENDOR, "Professional Fees"
    fab["amount"] = np.round(lead * mag + rng.uniform(0, mag, n_benford), 2)
    fab["injected_type"] = "benford_vendor"

    full = pd.concat([base, dups, fab], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    full.insert(0, "entry_id", [f"JE-{i + 1:06d}" for i in range(len(full))])
    truth = full[["entry_id", "injected_type"]].copy()
    ledger = full.drop(columns="injected_type")
    ledger["timestamp"] = ledger["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return ledger, truth


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic ledger with seeded irregularities")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fraud-rate", type=float, default=0.03)
    p.add_argument("--out", default="data/sample_ledger.csv")
    a = p.parse_args()
    ledger, truth = generate_ledger(a.n, a.seed, a.fraud_rate)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(out, index=False)
    truth.to_csv(out.with_name(out.stem + "_truth.csv"), index=False)
    print(f"Wrote {len(ledger):,} entries to {out} ({(truth.injected_type != 'clean').sum()} seeded irregularities)")


if __name__ == "__main__":
    main()
