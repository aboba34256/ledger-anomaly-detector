"""Command-line entry point: python -m ledgerguard.cli ledger.csv --out flagged.csv"""
import argparse

import pandas as pd

from .config import Config
from .detectors import benford_table
from .scoring import analyze


def main() -> None:
    p = argparse.ArgumentParser(description="Flag suspicious entries in a general-ledger CSV")
    p.add_argument("csv")
    p.add_argument("--out", default="flagged_entries.csv")
    p.add_argument("--approval-limit", type=float, default=10_000)
    a = p.parse_args()
    ledger, _, scored = analyze(pd.read_csv(a.csv), Config(approval_limit=a.approval_limit))
    scored.to_csv(a.out, index=False)
    _, stats = benford_table(ledger["amount"])
    print(f"{len(ledger):,} entries analysed, {len(scored):,} flagged "
          f"({(scored.risk_tier == 'High').sum()} high risk) -> {a.out}")
    print(f"Overall Benford test: {stats['conformity']} (MAD={stats['mad']:.4f}, chi2={stats['chi2']:.1f})")


if __name__ == "__main__":
    main()
