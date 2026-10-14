"""Measure detector performance against the seeded ground truth."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .generate import generate_ledger
from .scoring import analyze

INTENDED_RULE = {k: k for k in ["duplicate", "round_number", "off_hours", "threshold",
                                "outlier", "self_approved", "benford_vendor"]}


def evaluate(truth: pd.DataFrame, flags: pd.DataFrame, scored: pd.DataFrame):
    flagged = set(scored["entry_id"])
    medplus = set(scored.loc[scored["risk_tier"].isin(["High", "Medium"]), "entry_id"])
    clean = set(truth.loc[truth["injected_type"] == "clean", "entry_id"])
    injected = set(truth["entry_id"]) - clean

    rows = []
    for kind, g in truth[truth["injected_type"] != "clean"].groupby("injected_type"):
        ids = set(g["entry_id"])
        hit_rule = set(flags.loc[flags["rule"] == INTENDED_RULE[kind], "entry_id"])
        rows.append({"irregularity": kind, "seeded": len(ids),
                     "caught_by_intended_rule": len(ids & hit_rule),
                     "caught_by_any_rule": len(ids & flagged)})
    per_type = pd.DataFrame(rows)
    per_type["recall"] = per_type["caught_by_any_rule"] / per_type["seeded"]

    def pr(sel):
        return {"recall": len(injected & sel) / len(injected),
                "false_positive_rate": len(clean & sel) / len(clean),
                "precision": len(injected & sel) / len(sel) if sel else float("nan"),
                "flagged": len(sel)}

    summary = {"entries": len(truth), "seeded": len(injected),
               "any_flag": pr(flagged), "medium_plus": pr(medplus)}
    return per_type, summary


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate detector on seeded synthetic ledgers")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--fraud-rate", type=float, default=0.03)
    p.add_argument("--out", default="docs/evaluation_results.md")
    a = p.parse_args()

    per_runs, sums = [], []
    for seed in range(a.seeds):
        ledger, truth = generate_ledger(a.n, seed, a.fraud_rate)
        _, flags, scored = analyze(ledger, Config())
        per, s = evaluate(truth, flags, scored)
        per_runs.append(per.set_index("irregularity"))
        sums.append(s)

    per = sum(per_runs) / len(per_runs)
    avg = lambda tier, k: float(np.mean([s[tier][k] for s in sums]))  # noqa: E731

    lines = ["# Evaluation results", "",
             f"Averaged over {a.seeds} random synthetic ledgers of {a.n:,} entries "
             f"({sums[0]['seeded']} seeded irregularities each, {a.fraud_rate:.0%} rate).", "",
             "## Recall by irregularity type", "",
             "| Irregularity | Seeded (avg) | Caught by intended rule | Caught by any rule | Recall |",
             "|---|---|---|---|---|"]
    for kind, r in per.iterrows():
        lines.append(f"| {kind} | {r.seeded:.0f} | {r.caught_by_intended_rule:.0f} | "
                     f"{r.caught_by_any_rule:.0f} | {r.recall:.0%} |")
    lines += ["", "## Overall", "",
              "| Alert level | Recall | Precision | False-positive rate | Entries flagged (avg) |",
              "|---|---|---|---|---|"]
    for name, key in [("Any flag", "any_flag"), ("Medium risk or higher", "medium_plus")]:
        lines.append(f"| {name} | {avg(key, 'recall'):.1%} | {avg(key, 'precision'):.1%} | "
                     f"{avg(key, 'false_positive_rate'):.2%} | {avg(key, 'flagged'):.0f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


if __name__ == "__main__":
    main()
