import numpy as np
import pandas as pd

from ledgerguard import Config, analyze, benford_table, generate_ledger, normalize_ledger, template_note
from ledgerguard.detectors import (detect_control_breaks, detect_duplicates, detect_off_hours,
                                   detect_outliers, detect_round_numbers, detect_threshold)
from ledgerguard.evaluation import evaluate

CFG = Config()


def make(rows):
    base = {"entry_id": None, "timestamp": "2025-03-04 10:00:00", "account": "Office Supplies",
            "vendor": "Acme", "description": "x", "amount": 123.45,
            "posted_by": "alice", "approved_by": "bob"}
    df = pd.DataFrame([{**base, **r, "entry_id": f"E{i}"} for i, r in enumerate(rows)])
    return normalize_ledger(df)


def test_duplicate_flags_later_entry_only():
    d = make([{"amount": 500.0}, {"amount": 500.0, "timestamp": "2025-03-06 10:00:00"}, {"amount": 501.0}])
    f = detect_duplicates(d, CFG)
    assert list(f.entry_id) == ["E1"]
    assert "E0" in f.detail.iloc[0]


def test_duplicate_outside_window_not_flagged():
    d = make([{"amount": 500.0}, {"amount": 500.0, "timestamp": "2025-04-20 10:00:00"}])
    assert detect_duplicates(d, CFG).empty


def test_round_numbers():
    d = make([{"amount": 5000.0}, {"amount": 5000.5}, {"amount": 200.0}])
    assert list(detect_round_numbers(d, CFG).entry_id) == ["E0"]


def test_off_hours_weekend_and_night():
    d = make([{"timestamp": "2025-03-08 11:00:00"},   # Saturday
              {"timestamp": "2025-03-04 02:30:00"},   # Tuesday 2:30am
              {"timestamp": "2025-03-04 11:00:00"}])  # normal
    assert list(detect_off_hours(d, CFG).entry_id) == ["E0", "E1"]


def test_date_only_data_does_not_flag_midnight():
    d = make([{"timestamp": "2025-03-04"}, {"timestamp": "2025-03-05"}])
    assert detect_off_hours(d, CFG).empty


def test_threshold_band():
    d = make([{"amount": 9800.0}, {"amount": 10000.0}, {"amount": 9400.0}])
    assert list(detect_threshold(d, CFG).entry_id) == ["E0"]


def test_outlier_needs_enough_history():
    rows = [{"amount": 100.0 + i} for i in range(30)] + [{"amount": 900_000.0}]
    assert list(detect_outliers(make(rows), CFG).entry_id) == ["E30"]
    assert detect_outliers(make(rows[-5:]), CFG).empty


def test_control_breaks():
    d = make([{"posted_by": "alice", "approved_by": "alice"},
              {"amount": 50_000.0, "approved_by": None},
              {"amount": 50_000.0, "approved_by": "bob"}])
    f = detect_control_breaks(d, CFG)
    assert set(zip(f.entry_id, f.rule)) == {("E0", "self_approved"), ("E1", "missing_approval")}


def test_benford_conforming_vs_fabricated():
    rng = np.random.default_rng(0)
    conforming = np.exp(rng.uniform(np.log(10), np.log(1e6), 5000))  # log-uniform obeys Benford
    _, ok = benford_table(conforming)
    _, bad = benford_table(rng.uniform(7000, 8999, 500))
    assert not ok["fails"] and ok["conformity"] in {"Close conformity", "Acceptable conformity"}
    assert bad["fails"] and bad["conformity"] == "Nonconformity"


def test_normalize_handles_messy_input():
    raw = pd.DataFrame({"Date": ["2025-01-02", "bad", "2025-01-03"], "Amount": ["$1,234.50", "5", "(oops)"]})
    d = normalize_ledger(raw)
    assert len(d) == 1 and d.amount.iloc[0] == 1234.50 and d.entry_id.iloc[0].startswith("ROW-")


def test_duplicate_entry_ids_made_unique():
    raw = pd.DataFrame({"entry_id": ["A", "A"], "date": ["2025-01-02"] * 2, "amount": [1, 2]})
    assert normalize_ledger(raw).entry_id.is_unique


def test_scoring_combines_rules_and_tiers():
    d = make([{"amount": 5000.0, "approved_by": "alice", "posted_by": "alice",
               "timestamp": "2025-03-08 03:00:00"}] +
             [{"amount": 100.0 + i} for i in range(5)])
    _, _, scored = analyze(d, CFG)
    top = scored.iloc[0]
    assert top.n_rules >= 3 and top.risk_tier == "High" and 0 < top.risk_score < 1


def test_clean_ledger_produces_no_flags():
    d = make([{"amount": 100.0 + i * 7.31, "timestamp": f"2025-03-{4 + i % 3:02d} 10:00:00"} for i in range(10)])
    _, flags, scored = analyze(d, CFG)
    assert flags.empty and scored.empty


def test_template_note_is_cautious_and_complete():
    d = make([{"amount": 9900.0}] + [{"amount": 100.0 + i} for i in range(3)])
    _, _, scored = analyze(d, CFG)
    note = template_note(scored.iloc[0])
    assert "E0" in note and "not a conclusion of error or fraud" in note and "approval" in note


def test_generator_is_deterministic_and_truth_aligned():
    a, ta = generate_ledger(2000, seed=7)
    b, tb = generate_ledger(2000, seed=7)
    assert a.equals(b) and ta.equals(tb)
    assert len(a) == 2000 and a.entry_id.is_unique and set(ta.entry_id) == set(a.entry_id)
    assert (ta.injected_type != "clean").sum() > 0


def test_end_to_end_recall_and_false_positives():
    ledger, truth = generate_ledger(5000, seed=1)
    _, flags, scored = analyze(ledger, CFG)
    _, s = evaluate(truth, flags, scored)
    assert s["any_flag"]["recall"] > 0.9
    assert s["any_flag"]["false_positive_rate"] < 0.02


def test_generator_handles_small_ledgers_and_zero_fraud():
    small, t1 = generate_ledger(1000, seed=3)
    clean, t2 = generate_ledger(500, seed=3, fraud_rate=0.0)
    assert len(small) == 1000 and len(clean) == 500 and (t2.injected_type == "clean").all()
