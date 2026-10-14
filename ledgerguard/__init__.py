"""LedgerGuard: explainable anomaly detection for general-ledger data."""
from .config import Config
from .detectors import RULE_LABELS, RULE_WEIGHTS, benford_table, run_all_detectors
from .explain import explain_entry, template_note
from .ledger import normalize_ledger
from .scoring import analyze, score_entries

__version__ = "1.0.0"
__all__ = ["Config", "RULE_LABELS", "RULE_WEIGHTS", "analyze", "benford_table", "explain_entry",
           "generate_ledger", "normalize_ledger", "run_all_detectors", "score_entries", "template_note"]


def __getattr__(name):  # lazy import keeps `python -m ledgerguard.generate` warning-free
    if name == "generate_ledger":
        from .generate import generate_ledger
        return generate_ledger
    raise AttributeError(name)
