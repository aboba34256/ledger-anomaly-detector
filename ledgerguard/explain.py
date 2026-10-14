"""Turn rule hits into audit-workpaper style notes.

Default: deterministic templates (free, offline, reproducible).
Optional: Claude writes the note when an Anthropic API key is available.
"""
from __future__ import annotations

import os

import pandas as pd

FOLLOW_UP = {
    "duplicate": "agree to the vendor statement and remittance records to confirm the invoice was not paid twice",
    "round_number": "inspect supporting documents; round amounts can indicate estimates or fabricated entries",
    "off_hours": "ask the preparer why the entry was posted outside business hours and check system access logs",
    "threshold": "check for split transactions or invoices sized to avoid the approval requirement",
    "outlier": "vouch the entry to source documents and confirm the amount and account coding are correct",
    "self_approved": "confirm whether a compensating review exists and whether system access rights allow this",
    "missing_approval": "locate approval evidence or escalate as a control deficiency",
    "benford_vendor": "review the vendor's invoice population and confirm the vendor is legitimate and invoices are genuine",
}


def template_note(row: pd.Series) -> str:
    rules = [r.strip() for r in str(row["rules"]).split(",")]
    actions = "; ".join(dict.fromkeys(FOLLOW_UP[r] for r in rules if r in FOLLOW_UP))
    reasons = " ".join(r.strip().rstrip(".") + "." for r in str(row["reasons"]).split(" | "))
    vendor, account = row.get("vendor"), row.get("account")
    who = f" to {vendor}" if isinstance(vendor, str) else ""
    acct = account if isinstance(account, str) else "no account"
    return (
        f"Entry {row['entry_id']} dated {row['date']} (${abs(row['amount']):,.2f}{who}, {acct}) "
        f"is rated {row['risk_tier']} risk ({row['risk_score']:.2f}). {reasons} "
        f"Suggested follow-up: {actions}. This is an indicator for review, not a conclusion of error or fraud."
    )


def llm_note(row: pd.Series, api_key: str | None = None, model: str | None = None) -> str:
    """Ask Claude to write the note. Raises on any failure so callers can fall back."""
    import anthropic  # optional dependency

    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    keep = {"entry_id", "date", "account", "vendor", "description", "amount",
            "posted_by", "approved_by", "reasons", "risk_tier"}
    facts = {k: (None if pd.isna(v) else str(v)) for k, v in row.items() if k in keep}
    msg = client.messages.create(
        model=model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        max_tokens=300,
        system=("You are an internal auditor writing a short workpaper note about a flagged journal entry. "
                "Write 2-4 plain sentences: what was flagged, why it matters, and the next audit step. "
                "Use only the facts provided. Never accuse anyone of fraud; say the entry warrants follow-up."),
        messages=[{"role": "user", "content": f"Flagged entry facts: {facts}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def explain_entry(row: pd.Series, use_llm: bool = False, api_key: str | None = None) -> tuple[str, str]:
    """Return (note, source) where source is 'claude' or 'template'."""
    if use_llm:
        try:
            return llm_note(row, api_key=api_key), "claude"
        except Exception:
            pass
    return template_note(row), "template"
