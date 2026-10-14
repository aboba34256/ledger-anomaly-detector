# Resume bullets and LinkedIn copy

Only use numbers you have reproduced yourself (`python -m ledgerguard.evaluation`). Always say the data is synthetic.

## Resume bullets (pick 2-3)

- Built **LedgerGuard**, an explainable ledger-auditing tool in Python and Streamlit with 8 detection rules (duplicate payments, round-dollar amounts, Benford's Law, segregation-of-duties conflicts, missing approvals, off-hours postings, approval-threshold avoidance, outliers) and a weighted risk-scoring model.
- Validated against 5 synthetic 10,000-entry ledgers with ~300 seeded irregularities each: flagged 98.9% with a 0.26% false-positive rate; documented limitations and tuning options.
- Generated audit-style workpaper notes for each flagged entry (rule-based, with optional LLM drafting via the Anthropic API) to explain the risk and recommend follow-up procedures.
- Wrote 17 automated tests and CI (GitHub Actions); published open-source with documentation.

**Skills line:** Python, pandas, Streamlit, Benford's Law / forensic analytics, journal entry testing, internal controls (segregation of duties), data validation, Git/GitHub

## LinkedIn "Projects" entry

**Title:** LedgerGuard: Ledger Anomaly Detector
**Description:** Explainable audit analytics tool that scans a general ledger for duplicate payments, Benford's Law violations, segregation-of-duties conflicts, off-hours postings, and amounts sitting just under approval limits, then writes an audit-style note for each flag. Built in Python and Streamlit, validated on synthetic ledgers with seeded irregularities. Open source.
**Link:** your live demo or GitHub URL

## LinkedIn post (edit to sound like you)

I wanted to see how data analytics and AI could actually help with audit work, so I built a tool that scans a general ledger and flags what an auditor should look at first.

It checks for:
- Possible duplicate payments
- Round-dollar amounts and weekend/2 a.m. postings
- Invoices sitting just under the approval limit
- Preparers approving their own entries
- Vendors whose invoice digits fail Benford's Law

Each flag gets a plain-English audit note explaining why it matters and the next procedure to perform.

How I tested it: I planted ~300 irregularities in fake ledgers of 10,000 entries and measured what it caught. About 99% flagged, with a 0.26% false-positive rate. Caveat: synthetic data is cleaner than real life, so I documented the limitations too.

The biggest thing I learned: the hard part isn't flagging entries, it's keeping false positives low enough that someone would actually use it.

Code and demo: [link]

#accounting #audit #dataanalytics #python #forensicaccounting
