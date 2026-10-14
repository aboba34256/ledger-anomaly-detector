"""LedgerGuard - Streamlit dashboard.  Run with:  streamlit run app.py"""
from dataclasses import asdict

import pandas as pd
import streamlit as st

from ledgerguard import (RULE_LABELS, Config, analyze, benford_table, explain_entry,
                         generate_ledger)
from ledgerguard.evaluation import evaluate

st.set_page_config(page_title="LedgerGuard | Ledger Anomaly Detector", page_icon="🔎", layout="wide")


@st.cache_data(show_spinner=False)
def cached_demo(n: int, seed: int, rate: float):
    return generate_ledger(n, seed, rate)


@st.cache_data(show_spinner=False)
def cached_analyze(df: pd.DataFrame, cfg_dict: dict):
    return analyze(df, Config(**cfg_dict))


# ----------------------------------------------------------------- sidebar
st.sidebar.title("🔎 LedgerGuard")
st.sidebar.caption("Explainable anomaly detection for general-ledger data")
source = st.sidebar.radio("Data source", ["Synthetic demo ledger", "Upload CSV"])

truth = None
if source == "Synthetic demo ledger":
    n = st.sidebar.slider("Entries", 1_000, 20_000, 10_000, step=1_000)
    rate = st.sidebar.slider("Seeded irregularity rate", 0.01, 0.10, 0.03, step=0.01, format="%.2f")
    seed = st.sidebar.number_input("Random seed", 0, 9_999, 42)
    raw, truth = cached_demo(n, int(seed), rate)
else:
    up = st.sidebar.file_uploader("General ledger (.csv)", type="csv")
    st.sidebar.caption("Needs `amount` and `date` (or `timestamp`). Optional: entry_id, account, vendor, "
                       "description, posted_by, approved_by. Data is processed in memory only.")
    if up is None:
        st.title("Upload a ledger CSV to begin")
        st.info("Or switch to the synthetic demo ledger in the sidebar to see the tool in action.")
        st.stop()
    raw = pd.read_csv(up)

with st.sidebar.expander("Detection settings"):
    limit = st.number_input("Approval limit ($)", 1_000, 1_000_000, 10_000, step=1_000)
    dup_days = st.slider("Duplicate window (days)", 1, 30, 7)
    z = st.slider("Outlier threshold (robust SDs)", 2.0, 8.0, 4.0, step=0.5)
    start, end = st.slider("Normal posting hours", 0, 24, (7, 19))
    min_tier = st.selectbox("Show tiers", ["Low and above", "Medium and above", "High only"])

with st.sidebar.expander("AI explanations (optional)"):
    use_llm = st.checkbox("Write notes with Claude")
    key = st.text_input("Anthropic API key", type="password", disabled=not use_llm,
                        help="Used only for this session. Without a key, rule-based notes are used.")

cfg = Config(approval_limit=float(limit), duplicate_window_days=dup_days, outlier_z=z,
             business_start=start, business_end=end)
try:
    ledger, flags, scored = cached_analyze(raw, asdict(cfg))
except ValueError as e:
    st.error(str(e))
    st.stop()

tiers = {"Low and above": ["Low", "Medium", "High"], "Medium and above": ["Medium", "High"], "High only": ["High"]}
view = scored[scored["risk_tier"].isin(tiers[min_tier])]

# -------------------------------------------------------------------- header
st.title("Ledger Anomaly Detector")
st.caption("Rule-based audit analytics (duplicates, Benford's Law, control breaks, timing, outliers) "
           "with plain-English audit notes. Flags are indicators for follow-up, not proof of error or fraud.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Entries analysed", f"{len(ledger):,}")
c2.metric("Flagged", f"{len(scored):,}", f"{len(scored) / max(len(ledger), 1):.1%} of ledger", delta_color="off")
c3.metric("High risk", f"{(scored.risk_tier == 'High').sum():,}")
c4.metric("Flagged value", f"${scored['amount'].abs().sum():,.0f}")

tab_flags, tab_benford, tab_rules, tab_eval = st.tabs(
    ["Flagged entries", "Benford's Law", "Rule breakdown", "Accuracy (demo only)"])

# ------------------------------------------------------------- flagged entries
with tab_flags:
    if view.empty:
        st.success("No entries flagged at this level.")
    else:
        show = ["entry_id", "date", "account", "vendor", "amount", "risk_tier", "risk_score", "rules"]
        st.dataframe(view[show], width='stretch', hide_index=True, height=360,
                     column_config={"amount": st.column_config.NumberColumn(format="$%.2f"),
                                    "risk_score": st.column_config.ProgressColumn(
                                        "risk", min_value=0.0, max_value=1.0, format="%.2f")})
        st.download_button("⬇ Download flagged entries (CSV)", view.to_csv(index=False),
                           "flagged_entries.csv", "text/csv")
        st.subheader("Entry detail & audit note")
        pick = st.selectbox("Select an entry", view["entry_id"].tolist())
        row = view.set_index("entry_id", drop=False).loc[pick]
        left, right = st.columns([1, 2])
        with left:
            st.markdown(f"**Risk:** {row['risk_tier']} ({row['risk_score']:.2f})")
            st.markdown(f"**Amount:** ${abs(row['amount']):,.2f}")
            st.markdown(f"**Vendor:** {row['vendor'] if isinstance(row['vendor'], str) else 'n/a'}")
            st.markdown(f"**Account:** {row['account'] if isinstance(row['account'], str) else 'n/a'}")
            st.markdown(f"**Posted / approved by:** {row['posted_by']} / {row['approved_by']}")
        with right:
            for r in row["reasons"].split(" | "):
                st.markdown(f"- {r}")
            note, src = explain_entry(row, use_llm=use_llm and bool(key), api_key=key or None)
            st.info(note)
            if use_llm and key and src == "template":
                st.warning("Claude request failed; showing the rule-based note instead.")
            st.caption(f"Note source: {'Claude' if src == 'claude' else 'rule-based template'}")

# --------------------------------------------------------------------- benford
with tab_benford:
    table, stats = benford_table(ledger["amount"], cfg)
    a, b, c = st.columns(3)
    a.metric("Conformity", stats["conformity"])
    b.metric("MAD", f"{stats['mad']:.4f}", help="Mean absolute deviation from Benford (Nigrini's bands)")
    c.metric("Chi-square (df=8)", f"{stats['chi2']:.1f}", help="Critical value at p=0.001 is 26.1")
    st.bar_chart(table.set_index("digit")[["expected", "observed"]])
    st.markdown(
        "**Why this works:** in naturally occurring financial data, about 30% of amounts start with 1 and "
        "under 5% start with 9. People inventing numbers tend to spread digits too evenly or favour "
        "certain digits. A mismatch is a *reason to look closer*, not proof of fraud. Expect natural "
        "deviations in data with caps, fixed prices or small ranges. The tool also runs this test per vendor.")
    vend = []
    for v, g in ledger.dropna(subset=["vendor"]).groupby("vendor"):
        if (g["amount"].abs() > 0).sum() >= cfg.benford_min_n:
            _, s = benford_table(g["amount"], cfg)
            vend.append({"vendor": v, "entries": s["n"], "chi2": round(s["chi2"], 1),
                         "MAD": round(s["mad"], 4), "result": "FAILS" if s["fails"] else "ok"})
    if vend:
        st.subheader("Per-vendor test")
        st.dataframe(pd.DataFrame(vend).sort_values("chi2", ascending=False), hide_index=True,
                     width='stretch', height=260)

# ------------------------------------------------------------------- rule stats
with tab_rules:
    if flags.empty:
        st.write("No rule hits.")
    else:
        rc = flags["rule"].value_counts().rename_axis("rule").reset_index(name="entries")
        rc["description"] = rc["rule"].map(RULE_LABELS)
        st.bar_chart(rc.set_index("description")["entries"])
        st.dataframe(rc, hide_index=True, width='stretch')

# ------------------------------------------------------------------- accuracy
with tab_eval:
    if truth is None:
        st.info("Accuracy can only be measured on the synthetic demo ledger, where the irregularities "
                "were planted and are therefore known.")
    else:
        per, s = evaluate(truth, flags, scored)
        st.write("Every irregularity in the demo ledger was planted on purpose, so we can measure "
                 "exactly how many the detector catches.")
        x, y, w = st.columns(3)
        x.metric("Recall (any flag)", f"{s['any_flag']['recall']:.1%}")
        y.metric("Precision (any flag)", f"{s['any_flag']['precision']:.1%}")
        w.metric("False-positive rate", f"{s['any_flag']['false_positive_rate']:.2%}")
        st.dataframe(per.assign(recall=per["recall"].map("{:.0%}".format)), hide_index=True,
                     width='stretch')
        st.caption("Synthetic irregularities are cleaner than real-world ones, so real recall will be lower. "
                   "See the README for limitations.")
