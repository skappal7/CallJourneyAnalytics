from __future__ import annotations
import os
import io
import json
import time
import altair as alt
import polars as pl
import streamlit as st
from engine.utils_polars import read_any_to_polars, write_parquet, ensure_columns, explode_raw_transcript_column
from engine.rule_engine_duckdb import load_rules, categorize_utterances, speaker_term_buckets
from engine.ui_components import inject_branding, section

st.set_page_config(page_title="Customer Journey Analyzer", page_icon="🧭", layout="wide")
inject_branding()

st.markdown("""
# 🧭 Customer Interaction Journey Analyzer
Build explainable, rule-driven insights from call/chat transcripts — fast and at scale (Polars + DuckDB).
""")

# Sidebar: upload + config
with st.sidebar:
    st.header("Upload Transcripts")
    up = st.file_uploader(
        "Upload CSV / XLSX / XLS (utterance-level or raw transcript)",
        type=["csv", "xlsx", "xls", "parquet"],
        accept_multiple_files=False,
    )
    st.caption(
        "Accepted schemas:\n\n"
        "• Utterance-level: call_id, timestamp (HH:MM:SS or seconds), speaker, text\n"
        "• Raw transcript single column: parse lines like [HH:MM:SS AGENT]: message"
    )

    st.divider()
    st.header("Column Mapping")
    call_id_col = st.text_input("Call ID column (if present)", value="call_id")
    ts_col = st.text_input("Timestamp column", value="timestamp")
    speaker_col = st.text_input("Speaker column", value="speaker")
    text_col = st.text_input("Text column or Raw Transcript column", value="text")

    st.divider()
    st.header("Processing")
    parse_raw = st.checkbox(
        "Parse raw transcript column (split [HH:MM:SS SPEAKER]: ...)", value=False
    )

@st.cache_resource(show_spinner=False)
def _load_rules_cached():
    return load_rules()

try:
    rules_df = _load_rules_cached()
    with st.expander("Rules loaded (overview)"):
        st.write(
            rules_df.select(
                ["rule_id", "query_name", "industry", "category", "subcategory"]
            )
            .head(20)
            .to_pandas()
        )
except Exception as e:
    st.error(f"Failed to load embedded rules: {e}")
    st.stop()

def to_seconds(x) -> int:
    if x is None:
        return 0
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip()
    if s.isdigit():
        return int(s)
    try:
        hh, mm, ss = s.split(":")
        return int(hh) * 3600 + int(mm) * 60 + int(ss)
    except Exception:
        return 0

if up is not None:
    t0 = time.time()
    raw_df = read_any_to_polars(up.getvalue(), up.name)
    st.success(f"Loaded file: {up.name} (rows: {raw_df.height:,}, cols: {len(raw_df.columns)})")

    if parse_raw:
        utter = explode_raw_transcript_column(raw_df, raw_col=text_col, call_id_col=call_id_col or None)
    else:
        utter = ensure_columns(raw_df, call_id_col, ts_col, speaker_col, text_col)
        utter = utter.with_columns([
            pl.col("timestamp").map_elements(to_seconds).cast(pl.Int64),
            pl.col("speaker").str.to_uppercase(),
        ])

    st.info(f"Utterance rows prepared: {utter.height:,}")

    with st.spinner("Running DuckDB rule engine..."):
        matches, journey = categorize_utterances(utter, rules_df)

    left, right = st.columns([3, 2])
    with left:
        section("Per-utterance Matches")
        st.dataframe(matches.head(500).to_pandas(), use_container_width=True, height=420)
        section("Journey Summary (per call)")
        st.dataframe(journey.to_pandas(), use_container_width=True, height=300)

    with right:
        section("Category Breakdown")
        if not matches.is_empty():
            chart_df = (
                matches.group_by(["category"])
                .len()
                .rename({"len": "hits"})
                .sort("hits", descending=True)
                .head(30)
            ).to_pandas()
            bar = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("hits:Q", title="Rule Hits"),
                    y=alt.Y("category:N", sort="-x", title="Category"),
                    tooltip=["category", "hits"],
                )
                .properties(height=420)
            )
            st.altair_chart(bar, use_container_width=True)

        section("Speaker Term Buckets")
        buckets = speaker_term_buckets(utter)
        if not buckets.is_empty():
            st.dataframe(buckets.to_pandas(), use_container_width=True, height=300)

    section("Downloads")
    matches_bytes = io.BytesIO()
    matches.write_parquet(matches_bytes)
    st.download_button(
        "Download Matches (Parquet)",
        data=matches_bytes.getvalue(),
        file_name="categorized_results.parquet",
    )

    xlsx_bytes = io.BytesIO()
    matches.to_pandas().to_excel(xlsx_bytes, index=False)
    st.download_button(
        "Download Matches (Excel)",
        data=xlsx_bytes.getvalue(),
        file_name="categorized_results.xlsx",
    )

    st.caption(f"Processed in {(time.time()-t0):.2f}s")

else:
    st.info("Upload a transcript file to get started. Embed your rules at rules/rules_embedded.csv or .parquet before running.")
