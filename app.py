from __future__ import annotations
import os
import io
import json
import time
import altair as alt
import polars as pl
import streamlit as st
from engine.utils_polars import read_any_to_polars, explode_raw_transcript_column
from engine.rule_engine_duckdb import load_rules, categorize_utterances, speaker_term_buckets
from engine.ui_components import inject_branding, section

# ------------------------------------------------------------
# Streamlit App Config
# ------------------------------------------------------------
st.set_page_config(page_title="Customer Journey Analyzer", page_icon="🧭", layout="wide")
inject_branding()

st.markdown("""
# 🧭 Customer Interaction Journey Analyzer
Upload your call/chat transcripts and instantly extract categories, intents, and conversational journeys.
""")

# ------------------------------------------------------------
# Sidebar — Upload section
# ------------------------------------------------------------
with st.sidebar:
    st.header("Upload Transcript File")
    up = st.file_uploader(
        "Upload your transcript file (CSV, XLSX, or XLS)",
        type=["csv", "xlsx", "xls", "parquet"],
        accept_multiple_files=False
    )
    st.caption("Each row should contain a full transcript like `[01:19:57 AGENT]: message...`")

    st.divider()
    st.header("Column Configuration")
    text_col = st.text_input(
        "Select the column that contains the full transcript text",
        value="transcript"
    )
    call_id_col = st.text_input(
        "Optional: Call ID column (if available in file)",
        value=None
    )

# ------------------------------------------------------------
# Load embedded rules (cached)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_rules_cached():
    return load_rules()

try:
    rules_df = _load_rules_cached()
    with st.expander("Rules loaded successfully (first 20 rows)"):
        st.dataframe(
            rules_df.select(["rule_id", "query_name", "industry", "category", "subcategory"])
            .head(20)
            .to_pandas(),
            use_container_width=True,
        )
except Exception as e:
    st.error(f"Failed to load embedded rules: {e}")
    st.stop()

# ------------------------------------------------------------
# Main App Logic
# ------------------------------------------------------------
if up is not None:
    t0 = time.time()

    # Read input file
    raw_df = read_any_to_polars(up.getvalue(), up.name)
    st.success(f"✅ Loaded file: {up.name} — {raw_df.height:,} rows, {len(raw_df.columns)} columns")

    # Parse transcripts into utterances
    st.info("📄 Splitting transcript lines into speaker and timestamp segments...")
    utter = explode_raw_transcript_column(raw_df, raw_col=text_col, call_id_col=call_id_col)
    st.success(f"Utterances extracted: {utter.height:,}")

    # Categorize with DuckDB rules
    with st.spinner("⚙️ Applying categorization rules via DuckDB..."):
        matches, journey = categorize_utterances(utter, rules_df)

    # Display results
    left, right = st.columns([3, 2])
    with left:
        section("Per-Utterance Matches")
        st.dataframe(matches.head(500).to_pandas(), use_container_width=True, height=420)

        section("Journey Summary (Per Call)")
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

        section("Speaker Term Highlights")
        buckets = speaker_term_buckets(utter)
        if not buckets.is_empty():
            st.dataframe(buckets.to_pandas(), use_container_width=True, height=300)

    # ------------------------------------------------------------
    # Download outputs
    # ------------------------------------------------------------
    section("Downloads")
    matches_parquet = io.BytesIO()
    matches.write_parquet(matches_parquet)
    st.download_button(
        "⬇️ Download Matches (Parquet)",
        data=matches_parquet.getvalue(),
        file_name="categorized_results.parquet"
    )

    matches_xlsx = io.BytesIO()
    matches.to_pandas().to_excel(matches_xlsx, index=False)
    st.download_button(
        "⬇️ Download Matches (Excel)",
        data=matches_xlsx.getvalue(),
        file_name="categorized_results.xlsx"
    )

    st.caption(f"Processed in {(time.time() - t0):.2f} seconds")

else:
    st.info("👆 Upload a transcript file to begin analysis.")
