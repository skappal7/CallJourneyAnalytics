import os
import duckdb
import polars as pl


# ---------------------------------------------------------------------
# Default paths to embedded rules
# ---------------------------------------------------------------------
RULE_PATHS = ["rules/rules_embedded.parquet", "rules/rules_embedded.csv"]


# ---------------------------------------------------------------------
# Load rules
# ---------------------------------------------------------------------
def load_rules() -> pl.DataFrame:
    """
    Load embedded query rules from CSV or Parquet.
    Must contain columns: rule_id, query_name, category, subcategory, compiled_regex
    """
    for p in RULE_PATHS:
        if os.path.exists(p):
            return (
                pl.read_parquet(p)
                if p.endswith(".parquet")
                else pl.read_csv(p, infer_schema_length=10000)
            )
    raise FileNotFoundError(
        "Rules not found. Place your compiled rules at rules/rules_embedded.csv or .parquet"
    )


# ---------------------------------------------------------------------
# Register DataFrame to DuckDB (zero-copy via Arrow)
# ---------------------------------------------------------------------
def _register(con, name, df):
    con.register(name, df.to_arrow())


# ---------------------------------------------------------------------
# Categorization engine
# ---------------------------------------------------------------------
def categorize_utterances(utter_df: pl.DataFrame, rules_df: pl.DataFrame):
    """
    Join utterances with rule patterns using DuckDB regex engine.
    Returns: (matches_df, journey_summary_df)
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=ALL;")

    _register(con, "utter", utter_df)
    _register(con, "rules", rules_df)

    # Rule application
    sql = """
        SELECT 
            u.call_id, 
            u.timestamp, 
            u.speaker, 
            u.text,
            r.rule_id,
            r.query_name,
            r.category,
            r.subcategory,
            REGEXP_LIKE(u.text, r.compiled_regex) AS match
        FROM utter u, rules r
        WHERE match
    """

    matches = con.execute(sql).pl()

    # Summary aggregation
    summary_sql = """
        SELECT 
            call_id, 
            LIST(DISTINCT category) AS categories,
            LIST(DISTINCT subcategory) AS subcategories,
            COUNT(*) AS total_hits
        FROM matches
        GROUP BY call_id
    """
    summary = con.execute(summary_sql).pl()

    return matches, summary


# ---------------------------------------------------------------------
# Speaker-term distribution
# ---------------------------------------------------------------------
def speaker_term_buckets(utter_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract frequently used words per speaker.
    Useful for conversational tone and engagement analysis.
    """
    def extract_terms(text: str):
        import re
        return [w.lower() for w in re.findall(r"[A-Za-z]{4,}", str(text))][:50]

    df = utter_df.with_columns(
        pl.col("text").map_elements(extract_terms).alias("terms")
    ).explode("terms")

    return (
        df.group_by(["call_id", "speaker", "terms"])
        .len()
        .sort(["call_id", "speaker", "len"], descending=[False, False, True])
    )
