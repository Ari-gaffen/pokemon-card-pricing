from pathlib import Path

import pandas as pd

from pokemon_pricing.config import GRADING_COMPANY_SCORE, RAW_CONDITION_SCORE

ENRICHMENT_COLUMNS = [
    "card_id",
    "language",
    "tcg_price_type",
    "market_segment",
    "condition",
    "grading_company",
    "grade",
    "observed_price",
    "source_name",
    "recent_sold_count",
    "population_count",
    "sold_window_days",
    "as_of_date",
]


def load_price_enrichment(paths: list[Path] | None) -> pd.DataFrame:
    """Load raw-condition or graded-price observations from CSV files."""
    if not paths:
        return pd.DataFrame(columns=ENRICHMENT_COLUMNS)

    frames = []
    for path in paths:
        if path:
            frames.append(pd.read_csv(path))

    if not frames:
        return pd.DataFrame(columns=ENRICHMENT_COLUMNS)

    enrichment = pd.concat(frames, ignore_index=True)
    missing = sorted(set(required_enrichment_columns()) - set(enrichment.columns))
    if missing:
        raise ValueError(f"Missing enrichment columns: {', '.join(missing)}")

    return normalize_enrichment(enrichment)


def required_enrichment_columns() -> list[str]:
    return [
        "card_id",
        "market_segment",
        "observed_price",
        "source_name",
    ]


def normalize_enrichment(enrichment: pd.DataFrame) -> pd.DataFrame:
    normalized = enrichment.copy()
    for column in ENRICHMENT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    normalized["market_segment"] = normalized["market_segment"].str.lower().str.strip()
    normalized["language"] = normalized["language"].fillna("English").str.strip()
    normalized["condition"] = normalized["condition"].fillna("").str.lower().str.strip()
    normalized["grading_company"] = (
        normalized["grading_company"].fillna("").str.lower().str.strip()
    )
    normalized.loc[normalized["market_segment"] == "raw", "grading_company"] = "raw"
    normalized.loc[normalized["market_segment"] == "raw", "grade"] = 0

    numeric_columns = ["grade", "observed_price", "recent_sold_count", "population_count"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized[ENRICHMENT_COLUMNS]


def append_price_variants(card_frame: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    """Return one row per priced variant: baseline raw NM plus optional enrichment rows."""
    base = card_frame.copy()
    base["market_segment"] = "raw"
    base["language"] = base.get("language", "English")
    base["condition"] = "near_mint"
    base["grading_company"] = "raw"
    base["grade"] = 0
    base["observed_price"] = base["tcg_market"]
    base["source_name"] = "pokemon_tcg_api_tcgplayer_market"
    base["recent_sold_count"] = None
    base["population_count"] = None
    base["sold_window_days"] = None
    base["as_of_date"] = None

    if enrichment.empty:
        variants = base
    else:
        join_keys = ["card_id"]
        if "tcg_price_type" in enrichment.columns and enrichment["tcg_price_type"].notna().any():
            join_keys.append("tcg_price_type")

        enriched = card_frame.merge(enrichment, on=join_keys, how="inner", suffixes=("", "_source"))
        variants = pd.concat([base, enriched], ignore_index=True, sort=False)

    return add_variant_features(variants)


def add_variant_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["is_graded"] = (frame["market_segment"] == "graded").astype(int)
    frame["condition_score"] = frame["condition"].map(RAW_CONDITION_SCORE)
    frame["condition_score"] = frame["condition_score"].fillna(0)
    frame["grade_numeric"] = pd.to_numeric(frame["grade"], errors="coerce").fillna(0)
    frame["grading_company_score"] = frame["grading_company"].map(GRADING_COMPANY_SCORE).fillna(0)
    frame["recent_sold_count"] = pd.to_numeric(frame["recent_sold_count"], errors="coerce")
    frame["population_count"] = pd.to_numeric(frame["population_count"], errors="coerce")
    frame["sold_window_days"] = pd.to_numeric(frame["sold_window_days"], errors="coerce")
    return frame
