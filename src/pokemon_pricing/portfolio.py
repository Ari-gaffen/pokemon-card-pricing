from __future__ import annotations

import json
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from pokemon_pricing.config import PORTFOLIO_DIR, PROCESSED_DIR, VARIANT_KEY_COLUMNS

PORTFOLIO_COLUMNS = [
    "holding_id",
    "item_type",
    "card_id",
    "language",
    "tcg_price_type",
    "market_segment",
    "condition",
    "grading_company",
    "grade",
    "product_name",
    "image_url",
    "copies_owned",
    "purchase_price_each",
    "estimated_unit_value",
    "acquired_date",
    "notes",
]

WISHLIST_COLUMNS = [
    "wishlist_id",
    "item_type",
    "card_id",
    "language",
    "tcg_price_type",
    "market_segment",
    "condition",
    "grading_company",
    "grade",
    "product_name",
    "target_price",
    "priority",
    "notes",
]

VALUE_COLUMNS = [
    "holding_id",
    "item_type",
    "card_id",
    "name",
    "set_name",
    "rarity",
    "product_name",
    "image_url",
    "language",
    "tcg_price_type",
    "market_segment",
    "condition",
    "grading_company",
    "grade",
    "copies_owned",
    "estimated_unit_value",
    "modeled_fair_price",
    "total_estimated_value",
    "pricing_label",
    "source_name",
]


def init_portfolio_files(portfolio_dir: Path = PORTFOLIO_DIR) -> dict[str, Path]:
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "portfolio": portfolio_dir / "portfolio.csv",
        "wishlist": portfolio_dir / "wishlist.csv",
        "history": portfolio_dir / "portfolio_value_history.csv",
    }

    if not paths["portfolio"].exists():
        _template_portfolio().to_csv(paths["portfolio"], index=False)
    if not paths["wishlist"].exists():
        _template_wishlist().to_csv(paths["wishlist"], index=False)
    if not paths["history"].exists():
        pd.DataFrame(
            columns=[
                "snapshot_date",
                "raw_value",
                "graded_value",
                "sealed_value",
                "portfolio_value",
                "holding_count",
            ]
        ).to_csv(paths["history"], index=False)

    return paths


def value_portfolio(
    scored_cards_path: Path = PROCESSED_DIR / "scored_cards.csv",
    portfolio_path: Path = PORTFOLIO_DIR / "portfolio.csv",
) -> pd.DataFrame:
    portfolio = _read_csv(portfolio_path, PORTFOLIO_COLUMNS)
    if portfolio.empty:
        return pd.DataFrame(columns=VALUE_COLUMNS)

    scored = pd.read_csv(scored_cards_path, low_memory=False)
    portfolio = _normalize_portfolio(portfolio)
    scored = _normalize_variants(scored)

    card_holdings = portfolio[portfolio["item_type"] == "card"].copy()
    sealed_holdings = portfolio[portfolio["item_type"] == "sealed"].copy()
    valued_frames = []

    if not card_holdings.empty:
        columns = VARIANT_KEY_COLUMNS + [
            "name",
            "set_name",
            "rarity",
            "image_small",
            "image_large",
            "modeled_fair_price",
            "observed_price",
            "pricing_label",
            "source_name",
        ]
        available_columns = [column for column in columns if column in scored.columns]
        card_values = card_holdings.merge(
            scored[available_columns],
            on=VARIANT_KEY_COLUMNS,
            how="left",
        )
        card_values["estimated_unit_value"] = (
            card_values["modeled_fair_price"]
            .fillna(card_values["observed_price"])
            .fillna(card_values["estimated_unit_value"])
        )
        card_values["image_url"] = card_values["image_url"].replace("", pd.NA)
        if "image_large" in card_values.columns:
            card_values["image_url"] = card_values["image_url"].fillna(card_values["image_large"])
        if "image_small" in card_values.columns:
            card_values["image_url"] = card_values["image_url"].fillna(card_values["image_small"])
        card_values["language"] = card_values["language"].fillna("English")
        valued_frames.append(card_values)

    if not sealed_holdings.empty:
        sealed_values = sealed_holdings.copy()
        sealed_values["name"] = sealed_values["product_name"]
        sealed_values["pricing_label"] = "manual_value"
        sealed_values["source_name"] = "portfolio"
        valued_frames.append(sealed_values)

    valued = pd.concat(valued_frames, ignore_index=True, sort=False)
    valued["copies_owned"] = pd.to_numeric(valued["copies_owned"], errors="coerce").fillna(0)
    valued["estimated_unit_value"] = pd.to_numeric(
        valued["estimated_unit_value"], errors="coerce"
    ).fillna(0)
    valued["total_estimated_value"] = valued["copies_owned"] * valued["estimated_unit_value"]
    valued["product_name"] = valued["product_name"].fillna(valued["name"])
    valued["image_url"] = valued["image_url"].fillna("")
    if "modeled_fair_price" not in valued.columns:
        valued["modeled_fair_price"] = None
    valued["modeled_fair_price"] = pd.to_numeric(valued["modeled_fair_price"], errors="coerce")
    return valued.reindex(columns=VALUE_COLUMNS)


def summarize_portfolio(valued: pd.DataFrame) -> dict[str, Any]:
    if valued.empty:
        return {
            "raw_value": 0.0,
            "graded_value": 0.0,
            "sealed_value": 0.0,
            "portfolio_value": 0.0,
            "holding_count": 0,
        }

    raw_value = _segment_value(valued, "card", "raw")
    graded_value = _segment_value(valued, "card", "graded")
    sealed_value = float(valued.loc[valued["item_type"] == "sealed", "total_estimated_value"].sum())
    return {
        "raw_value": raw_value,
        "graded_value": graded_value,
        "sealed_value": sealed_value,
        "portfolio_value": raw_value + graded_value + sealed_value,
        "holding_count": int(valued["holding_id"].nunique()),
    }


def snapshot_portfolio_value(
    valued: pd.DataFrame,
    history_path: Path = PORTFOLIO_DIR / "portfolio_value_history.csv",
    snapshot_date: date | None = None,
) -> pd.DataFrame:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = summarize_portfolio(valued)
    snapshot["snapshot_date"] = (snapshot_date or date.today()).isoformat()
    snapshot_frame = pd.DataFrame([snapshot])

    history = _read_csv(
        history_path,
        [
            "snapshot_date",
            "raw_value",
            "graded_value",
            "sealed_value",
            "portfolio_value",
            "holding_count",
        ],
    )
    history = history[history["snapshot_date"] != snapshot["snapshot_date"]]
    history = pd.concat([history, snapshot_frame], ignore_index=True, sort=False)
    history = history.sort_values("snapshot_date")
    history.to_csv(history_path, index=False)
    return history


def recommend_cards(
    scored_cards_path: Path = PROCESSED_DIR / "scored_cards.csv",
    portfolio_path: Path = PORTFOLIO_DIR / "portfolio.csv",
    wishlist_path: Path = PORTFOLIO_DIR / "wishlist.csv",
    limit: int = 20,
) -> pd.DataFrame:
    scored = _normalize_variants(pd.read_csv(scored_cards_path, low_memory=False))
    portfolio = _normalize_portfolio(_read_csv(portfolio_path, PORTFOLIO_COLUMNS))
    wishlist = _normalize_wishlist(_read_csv(wishlist_path, WISHLIST_COLUMNS))

    owned_card_ids = set(portfolio.loc[portfolio["item_type"] == "card", "card_id"].dropna())
    wishlist_card_ids = set(wishlist.loc[wishlist["item_type"] == "card", "card_id"].dropna())
    candidates = scored[~scored["card_id"].isin(owned_card_ids | wishlist_card_ids)].copy()
    if candidates.empty:
        return candidates

    owned_variants = scored[scored["card_id"].isin(owned_card_ids)]
    taste_weights = _taste_weights(owned_variants)
    candidates["taste_score"] = candidates.apply(lambda row: _score_taste(row, taste_weights), axis=1)
    candidates["deal_bonus"] = candidates["pricing_label"].map(
        {"under_priced": 2.0, "well_priced": 0.75, "over_priced": -1.0}
    ).fillna(0)
    candidates["recommendation_score"] = candidates["taste_score"] + candidates["deal_bonus"]

    columns = [
        "card_id",
        "name",
        "set_name",
        "rarity",
        "language",
        "tcg_price_type",
        "market_segment",
        "condition",
        "grading_company",
        "grade",
        "modeled_fair_price",
        "observed_price",
        "pricing_label",
        "recommendation_score",
        "image_small",
        "image_large",
    ]
    return candidates.sort_values("recommendation_score", ascending=False).head(limit)[columns]


def build_dashboard(
    valued: pd.DataFrame,
    recommendations: pd.DataFrame,
    catalog: pd.DataFrame | None = None,
    wishlist: pd.DataFrame | None = None,
    history_path: Path = PORTFOLIO_DIR / "portfolio_value_history.csv",
    output_path: Path = PROCESSED_DIR / "portfolio_dashboard.html",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = _read_csv(history_path, ["snapshot_date", "portfolio_value"])
    summary = summarize_portfolio(valued)
    cards = _card_summary(valued)

    payload = {
        "summary": summary,
        "cards": cards,
        "holdings": valued.fillna("").to_dict(orient="records"),
        "wishlist": _wishlist_records(wishlist, catalog),
        "recommendations": recommendations.fillna("").to_dict(orient="records"),
        "catalog": _catalog_records(catalog),
        "character_premiums": _character_premium_records(catalog),
        "history": history.fillna("").to_dict(orient="records"),
    }
    html = _dashboard_html(payload)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _template_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "holding_id": "h001",
                "item_type": "card",
                "card_id": "base1-4",
                "language": "English",
                "tcg_price_type": "holofoil",
                "market_segment": "raw",
                "condition": "near_mint",
                "grading_company": "raw",
                "grade": 0,
                "product_name": "",
                "image_url": "",
                "copies_owned": 1,
                "purchase_price_each": "",
                "estimated_unit_value": "",
                "acquired_date": "",
                "notes": "Example raw card. Replace or delete this row.",
            },
            {
                "holding_id": "h002",
                "item_type": "card",
                "card_id": "base1-4",
                "language": "English",
                "tcg_price_type": "holofoil",
                "market_segment": "graded",
                "condition": "",
                "grading_company": "psa",
                "grade": 9,
                "product_name": "",
                "image_url": "",
                "copies_owned": 1,
                "purchase_price_each": "",
                "estimated_unit_value": "",
                "acquired_date": "",
                "notes": "Example graded card. Replace or delete this row.",
            },
            {
                "holding_id": "s001",
                "item_type": "sealed",
                "card_id": "",
                "language": "English",
                "tcg_price_type": "",
                "market_segment": "sealed",
                "condition": "sealed",
                "grading_company": "",
                "grade": "",
                "product_name": "Example Booster Box",
                "image_url": "",
                "copies_owned": 1,
                "purchase_price_each": "",
                "estimated_unit_value": 120.0,
                "acquired_date": "",
                "notes": "Sealed products use estimated_unit_value until a sealed API is added.",
            },
        ],
        columns=PORTFOLIO_COLUMNS,
    )


def _template_wishlist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "wishlist_id": "w001",
                "item_type": "card",
                "card_id": "base1-2",
                "language": "English",
                "tcg_price_type": "holofoil",
                "market_segment": "raw",
                "condition": "near_mint",
                "grading_company": "raw",
                "grade": 0,
                "product_name": "",
                "target_price": "",
                "priority": "high",
                "notes": "Example wishlist card. Replace or delete this row.",
            }
        ],
        columns=WISHLIST_COLUMNS,
    )


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def _normalize_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    normalized = portfolio.copy()
    for column in PORTFOLIO_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    normalized["item_type"] = normalized["item_type"].fillna("").str.lower().str.strip()
    normalized["language"] = normalized["language"].fillna("English").str.strip()
    normalized["market_segment"] = normalized["market_segment"].fillna("").str.lower().str.strip()
    normalized["condition"] = normalized["condition"].fillna("").str.lower().str.strip()
    normalized["grading_company"] = (
        normalized["grading_company"].fillna("").str.lower().str.strip()
    )
    normalized["grade"] = pd.to_numeric(normalized["grade"], errors="coerce").fillna(0)
    normalized["copies_owned"] = pd.to_numeric(
        normalized["copies_owned"], errors="coerce"
    ).fillna(0)
    normalized["estimated_unit_value"] = pd.to_numeric(
        normalized["estimated_unit_value"], errors="coerce"
    )
    return normalized[PORTFOLIO_COLUMNS]


def _normalize_wishlist(wishlist: pd.DataFrame) -> pd.DataFrame:
    normalized = wishlist.copy()
    for column in WISHLIST_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    normalized["item_type"] = normalized["item_type"].fillna("").str.lower().str.strip()
    normalized["language"] = normalized["language"].fillna("English").str.strip()
    return normalized[WISHLIST_COLUMNS]


def _normalize_variants(scored: pd.DataFrame) -> pd.DataFrame:
    normalized = scored.copy()
    for column in VARIANT_KEY_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    normalized["market_segment"] = normalized["market_segment"].fillna("").str.lower().str.strip()
    normalized["language"] = normalized["language"].fillna("English").str.strip()
    normalized["condition"] = normalized["condition"].fillna("").str.lower().str.strip()
    normalized["grading_company"] = (
        normalized["grading_company"].fillna("").str.lower().str.strip()
    )
    normalized["grade"] = pd.to_numeric(normalized["grade"], errors="coerce").fillna(0)
    return normalized


def _segment_value(valued: pd.DataFrame, item_type: str, segment: str) -> float:
    mask = (valued["item_type"] == item_type) & (valued["market_segment"] == segment)
    return float(valued.loc[mask, "total_estimated_value"].sum())


def _taste_weights(owned: pd.DataFrame) -> dict[str, dict[Any, float]]:
    if owned.empty:
        return {}
    fields = ["character", "set_series", "set_name", "rarity", "market_segment", "artist"]
    weights = {}
    for field in fields:
        if field in owned.columns:
            weights[field] = owned[field].value_counts(normalize=True).to_dict()
    return weights


def _score_taste(row: pd.Series, weights: dict[str, dict[Any, float]]) -> float:
    score = 0.0
    multipliers = {
        "character": 5.0,
        "set_series": 2.0,
        "set_name": 2.5,
        "rarity": 1.5,
        "market_segment": 1.0,
        "artist": 1.0,
    }
    for field, values in weights.items():
        score += values.get(row.get(field), 0.0) * multipliers.get(field, 1.0)
    return score


def _card_summary(valued: pd.DataFrame) -> list[dict[str, Any]]:
    if valued.empty:
        return []
    cards = valued[valued["item_type"] == "card"].copy()
    if cards.empty:
        return []
    grouped = (
        cards.groupby(["card_id", "name", "set_name"], dropna=False)
        .agg(
            total_estimated_value=("total_estimated_value", "sum"),
            copies_owned=("copies_owned", "sum"),
            image_url=("image_url", "first"),
        )
        .reset_index()
        .sort_values("total_estimated_value", ascending=False)
    )
    return grouped.to_dict(orient="records")


def _catalog_records(catalog: pd.DataFrame | None) -> list[dict[str, Any]]:
    if catalog is None or catalog.empty:
        return []
    columns = [
        "card_id",
        "name",
        "set_name",
        "rarity",
        "character",
        "supertype",
        "national_pokedex_number",
        "set_id",
        "set_series",
        "set_rarity_price_rank",
        "character_print_count",
        "character_avg_set_rarity_price_rank",
        "image_small",
        "image_large",
        "tcg_price_type",
        "market_segment",
        "condition",
        "grading_company",
        "grade",
        "modeled_fair_price",
        "observed_price",
        "pricing_label",
    ]
    available = [column for column in columns if column in catalog.columns]
    records = catalog[available].copy()
    return records.fillna("").to_dict(orient="records")


def _wishlist_records(
    wishlist: pd.DataFrame | None, catalog: pd.DataFrame | None
) -> list[dict[str, Any]]:
    if wishlist is None or wishlist.empty:
        return []
    normalized = _normalize_wishlist(wishlist)
    if catalog is None or catalog.empty:
        return normalized.fillna("").to_dict(orient="records")

    scored = _normalize_variants(catalog)
    columns = VARIANT_KEY_COLUMNS + [
        "name",
        "set_name",
        "rarity",
        "image_small",
        "image_large",
        "modeled_fair_price",
        "observed_price",
        "pricing_label",
    ]
    available = [column for column in columns if column in scored.columns]
    merged = normalized.merge(scored[available], on=VARIANT_KEY_COLUMNS, how="left")
    merged["image_url"] = merged.get("image_large", pd.Series(index=merged.index)).fillna(
        merged.get("image_small", pd.Series(index=merged.index))
    )
    return merged.fillna("").to_dict(orient="records")


def _character_premium_records(catalog: pd.DataFrame | None) -> list[dict[str, Any]]:
    if catalog is None or catalog.empty or "character" not in catalog.columns:
        return []

    frame = catalog.copy()
    if "image_small" not in frame.columns:
        frame["image_small"] = frame.get("image_large", "")
    if "set_id" not in frame.columns:
        frame["set_id"] = frame.get("set_name", "")
    if "rarity" not in frame.columns:
        frame["rarity"] = ""
    if "supertype" not in frame.columns:
        frame["supertype"] = ""
    if "national_pokedex_number" not in frame.columns:
        frame["national_pokedex_number"] = None
    frame["national_pokedex_number"] = pd.to_numeric(
        frame["national_pokedex_number"], errors="coerce"
    )
    frame = frame[
        (frame["supertype"] == "Pokémon")
        & frame["national_pokedex_number"].between(1, 1025)
    ].copy()
    if frame.empty:
        return []
    frame["character"] = _canonical_species_from_catalog(frame)

    price_column = "modeled_fair_price" if "modeled_fair_price" in frame.columns else "observed_price"
    frame[price_column] = pd.to_numeric(frame.get(price_column), errors="coerce")
    if "set_rarity_price_rank" not in frame.columns:
        frame["set_rarity_price_rank"] = frame.groupby(["set_id", "rarity"])[price_column].rank(
            method="average", ascending=False
        )
    else:
        frame["set_rarity_price_rank"] = pd.to_numeric(
            frame["set_rarity_price_rank"], errors="coerce"
        )
        if frame["set_rarity_price_rank"].dropna().le(1).all():
            frame["set_rarity_price_rank"] = frame.groupby(["set_id", "rarity"])[price_column].rank(
                method="average", ascending=False
            )

    grouped = (
        frame.dropna(subset=["character"])
        .groupby("character", dropna=False)
        .agg(
            avg_set_rarity_price_rank=("set_rarity_price_rank", "mean"),
            print_count=("card_id", "nunique"),
            image_url=("image_small", "first"),
        )
        .reset_index()
    )
    if grouped.empty:
        return []

    grouped["rank_signal"] = 1 / grouped["avg_set_rarity_price_rank"].clip(lower=0.01)
    grouped["print_signal"] = grouped["print_count"].rank(method="average", pct=True)
    grouped["premium_raw"] = grouped["rank_signal"] * (1 + grouped["print_signal"])
    grouped["print_count_weighted_rank"] = grouped["premium_raw"].rank(
        method="dense", ascending=False
    )
    min_raw = grouped["premium_raw"].min()
    max_raw = grouped["premium_raw"].max()
    if max_raw == min_raw:
        grouped["normalized_score"] = 10.0
    else:
        grouped["normalized_score"] = 1 + 9 * (
            (grouped["premium_raw"] - min_raw) / (max_raw - min_raw)
        )

    grouped = grouped.sort_values(["print_count_weighted_rank", "character"]).head(50)
    grouped["print_count_weighted_rank"] = grouped["print_count_weighted_rank"].astype(int)
    grouped["normalized_score"] = grouped["normalized_score"].round(1)
    grouped["avg_set_rarity_price_rank"] = grouped["avg_set_rarity_price_rank"].round(1)
    return grouped[
        [
            "character",
            "image_url",
            "print_count_weighted_rank",
            "normalized_score",
            "avg_set_rarity_price_rank",
            "print_count",
        ]
    ].fillna("").to_dict(orient="records")


def _canonical_species_from_catalog(frame: pd.DataFrame) -> pd.Series:
    candidates = frame[["national_pokedex_number", "name", "character"]].copy()
    candidates["species_candidate"] = candidates["name"].map(_clean_species_name)
    species_by_number = (
        candidates.sort_values(
            ["national_pokedex_number", "species_candidate"],
            key=lambda column: column.astype(str).str.len()
            if column.name == "species_candidate"
            else column,
        )
        .groupby("national_pokedex_number")["species_candidate"]
        .first()
        .to_dict()
    )
    return frame["national_pokedex_number"].map(species_by_number).fillna(frame["character"])


def _clean_species_name(name: str) -> str:
    text = str(name)
    if "'s " in text:
        text = text.split("'s ", 1)[1]
    for marker in [" with ", " VMAX", " VSTAR", "-EX", " ex", " GX", " V", " Lv.", " Star"]:
        text = text.split(marker, 1)[0]
    return text.strip()


def _dashboard_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload)
    title = "Pokemon Portfolio Dashboard"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }}
    body {{ margin: 0; background: #f7f7f4; color: #1e2328; }}
    header {{ padding: 24px 32px 12px; background: #ffffff; border-bottom: 1px solid #ddded8; }}
    h1 {{ margin: 0 0 16px; font-size: 28px; letter-spacing: 0; }}
    main {{ padding: 24px 32px 40px; display: grid; gap: 24px; }}
    nav {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .stat, .panel, .card-row, .owned-card {{ background: #ffffff; border: 1px solid #ddded8; border-radius: 8px; }}
    .stat {{ padding: 14px; }}
    .label {{ color: #667078; font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .grid {{ display: grid; grid-template-columns: minmax(220px, 340px) 1fr; gap: 18px; align-items: start; }}
    .panel {{ padding: 16px; }}
    select, input {{ width: 100%; box-sizing: border-box; padding: 10px; border-radius: 6px; border: 1px solid #b7bbb2; font-size: 14px; }}
    button {{ padding: 9px 12px; border: 1px solid #1e2328; border-radius: 6px; background: #1e2328; color: #ffffff; cursor: pointer; }}
    button.secondary {{ background: #ffffff; color: #1e2328; }}
    button.danger {{ background: #8d2430; border-color: #8d2430; }}
    button.tab-button {{ background: #ffffff; color: #1e2328; }}
    button.tab-button.active {{ background: #1e2328; color: #ffffff; }}
    .tab-section {{ display: none; }}
    .tab-section.active {{ display: grid; gap: 24px; }}
    .selected {{ display: grid; grid-template-columns: 160px 1fr; gap: 18px; margin-top: 16px; }}
    img {{ width: 100%; max-width: 160px; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid #e6e6e1; }}
    th {{ color: #667078; font-size: 12px; text-transform: uppercase; }}
    .cards {{ display: grid; gap: 10px; }}
    .card-row {{ padding: 12px; display: grid; grid-template-columns: 54px 1fr auto; gap: 12px; align-items: center; }}
    .card-row img {{ max-width: 54px; }}
    .owned-list {{ display: grid; gap: 8px; max-height: 520px; overflow: auto; }}
    .owned-card {{ width: 100%; text-align: left; padding: 10px; background: #ffffff; color: #1e2328; display: grid; grid-template-columns: 44px 1fr; gap: 10px; align-items: center; }}
    .owned-card img {{ max-width: 44px; }}
    .owned-card.active {{ border-color: #1e2328; box-shadow: inset 3px 0 0 #1e2328; }}
    .toolbar {{ display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; }}
    .search-results {{ display: grid; gap: 10px; margin-top: 12px; }}
    .search-card {{ display: grid; grid-template-columns: 52px 1fr auto; gap: 12px; align-items: center; padding: 10px 0; border-bottom: 1px solid #e6e6e1; }}
    .search-card img {{ max-width: 52px; }}
    .empty {{ color: #667078; padding: 10px 0; }}
    .premium-grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 28px; }}
    .premium-row {{ display: grid; grid-template-columns: 48px 1fr 70px 80px 90px 70px; gap: 10px; align-items: center; padding: 8px 0; border-bottom: 1px solid #e6e6e1; }}
    .premium-row img {{ max-width: 42px; }}
    .premium-head {{ color: #667078; font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    .zoomable {{ cursor: zoom-in; }}
    .modal {{ position: fixed; inset: 0; display: none; place-items: center; background: rgba(0,0,0,.72); z-index: 20; padding: 24px; }}
    .modal.active {{ display: grid; }}
    .modal img {{ max-width: min(92vw, 680px); max-height: 92vh; }}
    .right-stack {{ display: grid; gap: 4px; justify-items: end; }}
    .small-controls {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 8px; margin-top: 8px; }}
    .muted {{ color: #667078; }}
    @media (max-width: 780px) {{
      main, header {{ padding-left: 16px; padding-right: 16px; }}
      .grid, .selected {{ grid-template-columns: 1fr; }}
      .premium-grid {{ grid-template-columns: 1fr; }}
      .premium-row {{ grid-template-columns: 42px 1fr; }}
      .premium-head {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <div class="stats" id="stats"></div>
    <nav>
      <button class="tab-button active" data-tab="portfolio">Portfolio</button>
      <button class="tab-button" data-tab="add">Add Cards</button>
      <button class="tab-button" data-tab="wishlist">Wishlist</button>
      <button class="tab-button" data-tab="premium">Character Premium</button>
    </nav>
  </header>
  <main>
    <section id="tab-portfolio" class="tab-section active">
      <div class="grid">
        <div class="panel">
          <div class="label">Owned Cards</div>
          <div id="ownedList" class="owned-list"></div>
          <div id="selectedCard" class="selected"></div>
        </div>
        <div class="panel">
          <h2>Owned Cards</h2>
          <div class="muted">Edit copies or manual value, remove rows, then export the portfolio CSV.</div>
          <div id="variantTable"></div>
        </div>
      </div>
      <div class="panel">
        <h2>Cards You Might Like</h2>
        <div id="recommendations" class="cards"></div>
      </div>
      <div class="panel">
        <h2>Portfolio Value Over Time</h2>
        <div id="history"></div>
      </div>
    </section>
    <section id="tab-add" class="tab-section">
      <div class="panel">
        <h2>Add Cards</h2>
        <div class="toolbar">
          <input id="catalogSearch" type="search" placeholder="Search by card name, set, rarity, or card id">
          <button class="secondary" id="exportPortfolio">Export portfolio CSV</button>
        </div>
        <div id="searchResults" class="search-results"></div>
      </div>
    </section>
    <section id="tab-wishlist" class="tab-section">
      <div class="panel">
        <h2>Wishlist</h2>
        <div class="muted">Wishlist items do not count toward portfolio totals.</div>
        <button class="secondary" id="exportWishlist">Export wishlist CSV</button>
        <div id="wishlistTable"></div>
      </div>
    </section>
    <section id="tab-premium" class="tab-section">
      <div class="panel">
        <h2>Character Premium</h2>
        <div class="muted">
          Score calculation: 1. keep only Pokemon species rows with a National Pokedex number from 1 to 1025;
          2. roll special names to the species by Pokedex number; 3. rank each card by evaluated price within its set and rarity;
          4. average those ranks by species; 5. combine inverse average rank with print-count percentile;
          6. normalize the result to a 1.0-10.0 score.
        </div>
        <div id="characterPremium"></div>
      </div>
    </section>
  </main>
  <div class="modal" id="imageModal"><img id="modalImage" alt="Large card image"></div>
  <script>
    const data = {data};
    const portfolioColumns = [
      "holding_id", "item_type", "card_id", "language", "tcg_price_type", "market_segment", "condition",
      "grading_company", "grade", "product_name", "image_url", "copies_owned",
      "purchase_price_each", "estimated_unit_value", "acquired_date", "notes"
    ];
    const wishlistColumns = [
      "wishlist_id", "item_type", "card_id", "language", "tcg_price_type", "market_segment",
      "condition", "grading_company", "grade", "product_name", "target_price", "priority", "notes"
    ];
    let workingHoldings = data.holdings.map(item => ({{ ...item }}));
    let workingWishlist = (data.wishlist || []).map(item => ({{ ...item }}));
    let selectedCardId = "";
    const money = value => Number(value || 0).toLocaleString(
      undefined, {{ style: "currency", currency: "USD" }}
    );

    function summarize(holdings) {{
      const totalFor = (itemType, segment) => holdings
        .filter(item => item.item_type === itemType && item.market_segment === segment)
        .reduce((sum, item) => sum + Number(item.total_estimated_value || 0), 0);
      const raw = totalFor("card", "raw");
      const graded = totalFor("card", "graded");
      const sealed = totalFor("sealed", "sealed");
      return {{ raw_value: raw, graded_value: graded, sealed_value: sealed, portfolio_value: raw + graded + sealed }};
    }}

    function cardSummaries() {{
      const groups = new Map();
      workingHoldings.filter(item => item.item_type === "card").forEach(item => {{
        const existing = groups.get(item.card_id) || {{
          card_id: item.card_id,
          name: item.name || item.product_name || item.card_id,
          set_name: item.set_name || "",
          image_url: item.image_url || "",
          copies_owned: 0,
          total_estimated_value: 0
        }};
        existing.copies_owned += Number(item.copies_owned || 0);
        existing.total_estimated_value += Number(item.total_estimated_value || 0);
        if (!existing.image_url && item.image_url) existing.image_url = item.image_url;
        groups.set(item.card_id, existing);
      }});
      return Array.from(groups.values()).sort((a, b) => b.total_estimated_value - a.total_estimated_value);
    }}

    function renderStats() {{
      const summary = summarize(workingHoldings);
      const stats = [
        ["Portfolio", summary.portfolio_value],
        ["Raw Cards", summary.raw_value],
        ["Graded Cards", summary.graded_value],
        ["Sealed", summary.sealed_value],
      ];
      document.getElementById("stats").innerHTML = stats.map(([label, value]) => `
        <div class="stat"><div class="label">${{label}}</div><div class="value">${{money(value)}}</div></div>
      `).join("");
    }}

    function renderOwnedList() {{
      const cards = cardSummaries();
      document.getElementById("ownedList").innerHTML = cards.map(card => `
        <button class="owned-card ${{card.card_id === selectedCardId ? "active" : ""}}" data-card-id="${{card.card_id}}">
          <img class="zoomable" src="${{card.image_url || ""}}" alt="${{card.name || "Card image"}}" data-large-src="${{card.image_url || ""}}">
          <span>
            <strong>${{card.name}}</strong>
            <div class="muted">${{card.set_name || "Unknown set"}} - ${{Number(card.copies_owned || 0)}} copies - ${{money(card.total_estimated_value)}}</div>
          </span>
        </button>
      `).join("") || `<div class="empty">No card holdings yet. Search below to add one.</div>`;
      document.querySelectorAll(".owned-card").forEach(button => {{
        button.addEventListener("click", () => renderSelected(button.dataset.cardId));
      }});
      bindZoomImages();
    }}

    function renderSelected(cardId) {{
      selectedCardId = cardId || "";
      renderOwnedList();
      if (!cardId) {{
        const summary = summarize(workingHoldings);
        document.getElementById("selectedCard").innerHTML = `
          <div>
            <h2>Portfolio Summary</h2>
            <p class="muted">Select a card to inspect owned variants.</p>
            <p><strong>${{money(summary.portfolio_value)}} total estimated value</strong></p>
          </div>
        `;
        renderVariants(workingHoldings);
        return;
      }}
      const card = cardSummaries().find(item => item.card_id === cardId);
      const holdings = workingHoldings.filter(item => item.card_id === cardId);
      document.getElementById("selectedCard").innerHTML = card ? `
        <img class="zoomable" src="${{card.image_url || ""}}" alt="${{card.name || "Card image"}}" data-large-src="${{card.image_url || ""}}">
        <div>
          <h2>${{card.name}}</h2>
          <p class="muted">${{card.set_name || "Unknown set"}} - ${{card.card_id}}</p>
          <p><strong>${{Number(card.copies_owned || 0)}} copies</strong></p>
          <p><strong>${{money(card.total_estimated_value)}} total estimated value</strong></p>
        </div>
      ` : "<p>No card holdings yet.</p>";
      renderVariants(holdings);
      bindZoomImages();
    }}

    function renderVariants(holdings) {{
      const rows = holdings.map(item => `
        <tr>
          <td><img class="zoomable" src="${{item.image_url || ""}}" alt="${{item.name || item.product_name || "Item image"}}" data-large-src="${{item.image_url || ""}}" style="max-width:42px"></td>
          <td>${{item.name || item.product_name || ""}}</td>
          <td>${{item.set_name || ""}}</td>
          <td>${{item.language || "English"}}</td>
          <td>${{item.market_segment}}</td>
          <td>${{item.condition || item.grading_company + " " + item.grade}}</td>
          <td><input type="number" min="0" step="1" value="${{Number(item.copies_owned || 0)}}" data-holding-id="${{item.holding_id}}" data-field="copies_owned"></td>
          <td><input type="number" min="0" step="0.01" value="${{Number(item.estimated_unit_value || 0).toFixed(2)}}" data-holding-id="${{item.holding_id}}" data-field="estimated_unit_value"></td>
          <td>${{money(item.modeled_fair_price || 0)}}</td>
          <td>${{money(item.total_estimated_value)}}</td>
          <td>${{item.pricing_label || ""}}</td>
          <td><button class="danger" data-remove-id="${{item.holding_id}}">Remove</button></td>
        </tr>
      `).join("");
      document.getElementById("variantTable").innerHTML = `
        <table>
          <thead><tr><th>Image</th><th>Name</th><th>Set</th><th>Language</th><th>Type</th><th>Variant</th><th>Copies</th><th>Each</th><th>Evaluated Price</th><th>Total</th><th>Perceived Value</th><th></th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>
      `;
      document.querySelectorAll("#variantTable input").forEach(input => {{
        input.addEventListener("change", () => updateHolding(input.dataset.holdingId, input.dataset.field, input.value));
      }});
      document.querySelectorAll("#variantTable button[data-remove-id]").forEach(button => {{
        button.addEventListener("click", () => removeHolding(button.dataset.removeId));
      }});
      bindZoomImages();
    }}

    function updateHolding(holdingId, field, value) {{
      const holding = workingHoldings.find(item => item.holding_id === holdingId);
      if (!holding) return;
      holding[field] = Number(value || 0);
      holding.total_estimated_value = Number(holding.copies_owned || 0) * Number(holding.estimated_unit_value || 0);
      renderStats();
      renderSelected(selectedCardId);
    }}

    function removeHolding(holdingId) {{
      const removed = workingHoldings.find(item => item.holding_id === holdingId);
      workingHoldings = workingHoldings.filter(item => item.holding_id !== holdingId);
      const stillHasSelected = selectedCardId && workingHoldings.some(item => item.card_id === selectedCardId);
      renderStats();
      renderSelected(stillHasSelected ? selectedCardId : "");
      if (removed) renderSearch();
    }}

    function variantLabel(item) {{
      return item.market_segment === "graded"
        ? `${{item.grading_company || ""}} ${{item.grade || ""}}`.trim()
        : item.condition || "raw";
    }}

    function selectedLanguage(cardId) {{
      return document.getElementById(`language-${{cardId}}`)?.value || "English";
    }}

    function standardVariantOptions() {{
      const raw = [
        ["raw_nm", "RAW - NM", "raw", "near_mint", "raw", 0],
        ["raw_lp", "RAW - LP", "raw", "lightly_played", "raw", 0],
        ["raw_mp", "RAW - MP", "raw", "moderately_played", "raw", 0],
        ["raw_hp", "RAW - HP", "raw", "heavily_played", "raw", 0],
        ["raw_dmg", "RAW - DMG", "raw", "damaged", "raw", 0],
      ];
      const psa = Array.from({{ length: 10 }}, (_, index) => {{
        const grade = index + 1;
        return [`psa_${{grade}}`, `PSA ${{grade}}`, "graded", "", "psa", grade];
      }});
      return raw.concat(psa).map(([code, label, market_segment, condition, grading_company, grade]) => ({{
        code, label, market_segment, condition, grading_company, grade
      }}));
    }}

    function selectedVariant(cardId) {{
      const code = document.getElementById(`variant-${{cardId}}`)?.value || "raw_nm";
      return standardVariantOptions().find(option => option.code === code) || standardVariantOptions()[0];
    }}

    function renderSearch() {{
      const term = document.getElementById("catalogSearch").value.toLowerCase().trim();
      const seen = new Set();
      const results = data.catalog.filter(item => {{
        const haystack = [item.card_id, item.name, item.set_name, item.rarity, item.pricing_label]
          .join(" ").toLowerCase();
        if (term && !haystack.includes(term)) return false;
        if (seen.has(item.card_id)) return false;
        seen.add(item.card_id);
        return true;
      }}).slice(0, 25);
      document.getElementById("searchResults").innerHTML = results.map(item => `
        <div class="search-card">
          <img class="zoomable" src="${{item.image_large || ""}}" alt="${{item.name || "Card image"}}" data-large-src="${{item.image_large || ""}}">
          <div>
            <strong>${{item.name}}</strong>
            <div class="muted">${{item.set_name || ""}} - ${{item.rarity || ""}}</div>
            <div class="small-controls">
              <select id="destination-${{item.card_id}}">
                <option value="portfolio">Portfolio</option>
                <option value="wishlist">Wishlist</option>
              </select>
              <select id="language-${{item.card_id}}">
                <option>English</option>
                <option>Japanese</option>
              </select>
              <select id="variant-${{item.card_id}}">
                ${{standardVariantOptions().map(variant => `
                  <option value="${{variant.code}}">${{variant.label}}</option>
                `).join("")}}
              </select>
              <button data-card-id="${{item.card_id}}">Add</button>
            </div>
          </div>
          <div class="right-stack">
            <span class="muted">Evaluated Price</span>
            <strong>${{money(item.modeled_fair_price || item.observed_price)}}</strong>
            <span class="muted">${{item.pricing_label || ""}}</span>
          </div>
        </div>
      `).join("") || `<div class="empty">No matching cards found.</div>`;
      document.querySelectorAll("#searchResults button").forEach(button => {{
        button.addEventListener("click", () => addCatalogItem(button.dataset.cardId));
      }});
      bindZoomImages();
    }}

    function addCatalogItem(cardId) {{
      const item = data.catalog.find(record => record.card_id === cardId);
      const variant = selectedVariant(cardId);
      const destination = document.getElementById(`destination-${{cardId}}`).value;
      const language = selectedLanguage(cardId);
      const copies = 1;
      const unitValue = Number(item.modeled_fair_price || item.observed_price || 0);
      const common = {{
        item_type: "card",
        card_id: item.card_id,
        language,
        tcg_price_type: item.tcg_price_type || "",
        market_segment: variant.market_segment,
        condition: variant.condition,
        grading_company: variant.grading_company,
        grade: variant.grade,
        product_name: item.name,
        notes: "Added from dashboard",
      }};
      if (destination === "wishlist") {{
        workingWishlist.push({{
          wishlist_id: `wish-${{Date.now()}}`,
          ...common,
          target_price: "",
          priority: "medium",
          name: item.name,
          set_name: item.set_name,
          rarity: item.rarity,
          image_url: item.image_large || "",
          modeled_fair_price: item.modeled_fair_price || "",
          observed_price: item.observed_price || "",
          pricing_label: item.pricing_label || "",
        }});
        renderWishlist();
        showTab("wishlist");
        return;
      }}
      workingHoldings.push({{
        holding_id: `web-${{Date.now()}}`,
        ...common,
        name: item.name,
        set_name: item.set_name,
        rarity: item.rarity,
        image_url: item.image_large || "",
        copies_owned: copies,
        purchase_price_each: "",
        estimated_unit_value: unitValue,
        modeled_fair_price: Number(item.modeled_fair_price || 0),
        total_estimated_value: unitValue * copies,
        acquired_date: "",
        pricing_label: item.pricing_label || "",
        source_name: "dashboard"
      }});
      renderStats();
      renderSelected(item.card_id);
      showTab("portfolio");
    }}

    function renderCharacterPremium() {{
      const premiums = data.character_premiums || [];
      const columns = Math.ceil(premiums.length / 2);
      const groups = [premiums.slice(0, columns), premiums.slice(columns)];
      const block = rows => `
        <div>
          <div class="premium-row premium-head">
            <span></span><span>Character</span><span>Rank</span><span>Score</span><span>Avg Price Rank</span><span>Prints</span>
          </div>
          ${{rows.map(item => `
            <div class="premium-row">
              <img src="${{item.image_url || ""}}" alt="${{item.character}}">
              <strong>${{item.character}}</strong>
              <span>${{item.print_count_weighted_rank}}</span>
              <span>${{item.normalized_score}}</span>
              <span>${{item.avg_set_rarity_price_rank}}</span>
              <span>${{item.print_count}}</span>
            </div>
          `).join("")}}
        </div>
      `;
      document.getElementById("characterPremium").innerHTML = premiums.length
        ? `<div class="premium-grid">${{groups.map(block).join("")}}</div>`
        : `<div class="empty">No character premium data yet. Score a larger card catalog first.</div>`;
    }}

    function showTab(tabName) {{
      document.querySelectorAll(".tab-button").forEach(button => {{
        button.classList.toggle("active", button.dataset.tab === tabName);
      }});
      document.querySelectorAll(".tab-section").forEach(section => {{
        section.classList.toggle("active", section.id === `tab-${{tabName}}`);
      }});
    }}

    function csvEscape(value) {{
      const text = String(value ?? "");
      return /[",\\n]/.test(text) ? `"${{text.replaceAll('"', '""')}}"` : text;
    }}

    function exportPortfolioCsv() {{
      const rows = [portfolioColumns.join(",")].concat(workingHoldings.map(item =>
        portfolioColumns.map(column => csvEscape(item[column])).join(",")
      ));
      const blob = new Blob([rows.join("\\n")], {{ type: "text/csv" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "portfolio.csv";
      link.click();
      URL.revokeObjectURL(url);
    }}

    function exportWishlistCsv() {{
      const rows = [wishlistColumns.join(",")].concat(workingWishlist.map(item =>
        wishlistColumns.map(column => csvEscape(item[column])).join(",")
      ));
      const blob = new Blob([rows.join("\\n")], {{ type: "text/csv" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "wishlist.csv";
      link.click();
      URL.revokeObjectURL(url);
    }}

    function removeWishlistItem(wishlistId) {{
      workingWishlist = workingWishlist.filter(item => item.wishlist_id !== wishlistId);
      renderWishlist();
    }}

    function renderWishlist() {{
      const rows = workingWishlist.map(item => `
        <tr>
          <td><img class="zoomable" src="${{item.image_url || ""}}" data-large-src="${{item.image_url || ""}}" alt="${{item.name || item.product_name || "Wishlist item"}}" style="max-width:42px"></td>
          <td>${{item.name || item.product_name || ""}}</td>
          <td>${{item.set_name || ""}}</td>
          <td>${{item.language || "English"}}</td>
          <td>${{variantLabel(item)}}</td>
          <td>${{money(item.modeled_fair_price || item.observed_price || 0)}}</td>
          <td>${{item.pricing_label || ""}}</td>
          <td><button class="danger" data-wishlist-remove-id="${{item.wishlist_id}}">Remove</button></td>
        </tr>
      `).join("");
      document.getElementById("wishlistTable").innerHTML = rows
        ? `<table><thead><tr><th>Image</th><th>Name</th><th>Set</th><th>Language</th><th>Variant</th><th>Evaluated Price</th><th>Perceived Value</th><th></th></tr></thead><tbody>${{rows}}</tbody></table>`
        : `<div class="empty">No wishlist cards yet. Add cards from the Add Cards tab.</div>`;
      document.querySelectorAll("#wishlistTable button[data-wishlist-remove-id]").forEach(button => {{
        button.addEventListener("click", () => removeWishlistItem(button.dataset.wishlistRemoveId));
      }});
      bindZoomImages();
    }}

    function renderRecommendations() {{
      const cards = data.recommendations.slice(0, 12).map(item => `
        <div class="card-row">
          <img class="zoomable" src="${{item.image_large || item.image_small || ""}}" data-large-src="${{item.image_large || item.image_small || ""}}" alt="${{item.name || "Card image"}}">
          <div><strong>${{item.name}}</strong><div class="muted">${{item.set_name || ""}} - ${{item.rarity || ""}} - ${{variantLabel(item)}}</div></div>
          <div class="right-stack">
            <span class="muted">Evaluated Price</span>
            <strong>${{money(item.modeled_fair_price || item.observed_price)}}</strong>
            <span class="muted">${{item.pricing_label || ""}}</span>
          </div>
        </div>
      `).join("");
      document.getElementById("recommendations").innerHTML = cards || "<p>No recommendations yet.</p>";
      bindZoomImages();
    }}

    function renderHistory() {{
      const rows = data.history.map(item => `
        <tr><td>${{item.snapshot_date}}</td><td>${{money(item.portfolio_value)}}</td></tr>
      `).join("");
      document.getElementById("history").innerHTML = rows ? `
        <table><thead><tr><th>Date</th><th>Total Value</th></tr></thead><tbody>${{rows}}</tbody></table>
      ` : "<p>No snapshots yet.</p>";
    }}

    function bindZoomImages() {{
      document.querySelectorAll("img.zoomable").forEach(image => {{
        image.onclick = event => {{
          event.stopPropagation();
          const source = image.dataset.largeSrc || image.src;
          if (!source) return;
          document.getElementById("modalImage").src = source;
          document.getElementById("imageModal").classList.add("active");
        }};
      }});
    }}

    renderStats();
    renderOwnedList();
    renderSelected("");
    renderRecommendations();
    renderWishlist();
    renderHistory();
    renderSearch();
    renderCharacterPremium();
    document.getElementById("catalogSearch").addEventListener("input", renderSearch);
    document.getElementById("exportPortfolio").addEventListener("click", exportPortfolioCsv);
    document.getElementById("exportWishlist").addEventListener("click", exportWishlistCsv);
    document.getElementById("imageModal").addEventListener("click", () => {{
      document.getElementById("imageModal").classList.remove("active");
    }});
    document.querySelectorAll(".tab-button").forEach(button => {{
      button.addEventListener("click", () => showTab(button.dataset.tab));
    }});
  </script>
</body>
</html>
"""
