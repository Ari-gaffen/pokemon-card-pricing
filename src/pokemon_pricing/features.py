import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pokemon_pricing.config import FRANCHISE_LEADERS, RARITY_ORDER
from pokemon_pricing.enrichment import append_price_variants, load_price_enrichment


def load_cards_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Card data file was not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cards = [json.loads(line) for line in handle if line.strip()]
    if not cards:
        raise ValueError(f"Card data file is empty: {path}. Run the fetch command successfully first.")
    return cards


def cards_to_frame(cards: list[dict[str, Any]]) -> pd.DataFrame:
    if not cards:
        raise ValueError("No cards were provided.")
    rows = [_flatten_card(card) for card in cards]
    frame = pd.DataFrame(rows)
    frame["character"] = canonical_species_names(frame)
    return add_character_premium_features(frame)


def cards_to_variant_frame(cards: list[dict[str, Any]], enrichment_paths: list[Path] | None) -> pd.DataFrame:
    card_frame = cards_to_frame(cards)
    enrichment = load_price_enrichment(enrichment_paths)
    return append_price_variants(card_frame, enrichment)


def _flatten_card(card: dict[str, Any]) -> dict[str, Any]:
    tcg_prices = _first_price_bucket(card.get("tcgplayer", {}).get("prices", {}))
    cardmarket_prices = card.get("cardmarket", {}).get("prices", {}) or {}
    set_info = card.get("set", {}) or {}

    return {
        "card_id": card.get("id"),
        "name": card.get("name"),
        "character": _character_name(card.get("name", "")),
        "language": "English",
        "supertype": card.get("supertype"),
        "rarity": card.get("rarity"),
        "rarity_score": RARITY_ORDER.get(card.get("rarity"), np.nan),
        "set_id": set_info.get("id"),
        "set_name": set_info.get("name"),
        "set_series": set_info.get("series"),
        "set_printed_total": set_info.get("printedTotal"),
        "set_total": set_info.get("total"),
        "release_date": set_info.get("releaseDate"),
        "artist": card.get("artist"),
        "number": card.get("number"),
        "image_small": (card.get("images", {}) or {}).get("small"),
        "image_large": (card.get("images", {}) or {}).get("large"),
        "national_pokedex_number": _first_value(card.get("nationalPokedexNumbers")),
        "tcg_price_type": tcg_prices.get("price_type"),
        "tcg_low": tcg_prices.get("low"),
        "tcg_mid": tcg_prices.get("mid"),
        "tcg_high": tcg_prices.get("high"),
        "tcg_market": tcg_prices.get("market"),
        "tcg_direct_low": tcg_prices.get("directLow"),
        "cardmarket_avg_sell": cardmarket_prices.get("averageSellPrice"),
        "cardmarket_trend": cardmarket_prices.get("trendPrice"),
        "cardmarket_avg_7": cardmarket_prices.get("avg7"),
        "cardmarket_avg_30": cardmarket_prices.get("avg30"),
    }


def add_character_premium_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["release_date"] = pd.to_datetime(frame["release_date"], errors="coerce")
    frame["card_age_days"] = (pd.Timestamp.today().normalize() - frame["release_date"]).dt.days
    frame["is_franchise_leader"] = frame["character"].isin(FRANCHISE_LEADERS).astype(int)

    print_counts = frame.groupby("character")["card_id"].transform("count")
    frame["character_print_count"] = print_counts

    rank_group = ["set_id", "rarity"]
    frame["set_rarity_price_rank"] = frame.groupby(rank_group)["tcg_market"].rank(
        method="average", ascending=False
    )
    frame["character_avg_set_rarity_price_rank"] = frame.groupby("character")[
        "set_rarity_price_rank"
    ].transform("mean")

    return frame


def canonical_species_names(frame: pd.DataFrame) -> pd.Series:
    """Roll special card names up to a likely Pokemon species name."""
    if "national_pokedex_number" not in frame.columns:
        return frame["character"]

    pokemon = frame[
        (frame.get("supertype") == "Pokémon")
        & pd.to_numeric(frame["national_pokedex_number"], errors="coerce").between(1, 1025)
    ].copy()
    if pokemon.empty:
        return frame["character"]

    pokemon["candidate_species"] = pokemon["name"].map(_character_name)
    species_by_number = (
        pokemon.sort_values(
            by=["national_pokedex_number", "candidate_species"],
            key=lambda column: column.astype(str).str.len()
            if column.name == "candidate_species"
            else column,
        )
        .groupby("national_pokedex_number")["candidate_species"]
        .first()
        .to_dict()
    )
    return frame.apply(
        lambda row: species_by_number.get(row.get("national_pokedex_number"), row.get("character")),
        axis=1,
    )


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    usable = frame.dropna(subset=["observed_price"]).copy()
    usable = usable[usable["observed_price"] > 0]
    usable["target_log_price"] = np.log1p(usable["observed_price"])
    return usable


def feature_columns() -> list[str]:
    return [
        "rarity_score",
        "set_printed_total",
        "set_total",
        "card_age_days",
        "national_pokedex_number",
        "cardmarket_avg_sell",
        "cardmarket_trend",
        "cardmarket_avg_7",
        "cardmarket_avg_30",
        "is_franchise_leader",
        "character_print_count",
        "character_avg_set_rarity_price_rank",
        "is_graded",
        "condition_score",
        "grade_numeric",
        "grading_company_score",
        "recent_sold_count",
        "population_count",
        "sold_window_days",
    ]


def _first_price_bucket(prices: dict[str, dict[str, float]]) -> dict[str, Any]:
    preferred = ["holofoil", "normal", "reverseHolofoil", "1stEditionHolofoil", "1stEditionNormal"]
    for price_type in preferred:
        if price_type in prices:
            return {"price_type": price_type, **prices[price_type]}
    if prices:
        price_type, values = next(iter(prices.items()))
        return {"price_type": price_type, **values}
    return {}


def _character_name(name: str) -> str:
    if "'s " in name:
        name = name.split("'s ", 1)[1]
    separators = [" VMAX", " VSTAR", "-EX", " ex", " GX", " V", " Lv.", " Star"]
    character = name
    for separator in separators:
        character = character.split(separator)[0]
    return character.strip()


def _first_value(values: list[Any] | None) -> Any:
    if not values:
        return None
    return values[0]
