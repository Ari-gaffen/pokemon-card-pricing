from pokemon_pricing.enrichment import append_price_variants, normalize_enrichment
from pokemon_pricing.features import cards_to_frame
from pokemon_pricing.portfolio import summarize_portfolio, value_portfolio
from pokemon_pricing.data_sources import fetch_pokemon_tcg_cards, series_query
import pandas as pd


def test_series_query_quotes_series_names():
    assert series_query("Sword & Shield") == 'set.series:"Sword & Shield"'


def test_fetch_accepts_unlimited_pages_signature(tmp_path, monkeypatch):
    class Response:
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": []}

    def fake_get(*args, **kwargs):
        return Response()

    monkeypatch.setattr("pokemon_pricing.data_sources.requests.get", fake_get)
    try:
        fetch_pokemon_tcg_cards(None, None, output_path=tmp_path / "cards.jsonl")
    except RuntimeError as error:
        assert "No cards were returned" in str(error)


def test_character_premium_features_are_created():
    cards = [
        {
            "id": "base1-4",
            "name": "Charizard",
            "rarity": "Rare Holo",
            "set": {"id": "base1", "name": "Base", "printedTotal": 102, "total": 102},
            "tcgplayer": {"prices": {"holofoil": {"market": 300.0, "low": 250.0, "mid": 300.0}}},
        },
        {
            "id": "base1-58",
            "name": "Pikachu",
            "rarity": "Common",
            "set": {"id": "base1", "name": "Base", "printedTotal": 102, "total": 102},
            "tcgplayer": {"prices": {"normal": {"market": 5.0, "low": 3.0, "mid": 5.0}}},
        },
    ]

    frame = cards_to_frame(cards)

    assert "character_print_count" in frame.columns
    assert "character_avg_set_rarity_price_rank" in frame.columns
    assert frame.loc[frame["name"] == "Charizard", "is_franchise_leader"].iloc[0] == 1


def test_raw_and_graded_variants_are_created():
    cards = [
        {
            "id": "base1-4",
            "name": "Charizard",
            "rarity": "Rare Holo",
            "set": {"id": "base1", "name": "Base", "printedTotal": 102, "total": 102},
            "tcgplayer": {"prices": {"holofoil": {"market": 300.0, "low": 250.0, "mid": 300.0}}},
        },
    ]
    card_frame = cards_to_frame(cards)
    enrichment = normalize_enrichment(
        pd.DataFrame(
            [
                {
                    "card_id": "base1-4",
                    "tcg_price_type": "holofoil",
                    "market_segment": "raw",
                    "condition": "lightly_played",
                    "observed_price": 225.0,
                    "source_name": "ebay_sold",
                },
                {
                    "card_id": "base1-4",
                    "tcg_price_type": "holofoil",
                    "market_segment": "graded",
                    "grading_company": "psa",
                    "grade": 9,
                    "observed_price": 1500.0,
                    "source_name": "pricecharting",
                    "population_count": 7500,
                },
            ]
        )
    )

    variants = append_price_variants(card_frame, enrichment)

    assert len(variants) == 3
    assert variants["observed_price"].notna().all()
    assert variants.loc[variants["market_segment"] == "graded", "is_graded"].iloc[0] == 1
    assert variants.loc[variants["condition"] == "lightly_played", "condition_score"].iloc[0] == 4


def test_portfolio_values_copies_and_segments(tmp_path):
    scored_path = tmp_path / "scored_cards.csv"
    portfolio_path = tmp_path / "portfolio.csv"
    pd.DataFrame(
        [
            {
                "card_id": "base1-4",
                "name": "Charizard",
                "image_large": "https://example.com/charizard.png",
                "tcg_price_type": "holofoil",
                "market_segment": "raw",
                "condition": "near_mint",
                "grading_company": "raw",
                "grade": 0,
                "modeled_fair_price": 300.0,
                "observed_price": 310.0,
                "pricing_label": "well_priced",
                "source_name": "test",
            },
            {
                "card_id": "base1-4",
                "name": "Charizard",
                "image_large": "https://example.com/charizard.png",
                "tcg_price_type": "holofoil",
                "market_segment": "graded",
                "condition": "",
                "grading_company": "psa",
                "grade": 9,
                "modeled_fair_price": 1500.0,
                "observed_price": 1450.0,
                "pricing_label": "well_priced",
                "source_name": "test",
            },
        ]
    ).to_csv(scored_path, index=False)
    pd.DataFrame(
        [
            {
                "holding_id": "h001",
                "item_type": "card",
                "card_id": "base1-4",
                "tcg_price_type": "holofoil",
                "market_segment": "raw",
                "condition": "near_mint",
                "grading_company": "raw",
                "grade": 0,
                "copies_owned": 2,
            },
            {
                "holding_id": "h002",
                "item_type": "card",
                "card_id": "base1-4",
                "tcg_price_type": "holofoil",
                "market_segment": "graded",
                "condition": "",
                "grading_company": "psa",
                "grade": 9,
                "copies_owned": 1,
            },
            {
                "holding_id": "s001",
                "item_type": "sealed",
                "product_name": "Booster Box",
                "market_segment": "sealed",
                "copies_owned": 3,
                "estimated_unit_value": 120.0,
            },
        ]
    ).to_csv(portfolio_path, index=False)

    valued = value_portfolio(scored_path, portfolio_path)
    summary = summarize_portfolio(valued)

    assert summary["raw_value"] == 600.0
    assert summary["graded_value"] == 1500.0
    assert summary["sealed_value"] == 360.0
    assert summary["portfolio_value"] == 2460.0
