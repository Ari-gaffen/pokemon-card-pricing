import argparse
import json
from pathlib import Path

from pokemon_pricing.config import PROCESSED_DIR, RAW_DIR
from pokemon_pricing.data_sources import fetch_pokemon_tcg_cards, series_query
from pokemon_pricing.features import cards_to_variant_frame, load_cards_jsonl
from pokemon_pricing.model import score_cards, train_model
from pokemon_pricing.portfolio import (
    build_dashboard,
    init_portfolio_files,
    recommend_cards,
    snapshot_portfolio_value,
    summarize_portfolio,
    value_portfolio,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokemon card pricing model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch card data")
    fetch_parser.add_argument("--query", default=None)
    fetch_parser.add_argument("--series", default=None)
    fetch_parser.add_argument("--max-pages", type=int, default=1)
    fetch_parser.add_argument("--output", type=Path, default=RAW_DIR / "cards.jsonl")

    train_parser = subparsers.add_parser("train", help="Train fair-value model")
    train_parser.add_argument("--input", type=Path, default=RAW_DIR / "cards.jsonl")
    train_parser.add_argument("--raw-prices", type=Path, default=None)
    train_parser.add_argument("--graded-prices", type=Path, default=None)

    score_parser = subparsers.add_parser("score", help="Score cards as under/well/over-priced")
    score_parser.add_argument("--input", type=Path, default=RAW_DIR / "cards.jsonl")
    score_parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "scored_cards.csv")
    score_parser.add_argument("--raw-prices", type=Path, default=None)
    score_parser.add_argument("--graded-prices", type=Path, default=None)

    init_portfolio_parser = subparsers.add_parser(
        "init-portfolio", help="Create portfolio and wishlist CSV files"
    )
    init_portfolio_parser.add_argument("--portfolio-dir", type=Path, default=None)

    value_parser = subparsers.add_parser("value-portfolio", help="Value portfolio holdings")
    value_parser.add_argument("--scored-cards", type=Path, default=PROCESSED_DIR / "scored_cards.csv")
    value_parser.add_argument("--portfolio", type=Path, default=None)
    value_parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "portfolio_valued.csv")

    snapshot_parser = subparsers.add_parser(
        "snapshot-portfolio", help="Save today's portfolio value to history"
    )
    snapshot_parser.add_argument("--scored-cards", type=Path, default=PROCESSED_DIR / "scored_cards.csv")
    snapshot_parser.add_argument("--portfolio", type=Path, default=None)

    recommend_parser = subparsers.add_parser(
        "recommend", help="Recommend cards based on portfolio taste"
    )
    recommend_parser.add_argument("--scored-cards", type=Path, default=PROCESSED_DIR / "scored_cards.csv")
    recommend_parser.add_argument("--portfolio", type=Path, default=None)
    recommend_parser.add_argument("--wishlist", type=Path, default=None)
    recommend_parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "recommendations.csv")
    recommend_parser.add_argument("--limit", type=int, default=20)

    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Generate portfolio dashboard HTML"
    )
    dashboard_parser.add_argument("--scored-cards", type=Path, default=PROCESSED_DIR / "scored_cards.csv")
    dashboard_parser.add_argument("--portfolio", type=Path, default=None)
    dashboard_parser.add_argument("--wishlist", type=Path, default=None)
    dashboard_parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "portfolio_dashboard.html")

    args = parser.parse_args()

    if args.command == "fetch":
        if args.query and args.series:
            raise ValueError("Use either --query or --series, not both.")
        query = series_query(args.series) if args.series else args.query
        output = fetch_pokemon_tcg_cards(query, args.max_pages, output_path=args.output)
        print(f"Saved cards to {output}")
        return

    if args.command == "init-portfolio":
        paths = init_portfolio_files(args.portfolio_dir) if args.portfolio_dir else init_portfolio_files()
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return

    if args.command == "value-portfolio":
        portfolio_path = args.portfolio or Path("portfolio/portfolio.csv")
        valued = value_portfolio(args.scored_cards, portfolio_path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        valued.to_csv(args.output, index=False)
        print(json.dumps(summarize_portfolio(valued), indent=2))
        print(f"Saved valued portfolio to {args.output}")
        return

    if args.command == "snapshot-portfolio":
        portfolio_path = args.portfolio or Path("portfolio/portfolio.csv")
        valued = value_portfolio(args.scored_cards, portfolio_path)
        history = snapshot_portfolio_value(valued)
        print(history.tail(1).to_json(orient="records", indent=2))
        return

    if args.command == "recommend":
        portfolio_path = args.portfolio or Path("portfolio/portfolio.csv")
        wishlist_path = args.wishlist or Path("portfolio/wishlist.csv")
        recommendations = recommend_cards(
            args.scored_cards, portfolio_path, wishlist_path, args.limit
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(args.output, index=False)
        print(f"Saved recommendations to {args.output}")
        return

    if args.command == "dashboard":
        portfolio_path = args.portfolio or Path("portfolio/portfolio.csv")
        wishlist_path = args.wishlist or Path("portfolio/wishlist.csv")
        valued = value_portfolio(args.scored_cards, portfolio_path)
        recommendations = recommend_cards(args.scored_cards, portfolio_path, wishlist_path)
        catalog = None
        if args.scored_cards.exists():
            import pandas as pd

            catalog = pd.read_csv(args.scored_cards)
        output = build_dashboard(valued, recommendations, catalog=catalog, output_path=args.output)
        print(f"Saved portfolio dashboard to {output}")
        return

    cards = load_cards_jsonl(args.input)
    enrichment_paths = [
        path for path in [getattr(args, "raw_prices", None), getattr(args, "graded_prices", None)] if path
    ]
    frame = cards_to_variant_frame(cards, enrichment_paths)

    if args.command == "train":
        metrics = train_model(frame)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "score":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scored = score_cards(frame)
        scored.to_csv(args.output, index=False)
        print(f"Saved scored cards to {args.output}")
