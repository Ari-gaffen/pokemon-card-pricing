import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from requests import HTTPError
from dotenv import load_dotenv

from pokemon_pricing.config import RAW_DIR

API_URL = "https://api.pokemontcg.io/v2/cards"


def series_query(series: str) -> str:
    escaped = series.replace("\\", "\\\\").replace('"', '\\"')
    return f'set.series:"{escaped}"'


def fetch_pokemon_tcg_cards(
    query: str | None,
    max_pages: int | None,
    page_size: int = 250,
    output_path: Path = RAW_DIR / "cards.jsonl",
) -> Path:
    """Fetch cards from the Pokemon TCG API and save JSON Lines."""
    load_dotenv()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    api_key = os.getenv("POKEMON_TCG_API_KEY")
    if api_key:
        headers["X-Api-Key"] = api_key

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_cards = 0

    with temp_path.open("w", encoding="utf-8") as handle:
        page = 1
        while max_pages is None or page <= max_pages:
            params: dict[str, Any] = {"page": page, "pageSize": page_size}
            if query:
                params["q"] = query

            response = requests.get(API_URL, headers=headers, params=params, timeout=30)
            try:
                response.raise_for_status()
            except HTTPError as error:
                detail = response.text.strip()
                message = f"Pokemon TCG API request failed for query {query!r}: {detail}"
                raise RuntimeError(message) from error
            payload = response.json()
            cards = payload.get("data", [])
            total_cards += len(cards)

            for card in cards:
                handle.write(json.dumps(card) + "\n")

            if len(cards) < page_size:
                break

            page += 1
            time.sleep(0.25)

    if total_cards == 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"No cards were returned for query {query!r}.")

    temp_path.replace(output_path)
    return output_path
