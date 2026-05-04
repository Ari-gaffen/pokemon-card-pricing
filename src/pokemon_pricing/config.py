from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
PORTFOLIO_DIR = PROJECT_ROOT / "portfolio"

FRANCHISE_LEADERS = {
    "Charizard",
    "Pikachu",
    "Mewtwo",
    "Mew",
    "Eevee",
    "Lugia",
    "Rayquaza",
    "Gengar",
    "Umbreon",
}

RARITY_ORDER = {
    "Common": 1,
    "Uncommon": 2,
    "Rare": 3,
    "Rare Holo": 4,
    "Rare Holo EX": 5,
    "Rare Holo GX": 5,
    "Rare Holo V": 5,
    "Rare Holo VMAX": 6,
    "Rare Holo VSTAR": 6,
    "Rare Ultra": 7,
    "Rare Secret": 8,
    "Rare Rainbow": 8,
    "Amazing Rare": 7,
    "Radiant Rare": 6,
    "Illustration Rare": 7,
    "Special Illustration Rare": 8,
    "Hyper Rare": 8,
}

RAW_CONDITION_SCORE = {
    "damaged": 1,
    "heavily_played": 2,
    "moderately_played": 3,
    "lightly_played": 4,
    "near_mint": 5,
}

GRADING_COMPANY_SCORE = {
    "raw": 0,
    "psa": 1,
    "bgs": 2,
    "cgc": 3,
    "sgc": 4,
}

VARIANT_KEY_COLUMNS = [
    "card_id",
    "language",
    "tcg_price_type",
    "market_segment",
    "condition",
    "grading_company",
    "grade",
]
