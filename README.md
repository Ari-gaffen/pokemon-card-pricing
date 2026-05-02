# Pokemon Card Pricing Model

This project builds an explainable pricing model for Pokemon TCG cards. The output for each card variant is:

- `under_priced`: market price is meaningfully below modeled fair value
- `well_priced`: market price is near modeled fair value
- `over_priced`: market price is meaningfully above modeled fair value

The first version uses a fair-value residual model. It predicts a card's expected market price from fundamentals, then compares the current market price to that expectation.

## Source Decisions

1. **Pokemon TCG API**
   - Best free starting source for card metadata, set data, rarity, images, TCGplayer prices, and Cardmarket prices.
   - Docs say the card object includes TCGplayer prices in USD and Cardmarket prices in EUR.
   - Source: https://docs.pokemontcg.io/api-reference/cards/card-object

2. **TCGplayer**
   - Strong US marketplace signal.
   - The Pokemon TCG API already exposes TCGplayer `low`, `mid`, `high`, `market`, and `directLow` fields when available.
   - Direct TCGplayer API access can be added later if you get API credentials.
   - Source: https://docs.tcgplayer.com/reference/pricing

3. **Cardmarket**
   - Strong EU marketplace signal.
   - Official Cardmarket API access is currently restricted for new applicants, but Pokemon TCG API exposes Cardmarket summary fields.
   - Source: https://help.cardmarket.com/de/cardmarket-api

4. **eBay sold listings**
   - Good demand/liquidity signal because it reflects completed transactions.
   - eBay Marketplace Insights can return sold history, but it is limited release and covers up to 90 days.
   - Source: https://www.edp.ebay.com/api-docs/buy/marketplace-insights/static/overview.html

5. **PriceCharting**
   - Useful optional source for graded and raw price points.
   - It requires a paid API subscription and has rate limits.
   - Source: https://www.pricecharting.com/api-documentation

## Character Premium Logic

Character premium is captured with three signals:

- `character_print_count`: how many cards in the dataset feature that Pokemon. This measures broad franchise exposure and supply of appearances.
- `character_avg_set_rarity_price_rank`: within each set and rarity, cards are ranked by price. A character whose cards repeatedly rank high has demonstrated demand.
- `is_franchise_leader`: starter premium for characters like Charizard and Pikachu. This is intentionally explicit so you can inspect or change it.

## Raw Condition And Graded Pricing

The model now prices **variants**, not just cards. A variant is a specific version of a card:

```text
base1-4 | holofoil | raw | near_mint
base1-4 | holofoil | raw | lightly_played
base1-4 | holofoil | graded | psa | 9
base1-4 | holofoil | graded | psa | 10
```

The Pokemon TCG API baseline row is treated as `raw near_mint` when no better condition-specific data is supplied. You can add richer observations with CSV enrichment files.

Raw condition template:

[data/templates/raw_condition_prices.csv](data/templates/raw_condition_prices.csv)

Graded template:

[data/templates/graded_prices.csv](data/templates/graded_prices.csv)

Required enrichment columns:

- `card_id`: Pokemon TCG API card id, such as `base1-4`
- `market_segment`: `raw` or `graded`
- `observed_price`: the current or recent sold price you want the model to judge
- `source_name`: where the price came from, such as `ebay_sold`, `pricecharting`, `psa`, or `manual_review`

Recommended enrichment columns:

- `tcg_price_type`: `normal`, `holofoil`, `reverseHolofoil`, `1stEditionHolofoil`, or `1stEditionNormal`
- `condition`: for raw cards, use `near_mint`, `lightly_played`, `moderately_played`, `heavily_played`, or `damaged`
- `grading_company`: for graded cards, use `psa`, `bgs`, `cgc`, or `sgc`
- `grade`: numeric grade, such as `8`, `9`, `9.5`, or `10`
- `recent_sold_count`: liquidity signal from sold listings
- `population_count`: scarcity signal for graded cards
- `sold_window_days`: usually `30`, `60`, or `90`
- `as_of_date`: date the observation was captured

Why CSV first? eBay sold-history access is limited release, Cardmarket direct API access is restricted for new applicants, and PriceCharting requires a paid subscription. CSV enrichment means the model is ready for those sources without blocking your first working version.

## Model Decision Logic

The model predicts `log1p(observed_price)`.

Why log price? Pokemon card prices are very skewed: many cards are cheap, a few are very expensive. Log price lets the model learn relative differences without letting one trophy card dominate the training.

The model uses `HistGradientBoostingRegressor` because:

- it handles nonlinear effects,
- it works well with tabular data,
- it is available in scikit-learn,
- it supports missing values in numeric inputs.

The model does **not** use TCGplayer `low`, `mid`, `high`, or `directLow` as predictors when judging TCGplayer market price. Those are too close to the target and would create leakage. Cardmarket fields are kept as optional cross-market signals because they come from a different market.

Raw and graded variant features are included:

- `is_graded`
- `condition_score`
- `grade_numeric`
- `grading_company_score`
- `recent_sold_count`
- `population_count`
- `sold_window_days`

After prediction:

```text
pricing_ratio = observed_price / modeled_fair_price
```

Labels:

- `under_priced`: ratio below `0.85`
- `well_priced`: ratio from `0.85` to `1.15`
- `over_priced`: ratio above `1.15`

The 15% band is a practical first threshold because card prices have marketplace spread, condition noise, shipping effects, and stale listings. You should tune it after reviewing results.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

An API key is optional for the Pokemon TCG API, but recommended for better rate limits.

Fetch cards:

```powershell
python -m pokemon_pricing fetch --series "Sword & Shield" --max-pages 2
```

Train:

```powershell
python -m pokemon_pricing train --input data/raw/cards.jsonl --raw-prices data/templates/raw_condition_prices.csv --graded-prices data/templates/graded_prices.csv
```

Score:

```powershell
python -m pokemon_pricing score --input data/raw/cards.jsonl --raw-prices data/templates/raw_condition_prices.csv --graded-prices data/templates/graded_prices.csv --output data/processed/scored_cards.csv
```

## Portfolio

Your portfolio is stored in CSV files under [portfolio](portfolio):

- [portfolio/portfolio.csv](portfolio/portfolio.csv): cards and sealed products you own
- [portfolio/wishlist.csv](portfolio/wishlist.csv): cards or sealed products you want
- [portfolio/portfolio_value_history.csv](portfolio/portfolio_value_history.csv): value snapshots over time

Create or refresh the empty templates:

```powershell
python -m pokemon_pricing init-portfolio
```

Portfolio rows include `copies_owned`, so multiple copies of the same variant are valued correctly.

Card holdings are matched to scored variants with:

```text
card_id + tcg_price_type + market_segment + condition + grading_company + grade
```

Sealed products use the `estimated_unit_value` that you enter in `portfolio.csv`. This keeps sealed products in the portfolio value immediately, while leaving room to add a sealed-product pricing source later.

Value the portfolio:

```powershell
python -m pokemon_pricing value-portfolio --scored-cards data/processed/scored_cards.csv
```

This creates [data/processed/portfolio_valued.csv](data/processed/portfolio_valued.csv), including:

- the variant owned,
- number of copies,
- image URL,
- estimated value per variant,
- total estimated value per holding,
- raw, graded, sealed, and full portfolio totals.

Track portfolio value over time:

```powershell
python -m pokemon_pricing snapshot-portfolio --scored-cards data/processed/scored_cards.csv
```

Generate recommendations based on your taste:

```powershell
python -m pokemon_pricing recommend --scored-cards data/processed/scored_cards.csv
```

The recommendation score is intentionally simple and inspectable. It looks at what you already own, then favors similar characters, sets, rarities, artists, and market segments. It also gives a small boost to cards currently labeled `under_priced` or `well_priced`.

Generate a clickable portfolio dashboard:

```powershell
python -m pokemon_pricing dashboard --scored-cards data/processed/scored_cards.csv
```

Open [data/processed/portfolio_dashboard.html](data/processed/portfolio_dashboard.html) in your browser. When no single card is selected, the top summary shows total estimated portfolio value split across raw cards, graded cards, and sealed product. When you select a card, it shows the variants owned, copies owned, picture, estimated value per variant, and total value for that card.

## Suggested Next Steps

1. Start with Pokemon TCG API for a working free baseline.
2. Add eBay sold-count and median-sold-price features if you can get Marketplace Insights access.
3. Add PriceCharting if graded cards matter to your use case.
4. Replace the residual thresholds with labels learned from your own buy/sell outcomes once you have them.
