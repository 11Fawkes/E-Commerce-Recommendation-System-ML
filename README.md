# E-Commerce Recommendation System

**Personalised product recommendations using collaborative filtering, content-based filtering, and a hybrid approach.**

## The Problem

E-commerce platforms show generic product listings. Without personalisation, users see the same products regardless of their browsing and purchase history — leading to low conversion and poor discovery of relevant items.

## What This Does

Takes user interaction data (views, cart additions, purchases) and product metadata → builds three recommendation models → returns personalised product rankings for any search keyword.

**Three approaches combined:**

1. **Collaborative Filtering** — SVD matrix factorisation on user-item interactions. Finds users with similar behaviour and recommends what they bought.
2. **Content-Based Filtering** — TF-IDF on product attributes (brand, categories). Recommends products similar to what a user has interacted with.
3. **Hybrid Model** — blends collaborative and content-based scores (configurable 50/50 default) for the best of both approaches.

## Key Features

- Implicit rating system: purchase → 2, add-to-cart → 1, view → 0.5
- SVD with k=10 latent factors via SciPy sparse matrix decomposition
- TF-IDF cosine similarity across brand and 3 category levels
- 4 sort modes: price asc/desc, most bought, least bought
- Handles cold-start users via content-based fallback

## Tech Stack

Python · SciPy (SVD) · scikit-learn (TF-IDF, cosine similarity) · Pandas · NumPy

## Run Locally

```bash
pip install pandas numpy scipy scikit-learn
python "ML final code.py"
```

> Note: requires a Parquet file with user interaction data (user_id, product_id, event_type, brand, price, categories). Not bundled due to size.

## About

Built by Dhruv Kumar — Business Analyst, Berlin.
