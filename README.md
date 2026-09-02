# SIH26102 — MPLADS AI-Powered Fraud & Anomaly Detection

An AI-powered system to detect anomalies, fraud, and inefficiencies in
MPLAD Scheme implementation, built for Smart India Hackathon 2026
(Problem Statement SIH26102).

## Problem

MPLADS gives every Member of Parliament ~₹5 crore/year to fund local
development works. This scheme has a long history of misuse fake
project reports, incomplete works marked as done, cost inflation, and
delayed fund utilization. This project builds a pipeline that traces
each project from **recommendation → completion** and flags suspicious
patterns automatically.

## Project structure

```
data/raw/        original MPLADS CSV files (unmodified)
data/cleaned/    cleaned CSVs, produced by scripts/clean_and_explore.py
matching/        tiered AI matching system linking recommended -> completed works
matching/output/ matched_works.csv, tier2_possible_matches.csv, unmatched_works.csv
```

## Setup

1. Install Python 3.12 and pip.
2. Install dependencies:
   ```
   pip3 install -r requirements.txt --break-system-packages
   ```

## How to run (in order)

1. **Clean the raw data:**
   ```
   python3.12 scripts/clean_and_explore.py
   ```
   Reads `data/raw/*.csv`, writes cleaned files to `data/cleaned/`.

2. **Match recommended works to completed works:**
   ```
   python3.12 matching/match_works_tiered_v2.py
   ```
   Reads from `data/cleaned/`, writes results to `matching/output/`:
   - `matched_works.csv` — Tier 1 (Gold): high-confidence matches
   - `tier2_possible_matches.csv` — Tier 2 (Silver): relaxed-confidence matches, worth a manual glance
   - `unmatched_works.csv` — no confident match found in either tier

   Note: the first run takes several minutes (AI model encodes all
   work descriptions); an `embeddings_cache.npz` file is saved so
   future runs are much faster.

## Team

|Cipher|
