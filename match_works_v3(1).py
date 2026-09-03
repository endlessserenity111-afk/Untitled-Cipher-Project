"""
match_works_v3.py
══════════════════
Links completed MPLADS works back to their recommended source, ranked
across two confidence tiers.

  TIER 1 · GOLD    all of:
    · same (MP, IDA) group, completion date ≥ recommendation date
    · embedding similarity ≥ 0.90
    · word-overlap (Jaccard) ≥ 0.95
    · amount delta ≤ 50%
    · no location conflict, no category conflict, no quantity conflict
    · duplicate-description tie-break (±10% amount)

  TIER 2 · SILVER   relaxed fallback for Tier-1 misses:
    · similarity ≥ 0.80 · Jaccard ≥ 0.60 · amount delta ≤ 80%
    · location conflict still disqualifies

  Everything else → unmatched_works.csv.
"""

import os
import re
import time
import pandas as pd
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(os.cpu_count() or 4)
from sentence_transformers import SentenceTransformer, util

# ── I/O ────────────────────────────────────────────────────────────────────
REC_FILE             = 'cleaned_mplads_recommended_works.csv'
COMP_FILE            = 'cleaned_mplads_completed_works.csv'
MATCHED_OUT          = 'matched_works.csv'
TIER2_OUT            = 'tier2_possible_matches.csv'
UNMATCHED_OUT        = 'unmatched_works.csv'
CACHE_FILE           = 'embeddings_cache.npz'
MODEL_NAME           = 'all-MiniLM-L6-v2'

# ── Tier 1 · Gold ────────────────────────────────────────────────────────────
SIM_THRESHOLD        = 0.90
JACCARD_THRESHOLD    = 0.95
AMOUNT_MAX_DIFF_PCT  = 50.0
DUP_AMOUNT_DIFF_PCT  = 10.0   # tie-break when two candidates share a description

# ── Tier 2 · Silver (fallback only) ─────────────────────────────────────────
TIER2_SIM_THRESHOLD       = 0.80
TIER2_JACCARD_THRESHOLD   = 0.60
TIER2_AMOUNT_MAX_DIFF_PCT = 80.0

# ── Stop words — generic terms stripped before token comparison ─────────────
_STOP = {
    # Construction/infra verbs and nouns
    'CONSTRUCTION','CONSTR','REPAIR','RENOVATION','PROVIDING','DEVELOPMENT',
    'IMPROVEMENT','INSTALLATION','SUPPLY','BUILDING','UPGRADATION','WIDENING',
    'RELAYING','LAYING','EXTENSION','PROTECTION','RESTORATION','STRENGTHENING',
    'MAINTENANCE','ERECTION','SETTING','CREATION','ESTABLISHMENT',
    'ROAD','STREET','LANE','PATH','PATHWAY','DRAIN','DRAINAGE',
    'TOILET','SCHOOL','BRIDGE','WELL','WATER','TANK','PIPE',
    'HALL','GROUND','PARK','COMPOUND','WALL','GATE','COMMUNITY',
    'LIGHT','HIGH','MAST','TUBE','PUMP','HAND',
    'METER','KM','RCC','PCC','WBM','WORK','WORKS','NIRMAN','KARYA',
    # Prepositions / articles / conjunctions
    'THE','AND','OF','IN','AT','TO','BY','WITH','FOR','FROM','NEAR',
    'UNDER','OVER','UPPER','LOWER','INTO','THROUGH','BETWEEN',
    # Administrative labels
    'PUBLIC','PRIMARY','SECONDARY','GOVERNMENT','GOVT','DISTRICT',
    'BLOCK','STATE','CENTRAL','NATIONAL','MUNICIPAL','CORPORATION',
    'TEHSIL','TALUK','TALUKA','MANDAL','SUBDIVISION','DIVISION',
    'ZONE','CIRCLE','JUNCTION','MAIN','OLD','NEW','LINK',
    # Directional
    'EAST','WEST','NORTH','SOUTH','ADJACENT','SIDE',
    # Common nouns (not location-specific)
    'GHAR','GAHR','HOUSE','HOME','PLACE','AREA','LOCATION','SITE',
    # Common surnames / titles (not location-specific)
    'YADAV','SHARMA','KUMAR','SINGH','DEVI','SMT','SHRI','SRI',
    # State names
    'BIHAR','ODISHA','ORISSA','PUNJAB','HARYANA','GUJARAT','RAJASTHAN',
    'KARNATAKA','KERALA','TAMILNADU','ANDHRA','TELANGANA','MAHARASHTRA',
    'MADHYA','PRADESH','UTTAR','UTTARAKHAND','HIMACHAL','JHARKHAND',
    'CHHATTISGARH','ASSAM','MANIPUR','MEGHALAYA','NAGALAND','TRIPURA',
    'SIKKIM','MIZORAM','ARUNACHAL','GOA','DELHI','BENGAL',
    # Common district / city names (too broad)
    'MADURAI','INDORE','PATNA','RANCHI','LUCKNOW','KANPUR','VARANASI',
    'AGRA','MEERUT','BHOPAL','JABALPUR','GWALIOR','NAGPUR','PUNE',
    'NASHIK','SURAT','VADODARA','AHMEDABAD','JAIPUR','JODHPUR','UDAIPUR',
    'KOTA','HYDERABAD','WARANGAL','VISAKHAPATNAM','VIJAYAWADA','GUNTUR',
    'COIMBATORE','SALEM','TIRUNELVELI','VELLORE','KOCHI','KOZHIKODE',
    'MYSURU','HUBLI','MANGALURU','BELAGAVI','BHUBANESWAR','CUTTACK',
    'ROURKELA','GUWAHATI','DEHRADUN','HARIDWAR','SHIMLA','CHANDIGARH',
    'AMRITSAR','LUDHIANA','JAMMU','SRINAGAR','GAYA','BODH','MUZAFFARPUR',
    'GORAKHPUR','PRAYAGRAJ','ALLAHABAD','MORADABAD','KOLKATA','HOWRAH',
    # Electrical / lighting / unit specs (not location-specific)
    'WATT','NOS','LED','SOLAR','FITTING','FITTINGS','LITER','LITRE','LTRS',
    'TYRE','CAPACITY','SPEED','VOLT','WATT','KVA','KW','KWH',
    # Block / administrative area words
    'BARUIPUR','PURBA','PASCHIM','UTTARA','DAKSHIN','PURBACHAL',
    # Administrative wrapper words (appear in template, not location-specific)
    'PANCHAYAT','GRAM','CONSTITUENCY','UNION','TALUKA','UPSIC',
    'IMPLEMENTING','AGENCY','IMPLEMNETING','RECOMMENDED','PURPOSE','PURPOSES',
    'COMMUNITY','TANKER','DRINKING','MINING','MOTOR','RECHARGE',
}



def _tokens(text: str) -> set:
    """Split text into uppercase tokens, drop stop words and short/numeric ones."""
    if not isinstance(text, str):
        return set()
    raw = [t for t in re.split(r'[\s,./\(\)\[\]\-:;\\]+', text.upper()) if t]
    
    _prefixes = {'WARD', 'PLOT', 'PROPERTY', 'NO', 'SURVEY', 'SY', 'HOUSE'}
    
    result = set()
    for i in range(len(raw)):
        t = raw[i]
        if t.isnumeric() and i > 0 and raw[i-1] in _prefixes:
            result.add(f"{raw[i-1]}_{t}")
            continue
            
        if len(t) >= 3 and not t.isnumeric() and t not in _STOP:
            result.add(t)
            
    return result


def jaccard(desc_a: str, desc_b: str) -> float:
    """
    Word-level Jaccard similarity on meaningful (non-stop) tokens.
    Returns 0.0 if either side has no meaningful tokens (reject that pair).
    """
    ta = _tokens(desc_a)
    tb = _tokens(desc_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def has_conflict(desc_a: str, desc_b: str) -> bool:
    """
    Returns True when the two descriptions contain CONFLICTING location tokens —
    i.e. both have at least one meaningful proper-noun token (>= 5 chars) that
    does NOT appear anywhere in the other description as a substring.
    This catches false positives where Jaccard is high due to shared boilerplate
    (e.g. 'SEVNI GP' vs 'TIMBA GP' in the same Taluka Kamrej description).
    """
    ta = _tokens(desc_a)
    tb = _tokens(desc_b)

    # Tokens exclusive to each side, only considering words >= 5 chars
    excl_a = {t for t in (ta - tb) if len(t) >= 5}
    excl_b = {t for t in (tb - ta) if len(t) >= 5}

    if not excl_a or not excl_b:
        return False  # One side is a subset of the other — no conflict

    b_upper = desc_b.upper()
    a_upper = desc_a.upper()

    # If ANY exclusive token from A appears as substring in B → might be same place
    a_found_in_b = any(tok in b_upper for tok in excl_a)
    # If ANY exclusive token from B appears as substring in A → might be same place
    b_found_in_a = any(tok in a_upper for tok in excl_b)

    # Conflict: both sides have unique location-like words invisible to the other
    return not a_found_in_b and not b_found_in_a


def category_conflict(cat_a, cat_b) -> bool:
    """True if both categories are known but genuinely don't align."""
    a = str(cat_a).strip().upper() if pd.notna(cat_a) else ''
    b = str(cat_b).strip().upper() if pd.notna(cat_b) else ''
    if not a or not b:
        return False               # missing data → can't judge, don't punish
    return a != b and a not in b and b not in a


_QTY_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(MTR|METER|METRE|KM|FT|FEET|NOS|NO|SQM|SQFT|LTR|LITRE|KVA|KW|HP)\b'
)

def _quantities(text: str) -> set:
    """Pull out measured quantities like '500MTR' or '10NOS' from text."""
    if not isinstance(text, str):
        return set()
    return {f"{m.group(1)}{m.group(2)}" for m in _QTY_RE.finditer(text.upper())}


def quantity_conflict(desc_a: str, desc_b: str) -> bool:
    """True if both sides quote measurements, but none of them agree."""
    qa, qb = _quantities(desc_a), _quantities(desc_b)
    if not qa or not qb:
        return False
    return qa.isdisjoint(qb)


# ── Data loading ─────────────────────────────────────────────────────────────

def embed_text(row) -> str:
    """Text actually handed to the AI model — description alone is thin;
    prefixing the category grounds the embedding in *what kind* of work
    this is, not just how it's worded."""
    cat  = str(row['Category']).strip() if pd.notna(row['Category']) else ''
    desc = str(row['Work Description']).strip()
    return f"{cat}: {desc}" if cat else desc


def load_data():
    print("-> Loading datasets...")
    df_rec  = pd.read_csv(REC_FILE)
    df_comp = pd.read_csv(COMP_FILE)

    df_rec['Recommendation Date'] = pd.to_datetime(df_rec['Recommendation Date'], errors='coerce')
    df_comp['Completed Date']     = pd.to_datetime(df_comp['Completed Date'],     errors='coerce')

    for df in (df_rec, df_comp):
        df['mp_key']  = df['MP Name'].astype(str).str.strip().str.upper()
        df['ida_key'] = df['IDA'].astype(str).str.strip().str.upper()

    print(f"   Recommended : {len(df_rec):,} rows")
    print(f"   Completed   : {len(df_comp):,} rows")
    return df_rec, df_comp


# ── Embedding cache ──────────────────────────────────────────────────────────

def get_embeddings(unique_texts: list) -> dict:
    """Pull vectors from disk cache; encode only what's missing."""
    cache = {}
    if os.path.exists(CACHE_FILE):
        print(f"-> Loading embedding cache '{CACHE_FILE}'...")
        try:
            npz = np.load(CACHE_FILE, allow_pickle=True)
            for d, v in zip(npz['descriptions'], npz['matrix']):
                cache[str(d)] = v
            print(f"   {len(cache):,} vectors loaded ({os.path.getsize(CACHE_FILE)/(1024**2):.0f} MB)")
        except Exception as e:
            print(f"   Cache load failed: {e} — recomputing.")
            cache = {}

    missing = [d for d in unique_texts if d not in cache]
    if missing:
        print(f"-> Encoding {len(missing):,} new descriptions on CPU (this takes time once)...")
        model = SentenceTransformer(MODEL_NAME, device='cpu')
        t0 = time.time()
        embs = model.encode(missing, batch_size=128, convert_to_numpy=True,
                            device='cpu', show_progress_bar=True)
        for d, v in zip(missing, embs):
            cache[str(d)] = v.astype(np.float32)
        print(f"   Done in {time.time()-t0:.1f}s. Saving cache...")
        np.savez_compressed(CACHE_FILE,
                            descriptions=np.array(list(cache.keys()), dtype=object),
                            matrix=np.array(list(cache.values()), dtype=np.float32))
        print(f"   Saved ({os.path.getsize(CACHE_FILE)/(1024**2):.0f} MB)")

    return {d: torch.from_numpy(v) for d, v in cache.items()}


# ── Matching ───────────────────────────────────────────────────────────────────

def run_matching(df_rec, df_comp):
    # Index candidates by (mp_key, ida_key)
    print("\n-> Grouping by (MP Name, IDA)...")
    rec_idx = {}
    for _, r in df_rec.iterrows():
        rec_idx.setdefault((r['mp_key'], r['ida_key']), []).append(r)

    comp_idx = {}
    for _, c in df_comp.iterrows():
        comp_idx.setdefault((c['mp_key'], c['ida_key']), []).append(c)

# ── Matching ─────────────────────────────────────────────────────────────────

def run_matching(df_rec, df_comp):
    print("\n-> Grouping by (MP Name, IDA)...")
    rec_idx = {}
    for _, r in df_rec.iterrows():
        rec_idx.setdefault((r['mp_key'], r['ida_key']), []).append(r)

    comp_idx = {}
    for _, c in df_comp.iterrows():
        comp_idx.setdefault((c['mp_key'], c['ida_key']), []).append(c)

    # Only embed text for groups that actually have candidates on both sides
    relevant = set()
    for key, rows in comp_idx.items():
        if key in rec_idx:
            for r in rows + rec_idx[key]:
                relevant.add(embed_text(r))
    print(f"   Unique texts to embed: {len(relevant):,}")

    emb_map = get_embeddings(list(relevant))
    ZERO    = torch.zeros(384)

    print("\n-> Matching — Tier 1 strict, Tier 2 relaxed fallback...")
    matched, tier2, unmatched = [], [], []
    total    = len(df_comp)
    done     = 0
    t0       = time.time()
    last_rep = 0

    for key, comp_rows in comp_idx.items():
        cand_rows = rec_idx.get(key, [])

        if not cand_rows:
            for c in comp_rows:
                done += 1
                unmatched.append(_umatch(c, None, 'No candidate with same MP & IDA'))
                done, last_rep = _maybe_print(done, total, t0, last_rep, matched, unmatched)
            continue

        c_embs = torch.stack([emb_map.get(embed_text(r), ZERO) for r in comp_rows])
        r_embs = torch.stack([emb_map.get(embed_text(r), ZERO) for r in cand_rows])
        sim_mat = util.cos_sim(c_embs, r_embs)

        # completion can't predate its own recommendation
        r_dates = np.array([r['Recommendation Date'] for r in cand_rows])
        c_dates = np.array([c['Completed Date']       for c in comp_rows])
        date_ok = torch.tensor(
            (c_dates[:, None] >= r_dates[None, :]) &
            pd.notna(c_dates[:, None]) & pd.notna(r_dates[None, :])
        )
        sim_mat[~date_ok] = -1.0

        # Every pair clearing the Tier 2 floor lands in edges_t2; those that
        # ALSO clear the Tier 1 bar (plus the three conflict checks) get
        # promoted into edges_t1 below.
        edges_t1 = []
        edges_t2 = []
        for i, c_row in enumerate(comp_rows):
            c_desc  = str(c_row['Work Description']).strip()
            c_amt   = c_row['Final Amount (₹)']

            for j in range(len(cand_rows)):
                sim_val = sim_mat[i, j].item()
                if sim_val < TIER2_SIM_THRESHOLD:
                    continue   # too dissimilar even for the relaxed tier — skip entirely

                r_row  = cand_rows[j]
                r_desc = str(r_row['Work Description']).strip()
                r_amt  = r_row['Recommended Amount (₹)']

                jac = jaccard(c_desc, r_desc)
                if jac < TIER2_JACCARD_THRESHOLD:
                    continue

                if has_conflict(c_desc, r_desc):
                    continue   # conflicting location tokens rejects BOTH tiers

                if pd.notna(r_amt) and r_amt > 0 and pd.notna(c_amt):
                    amt_diff = abs(c_amt - r_amt) / r_amt * 100.0
                    if amt_diff > TIER2_AMOUNT_MAX_DIFF_PCT:
                        continue
                else:
                    amt_diff = np.nan

                edge = (sim_val, jac, -amt_diff if pd.notna(amt_diff) else 0, i, j)

                passes_tier1 = (
                    sim_val >= SIM_THRESHOLD and
                    jac >= JACCARD_THRESHOLD and
                    (pd.isna(amt_diff) or amt_diff <= AMOUNT_MAX_DIFF_PCT) and
                    not category_conflict(c_row['Category'], r_row['Category']) and
                    not quantity_conflict(c_desc, r_desc)
                )
                if passes_tier1:
                    same_desc_cands = [r for r in cand_rows if str(r['Work Description']).strip() == r_desc]
                    if len(same_desc_cands) > 1:
                        if pd.notna(r_amt) and r_amt > 0 and pd.notna(c_amt):
                            tight_diff = abs(c_amt - r_amt) / r_amt * 100.0
                            if tight_diff > DUP_AMOUNT_DIFF_PCT:
                                passes_tier1 = False  # loses the duplicate tie-break — Tier 2 instead

                if passes_tier1:
                    edges_t1.append(edge)
                else:
                    edges_t2.append(edge)

        # Tier 1 claims first, best score wins
        edges_t1.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        matched_c = set()
        matched_r = set()
        c_to_edge = {}
        tier_of_c = {}

        for edge in edges_t1:
            sim_val, jac, neg_amt_diff, i, j = edge
            if i in matched_c or j in matched_r:
                continue
            matched_c.add(i)
            matched_r.add(j)
            c_to_edge[i] = edge
            tier_of_c[i] = 1

        # Tier 2 mops up whatever Tier 1 left unclaimed. It may reuse a
        # recommended row Tier 1 already took (one proposal, several real
        # phases isn't unusual) but not one another Tier 2 match already has.
        edges_t2.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        matched_r_t2 = set()
        for edge in edges_t2:
            sim_val, jac, neg_amt_diff, i, j = edge
            if i in matched_c:
                continue
            if j in matched_r_t2:
                continue
            matched_c.add(i)
            matched_r_t2.add(j)
            c_to_edge[i] = edge
            tier_of_c[i] = 2

        # Write out this group's verdicts
        for i, c_row in enumerate(comp_rows):
            done += 1
            if i in c_to_edge:
                sim_val, jac, _, _i, j = c_to_edge[i]
                tier   = tier_of_c[i]
                r_row   = cand_rows[j]
                c_amt   = c_row['Final Amount (₹)']
                r_amt   = r_row['Recommended Amount (₹)']
                r_date  = r_row['Recommendation Date']
                c_date  = c_row['Completed Date']
                delay   = (c_date - r_date).days if pd.notna(c_date) and pd.notna(r_date) else np.nan
                if pd.notna(r_amt) and r_amt > 0 and pd.notna(c_amt):
                    overrun = round((c_amt - r_amt) / r_amt * 100.0, 2)
                else:
                    overrun = np.nan

                row_out = {
                    'Match Tier'                    : tier,   # 1 = Gold (strict), 2 = Silver (relaxed)
                    'Completed Work ID'             : c_row['Work ID'],
                    'Completed Work Description'    : c_row['Work Description'],
                    'Completed Category'            : c_row['Category'],
                    'Completed MP Name'             : c_row['MP Name'],
                    'Completed Constituency'        : c_row['Constituency'],
                    'Completed State'               : c_row['State'],
                    'Completed House'               : c_row['House'],
                    'Final Amount (₹)'             : c_amt,
                    'Completed Date'                : c_date.strftime('%Y-%m-%d') if pd.notna(c_date) else '',
                    'Completed Has Images'          : c_row['Has Images'],
                    'Completed Average Rating'      : c_row['Average Rating'],
                    'Completed IDA'                 : c_row['IDA'],
                    'Matched Recommended Work ID'   : r_row['Work ID'],
                    'Matched Work Description'      : r_row['Work Description'],
                    'Matched Category'              : r_row['Category'],
                    'Matched Recommended Amount (₹)': r_amt,
                    'Matched Recommendation Date'   : r_date.strftime('%Y-%m-%d') if pd.notna(r_date) else '',
                    'Matched Has Images'            : r_row['Has Images'],
                    'Matched IDA'                   : r_row['IDA'],
                    'Similarity Score'              : round(sim_val, 4),
                    'Jaccard Score'                 : round(jac, 4),
                    'Delay (Days)'                  : delay,
                    'Cost Overrun %'                : overrun,
                }
                if tier == 1:
                    matched.append(row_out)
                else:
                    tier2.append(row_out)
            else:
                best_j = torch.argmax(sim_mat[i]).item() if len(cand_rows) > 0 else None
                raw_sim = sim_mat[i, best_j].item() if best_j is not None else np.nan
                unmatched.append(_umatch(c_row, raw_sim, 'Did not pass Tier 1 or Tier 2 gates'))

            done, last_rep = _maybe_print(done, total, t0, last_rep, matched, unmatched)

    return pd.DataFrame(matched), pd.DataFrame(tier2), pd.DataFrame(unmatched)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _umatch(c_row, best_sim, reason):
    c_date = c_row['Completed Date']
    return {
        'Completed Work ID'         : c_row['Work ID'],
        'Completed Work Description': c_row['Work Description'],
        'Completed Category'        : c_row['Category'],
        'Completed MP Name'         : c_row['MP Name'],
        'Completed Constituency'    : c_row['Constituency'],
        'Completed State'           : c_row['State'],
        'Completed House'           : c_row['House'],
        'Final Amount (₹)'         : c_row['Final Amount (₹)'],
        'Completed Date'            : c_date.strftime('%Y-%m-%d') if pd.notna(c_date) else '',
        'Completed IDA'             : c_row['IDA'],
        'Best Similarity Score'     : round(float(best_sim), 4) if pd.notna(best_sim) and best_sim > -1 else np.nan,
        'Unmatched Reason'          : reason,
    }


def _maybe_print(done, total, t0, last_rep, matched, unmatched):
    if done - last_rep >= 1000 or done == total:
        print(f"   {done:,}/{total:,}  ({time.time()-t0:.0f}s)  "
              f"Matches: {len(matched):,}  Unmatched: {len(unmatched):,}")
        last_rep = done
    return done, last_rep


def print_summary(df_m, df_t2, df_u, total):
    """Console verdict — headline numbers, then a fresh random peek at Tier 1."""
    n, n2, u = len(df_m), len(df_t2), len(df_u)
    avg  = df_m['Similarity Score'].mean() if n else 0.0
    avg2 = df_t2['Similarity Score'].mean() if n2 else 0.0
    print("\n" + "="*85)
    print("  MATCH SUMMARY  ·  Tier1: sim≥0.90 Jac≥0.95 amt≤50%  |  Tier2: sim≥0.80 Jac≥0.60 amt≤80%")
    print("="*85)
    print(f"  Total completed works   : {total:,}")
    print(f"  Tier 1 · Gold           : {n:,}   ({n/total*100:.2f}%)   avg sim {avg:.4f}")
    print(f"  Tier 2 · Silver         : {n2:,}   ({n2/total*100:.2f}%)   avg sim {avg2:.4f}")
    print(f"  Unmatched               : {u:,}   ({u/total*100:.2f}%)")
    print(f"  Combined                : {n+n2:,}   ({(n+n2)/total*100:.2f}%)")
    print("="*85)

    if n > 0:
        print("\n  10 RANDOM TIER-1 MATCHES  (a fresh draw every run)")
        print("="*85)
        for idx, (_, row) in enumerate(df_m.sample(min(10, n)).iterrows(), 1):
            print(f"\n#{idx}  Score:{row['Similarity Score']:.4f}  "
                  f"Jaccard:{row['Jaccard Score']:.3f}  "
                  f"Delay:{row['Delay (Days)']} days  "
                  f"Overrun:{row['Cost Overrun %']:.1f}%")
            print(f"  COMP  (ID {row['Completed Work ID']}) [{row['Completed Category']}]: {row['Completed Work Description']}")
            print(f"  REC   (ID {row['Matched Recommended Work ID']}) [{row['Matched Category']}]: {row['Matched Work Description']}")
            print("-"*85)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df_rec, df_comp = load_data()
    df_m, df_t2, df_u = run_matching(df_rec, df_comp)

    print(f"\n-> {len(df_m):,} Tier 1 rows → '{MATCHED_OUT}'")
    df_m.to_csv(MATCHED_OUT, index=False)

    print(f"-> {len(df_t2):,} Tier 2 rows → '{TIER2_OUT}'")
    df_t2.to_csv(TIER2_OUT, index=False)

    print(f"-> {len(df_u):,} unmatched rows → '{UNMATCHED_OUT}'")
    df_u.to_csv(UNMATCHED_OUT, index=False)

    print_summary(df_m, df_t2, df_u, len(df_comp))
