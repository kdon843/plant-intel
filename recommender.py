# recommender.py — drop-in, lazy-loaded, Streamlit-safe

import os, re, json, numpy as np, pandas as pd

# =========================
# Paths (adjust if needed)
# =========================
LENIENT_PARQ = os.getenv("LENIENT_PARQ", "data/ucanr/parsed/nlp/ucanr_nlp_dataset_lenient.parquet")
LENIENT_CSV  = os.getenv("LENIENT_CSV",  "data/ucanr/parsed/nlp/ucanr_nlp_dataset_lenient.csv")
PASS_PARQ    = os.getenv("PASS_PARQ",    "data/ucanr/parsed/nlp/ucanr_nlp_passages.parquet")
PASS_CSV     = os.getenv("PASS_CSV",     "data/ucanr/parsed/nlp/ucanr_nlp_passages.csv")
DETAILS_PARQ = os.getenv("DETAILS_PARQ", "data/ucanr/parsed/ucanr_details.parquet")

# =========================
# Globals (lazy init)
# =========================
rec_base: pd.DataFrame | None = None
passages: pd.DataFrame | None = None
details:  pd.DataFrame | None = None
DISEASE_VOCAB: list[str] | None = None

_TFIDF = None
_X = None
_LOADED = False

# RapidFuzz (optional, falls back to difflib)
try:
    from rapidfuzz import fuzz, process
    _HAS_RF = True
except Exception:
    _HAS_RF = False
    fuzz = process = None  # type: ignore


# =========================
# Normalization helpers
# =========================
def canonicalize_term(term):
    if term is None:
        return None
    t = re.sub(r"[_\-]+", " ", str(term).strip().lower())
    return re.sub(r"\s+", " ", t).strip()

def _strip_parens(s: str) -> str:
    """Remove any text in parentheses from a string."""
    return re.sub(r"\s*\(.*?\)\s*", "", s or "").strip()

def _trunc(s, n=420):
    s = "" if pd.isna(s) else str(s)
    return s if len(s) <= n else s[:n].rstrip()+"…"

# Host & disease synonym maps
HOST_SYNONYMS = {
    # originals
    "bell": "bell pepper",
    "pepper": "bell pepper",
    "pepper bell": "bell pepper",
    "orange": "citrus",
    # additions
    "caneberries": "caneberry",
    "cane berry": "caneberry",
    "grapes": "grape",
    "apples": "apple",
}

DISEASE_SYNONYMS = {
    # common UCANR-style aliasing
    "stripe rust": ["yellow rust", "wheat yellow rust"],   # wheat
    "black spot": ["rose black spot"],                     # rose
    "leaf mold": ["cladosporium leaf mold"],               # tomato
    # optional broad ones
    "anthracnose": ["watermelon anthracnose", "bean anthracnose"],
    "powdery mildew": ["oidium"],
}

def norm_host(h):
    if h is None or (isinstance(h, float) and pd.isna(h)):
        return None
    h = canonicalize_term(h)
    return HOST_SYNONYMS.get(h, h)

def norm_dis(d):
    """Normalize disease name by lowercasing, trimming, replacing underscores/hyphens."""
    if d is None or (isinstance(d, float) and pd.isna(d)):
        return None
    return canonicalize_term(d)


# =========================
# Data loading (lazy)
# =========================
def _load_frame(parq_path, csv_path) -> pd.DataFrame | None:
    if parq_path and os.path.exists(parq_path):
        return pd.read_parquet(parq_path)
    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def load():
    """Load datasets once, add normalized columns, and build DISEASE_VOCAB."""
    global rec_base, passages, details, DISEASE_VOCAB, _LOADED

    if _LOADED:
        return True

    rb = _load_frame(LENIENT_PARQ, LENIENT_CSV)
    ps = _load_frame(PASS_PARQ, PASS_CSV)
    dt = pd.read_parquet(DETAILS_PARQ) if os.path.exists(DETAILS_PARQ) else None

    if rb is None and ps is None and dt is None:
        raise FileNotFoundError(
            "No recommender data found. Ensure lenient/passages/details files exist in data/ucanr/parsed/."
        )

    # Add normalized columns
    for df in (rb, ps, dt):
        if df is not None:
            if "host" in df.columns:
                df["_host_norm"] = df["host"].map(norm_host)
            if "disease" in df.columns:
                df["_dis_norm"]  = df["disease"].map(norm_dis)
            # details uses disease_common
            if "disease_common" in df.columns and "_dis_norm" not in df.columns:
                df["_dis_norm"]  = df["disease_common"].map(norm_dis)

    # Assign globals
    rec_base = rb if rb is not None else pd.DataFrame()
    passages = ps if ps is not None else pd.DataFrame()
    details  = dt if dt is not None else pd.DataFrame()

    # Build disease vocab from everything we actually have
    vocab = set()
    if not rec_base.empty and "_dis_norm" in rec_base:
        vocab |= set(rec_base["_dis_norm"].dropna().tolist())
    if not passages.empty and "_dis_norm" in passages:
        vocab |= set(passages["_dis_norm"].dropna().tolist())
    if details is not None and not details.empty and "_dis_norm" in details:
        vocab |= set(details["_dis_norm"].dropna().tolist())

    if not vocab:
        # Create at least an empty list to avoid NameErrors; searches will just return query itself.
        vocab = set()

    DISEASE_VOCAB = sorted(vocab)
    _LOADED = True
    return True

def _ensure_loaded():
    if not _LOADED:
        load()


# =========================
# Candidate generation
# =========================
def normalize_disease_alias(name: str) -> str:
    n = norm_dis(name) or ""
    # Apply your synonym map if present
    for canon, alts in DISEASE_SYNONYMS.items():
        c = norm_dis(canon) or ""
        if n == c or n in [norm_dis(a) for a in alts]:
            return c
    return n

def disease_candidates(query: str, top_k: int = 6, strong: int = 90, weak: int = 80):
    _ensure_loaded()
    n = normalize_disease_alias(query)
    base = {n, _strip_parens(n)}

    # Exact/paren-equal
    hits = {d for d in DISEASE_VOCAB if d in base or _strip_parens(d) in base or _strip_parens(n) == _strip_parens(d)}
    if hits:
        return sorted(hits)

    # Fuzzy
    if _HAS_RF and DISEASE_VOCAB:
        cand = process.extract(n, DISEASE_VOCAB, scorer=fuzz.token_set_ratio, limit=top_k)
        good = {d for d, score, _ in cand if score >= strong} or {d for d, score, _ in cand if score >= weak}
        if good:
            return sorted(good)
    else:
        import difflib
        scores = [(d, int(100 * difflib.SequenceMatcher(None, n, d).ratio())) for d in DISEASE_VOCAB]
        scores.sort(key=lambda x: x[1], reverse=True)
        good = {d for d, s in scores[:top_k] if s >= 85}
        if good:
            return sorted(good)

    return [n]


# =========================
# TF-IDF index (lazy)
# =========================
def _ensure_tfidf():
    _ensure_loaded()
    global _TFIDF, _X
    if _TFIDF is None or _X is None:
        if passages is None or passages.empty or "text" not in passages.columns:
            raise RuntimeError("Passages data (with 'text' column) is required for TF-IDF fallback.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = passages["text"].fillna("").astype(str).tolist()
        _TFIDF = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=150_000)
        _X = _TFIDF.fit_transform(texts)


# =========================
# Fallbacks
# =========================
def tfidf_fallback(disease, host_hint=None, top_k=1, allow_other_hosts=False):
    _ensure_tfidf()
    dns = disease_candidates(disease)
    hn  = norm_host(host_hint) if host_hint else None

    cand = passages[passages["_dis_norm"].isin(dns)]
    scope = "unscoped"
    matched_host = None

    if hn:
        h_cand = cand[cand["_host_norm"] == hn]
        if not h_cand.empty:
            cand = h_cand
            scope = "host-strict"
            matched_host = hn
        elif not allow_other_hosts:
            return []
        else:
            scope = "other-host"

    if cand.empty:
        return []

    from sklearn.metrics.pairwise import linear_kernel
    q = f"{(host_hint or '')} {disease} management control treatment".strip()
    qv = _TFIDF.transform([q])
    sims = linear_kernel(qv, _X[cand.index])[0]
    idx = np.argsort(-sims)[:top_k]
    picks = cand.iloc[idx]

    out = []
    for s, (_, r) in zip(sims[idx], picks.iterrows()):
        out.append({
            "host": r.get("host"),
            "matched_host": matched_host or r.get("_host_norm"),
            "disease": r.get("disease"),
            "matched_disease": r.get("_dis_norm"),
            "detail_url": r.get("detail_url"),
            "field": r.get("field"),
            "score": float(s),
            "management_snippet": _trunc(r.get("text","")),
            "source": "tfidf-fallback",
            "stage": f"tfidf ({scope})",
        })
    return out


def details_fallback(disease, host_hint=None, k=1, allow_other_hosts=False, min_chars=60):
    _ensure_loaded()
    if details is None or details.empty:
        return []
    dns = disease_candidates(disease)
    hn  = norm_host(host_hint) if host_hint else None
    cand = details[details["_dis_norm"].isin(dns)].copy()
    scope = "unscoped"
    matched_host = None

    if hn:
        h_cand = cand[cand["_host_norm"] == hn]
        if not h_cand.empty:
            cand = h_cand
            scope = "host-strict"
            matched_host = hn
        elif not allow_other_hosts:
            return []
        else:
            scope = "other-host"

    if cand.empty:
        return []

    def _score(r):
        mg = len(str(r.get("management") or "")); sy = len(str(r.get("symptoms") or ""))
        return mg*2 + sy

    cand["__score"] = cand.apply(_score, axis=1)
    picks = cand.sort_values("__score", ascending=False).head(k)

    out = []
    for _, r in picks.iterrows():
        mg = re.sub(r"\s+"," ", str(r.get("management","") or "")).strip()
        if len(mg) < min_chars:
            mg = re.sub(r"\s+"," ", str(r.get("symptoms","") or "")).strip()
        out.append({
            "host": r.get("host"),
            "matched_host": matched_host or r.get("_host_norm"),
            "disease": r.get("disease_common"),
            "matched_disease": r.get("_dis_norm"),
            "detail_url": r.get("detail_url"),
            "status": "raw-details",
            "management_snippet": _trunc(mg),
            "source": "details-fallback",
            "stage": f"details ({scope})",
        })
    return out


# =========================
# Main API
# =========================
def recommend_for_disease(disease: str, host_hint: str | None = None, k: int = 1,
                          min_chars: int = 120, allow_other_hosts: bool = True):
    """Order: lenient dataset (strict host) → TF-IDF over passages → raw details."""
    _ensure_loaded()
    dns = disease_candidates(disease)
    hn  = norm_host(host_hint) if host_hint else None

    # 1) lenient dataset
    rows = rec_base[rec_base["_dis_norm"].isin(dns)].copy()
    if hn:
        strict = rows[rows["_host_norm"] == hn]
        if not strict.empty:
            rows = strict
        elif not allow_other_hosts:
            rows = rows.iloc[0:0]  # force empty to try scoped fallbacks

    if not rows.empty:
        def _score(r):
            mg = len(str(r.get("management_text","") or ""))
            sy = len(str(r.get("symptoms_text","") or ""))
            bonus = 200 if (hn and r.get("_host_norm")==hn) else 0
            return mg*2 + sy + bonus

        rows["__score"] = rows.apply(_score, axis=1)
        picks = rows.sort_values("__score", ascending=False).head(k)

        out = []
        for _, r in picks.iterrows():
            mg = re.sub(r"\s+"," ", str(r.get("management_text","") or "")).strip()

            rec = {
                "host": r.get("host"),
                "matched_host": r.get("_host_norm"),
                "disease": r.get("disease"),
                "matched_disease": r.get("_dis_norm"),
                "detail_url": r.get("detail_url"),
                "status": r.get("status"),
                "management_snippet": _trunc(mg),
                "source": "lenient-dataset",
                "stage": "lenient (host-strict)" if (hn and r.get("_host_norm")==hn) else "lenient (other-host)",
            }

            if len(mg) < min_chars:
                fb = tfidf_fallback(disease, host_hint=host_hint, top_k=1, allow_other_hosts=allow_other_hosts)
                if fb:
                    rec.update({
                        "management_snippet": fb[0]["management_snippet"],
                        "detail_url": fb[0]["detail_url"] or rec["detail_url"],
                        "source": fb[0]["source"],
                        "stage": fb[0]["stage"],
                    })
            out.append(rec)
        return out

    # 2) TF-IDF scoped
    fb = tfidf_fallback(disease, host_hint=host_hint, top_k=k, allow_other_hosts=allow_other_hosts)
    if fb: return fb

    # 3) Raw details scoped
    dfb = details_fallback(disease, host_hint=host_hint, k=k, allow_other_hosts=allow_other_hosts)
    return dfb


# Convenience wrappers for your app
def load_model():
    """Back-compat alias for Streamlit cache."""
    return load()

def recommend(disease: str, host: str | None = None, k: int = 3):
    return recommend_for_disease(disease, host_hint=host, k=k, allow_other_hosts=True)


# =========================
# Optional QA harness
# =========================
def evaluate_tests(test_pairs, k=1):
    _ensure_loaded()
    results = []
    suggestions = {"disease_synonyms": {}, "host_synonyms": {}}
    misses_strict = 0

    for dis, host in test_pairs:
        strict = recommend_for_disease(dis, host_hint=host, k=k, allow_other_hosts=False)
        fallback = None
        if not strict:
            misses_strict += 1
            fallback = recommend_for_disease(dis, host_hint=host, k=k, allow_other_hosts=True)

        row0 = (strict or fallback or [{}])[0]
        row = {
            "query_host": host,
            "query_disease": dis,
            "strict_hit": bool(strict),
            "stage": row0.get("stage"),
            "matched_host": row0.get("matched_host"),
            "matched_disease": row0.get("matched_disease"),
            "source": row0.get("source"),
            "detail_url": row0.get("detail_url"),
        }
        results.append(row)

        if not strict and fallback:
            qh = norm_host(host); mh = row["matched_host"]
            if qh and mh and qh != mh:
                suggestions["host_synonyms"].setdefault(qh, set()).add(mh)
            qd = normalize_disease_alias(dis); md = row["matched_disease"]
            if qd and md and qd != md:
                suggestions["disease_synonyms"].setdefault(md, set()).add(qd)

    df = pd.DataFrame(results)
    print("=== TEST RESULTS ===")
    with pd.option_context('display.max_colwidth', None):
        print(df)
    print(f"\nStrict misses: {misses_strict} / {len(test_pairs)}")

    if suggestions["disease_synonyms"]:
        print("\n# === Suggested DISEASE_SYNONYMS additions ===")
        for canon, alts in suggestions["disease_synonyms"].items():
            print(f'DISEASE_SYNONYMS.setdefault("{canon}", []).extend({sorted(alts)})')
    if suggestions["host_synonyms"]:
        print("\n# === Suggested HOST_SYNONYMS additions ===")
        for src, targets in suggestions["host_synonyms"].items():
            print(f'# Consider mapping "{src}" → one of {sorted(targets)} in HOST_SYNONYMS')

    return {"results": results, "suggestions": suggestions}
