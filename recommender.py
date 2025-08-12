# recommender.py
import os, pandas as pd
from typing import Optional, List, Dict

# --- make sure these imports are at top of recommender.py ---
import re, numpy as np, pandas as pd
tests = [
    ("yellow rust", "caneberry"),
    ("downy mildew", "caneberry"),
    ("black spot", "rose"),
    ("black rot", "grape"),
    ("brown spot", "rice"),
    ("apple scab", "apple"),
    ("yellow rust", "wheat"),          # sometimes listed as stripe rust
    ("leaf mold", "tomato"),           # if tomato exists in your crawl
]
# =========================
# Synonym dictionaries
# =========================

# Host synonyms – maps alternate names to a single canonical form
HOST_SYNONYMS = {
    # from your original code
    "bell": "bell pepper",
    "pepper": "bell pepper",
    "pepper bell": "bell pepper",
    "orange": "citrus",

    # additions for better coverage
    "caneberries": "caneberry",   # plural to singular
    "cane berry": "caneberry",
    "grapes": "grape",            # plural to singular
    "apples": "apple",            # plural to singular
}

# Disease synonyms – maps canonical disease names to lists of alternate spellings
DISEASE_SYNONYMS = {
    # from your original UCANR context
    # (none explicitly listed in your old code, but this is how to define them)

    # common aliasing in plant pathology
    "stripe rust": ["yellow rust", "wheat yellow rust"],  # wheat
    "black spot": ["rose black spot"],                    # rose
    "leaf mold": ["cladosporium leaf mold"],               # tomato

    # optional additional mappings if needed
    "anthracnose": ["watermelon anthracnose", "bean anthracnose"],
    "powdery mildew": ["oidium"],
}

# assumes you already have:
# - rec_base, passages, details loaded
# - norm_host, disease_candidates, _ensure_tfidf, _TFIDF, _X, _trunc
def canonicalize_term(term):
    if term is None:
        return None
    t = re.sub(r"[_\-]+", " ", str(term).strip().lower())
    return re.sub(r"\s+", " ", t).strip()

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
def norm_dis(d):
    """Normalize disease name by lowercasing, trimming, replacing underscores/hyphens."""
    if d is None or (isinstance(d, float) and pd.isna(d)):
        return None
    return canonicalize_term(d)

def normalize_disease_alias(name: str) -> str:
    n = norm_dis(name) or ""
    # Apply your synonym map if present
    for canon, alts in DISEASE_SYNONYMS.items():
        c = norm_dis(canon) or ""
        if n == c or n in [norm_dis(a) for a in alts]:
            return c
    return n

def disease_candidates(query: str, top_k: int = 6, strong: int = 90, weak: int = 80):
    n = normalize_disease_alias(query)
    base = {n, _strip_parens(n)}
    # Exact/paren-equal match
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

def recommend_for_disease(disease: str, host_hint: str | None = None, k: int = 1,
                          min_chars: int = 120, allow_other_hosts: bool = True):
    """Order: lenient dataset (strict host) → TF-IDF over passages → raw details."""
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
# Add two convenience wrappers:
def load():
    # returns nothing; just ensures TF-IDF is ready on first call
    # your globals (rec_base, passages, details) are already loaded by import
    return True

def recommend(disease: str, host: Optional[str] = None, k: int = 3) -> List[Dict]:
    return recommend_for_disease(disease, host_hint=host, k=k, allow_other_hosts=True)





