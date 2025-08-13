# recommender.py for Streamlit- ready recommender + NLP 
# - Lazy loads data (fast app start)
# - Reads from S3
# - SageMaker NLP endpoint for text->disease; fuzzy fallback if absent
# - Friendly errors for Streamlit

import os, re, io, json, typing as T
import numpy as np
import pandas as pd


LENIENT_PARQ = os.getenv("LENIENT_PARQ", "data/ucanr/parsed/nlp/ucanr_nlp_dataset_lenient.parquet")
LENIENT_CSV  = os.getenv("LENIENT_CSV",  "data/ucanr/parsed/nlp/ucanr_nlp_dataset_lenient.csv")
PASS_PARQ    = os.getenv("PASS_PARQ",    "data/ucanr/parsed/nlp/ucanr_nlp_passages.parquet")
PASS_CSV     = os.getenv("PASS_CSV",     "data/ucanr/parsed/nlp/ucanr_nlp_passages.csv")
DETAILS_PARQ = os.getenv("DETAILS_PARQ", "data/ucanr/parsed/ucanr_details.parquet")
DETAILS_CSV  = os.getenv("DETAILS_CSV",  "")  # optional CSV fallback for details

NLP_ENDPOINT = os.getenv("NLP_ENDPOINT", "").strip()
NLP_REGION = os.getenv("NLP_REGION", "").strip() or None
NLP_LABELS_JSON = os.getenv("NLP_LABELS_JSON", "artifacts/models/nlp_distilbert/labels.json")


# Globals 
rec_base: pd.DataFrame | None = None
passages: pd.DataFrame | None = None
details:  pd.DataFrame | None = None
DISEASE_VOCAB: list[str] | None = None

_TFIDF = None
_X = None
_LABEL_MAP: dict[str, str] | None = None
_LOADED = False

# RapidFuzz (falls back to difflib)
try:
    from rapidfuzz import fuzz, process
    _HAS_RF = True
except Exception:
    _HAS_RF = False
    fuzz = process = None  # type: ignore



# String utils / normalization
def canonicalize_term(term: T.Optional[T.Any]) -> T.Optional[str]:
    if term is None or (isinstance(term, float) and pd.isna(term)):
        return None
    t = re.sub(r"[_\-]+", " ", str(term).strip().lower())
    return re.sub(r"\s+", " ", t).strip()

def _strip_parens(s: str) -> str:
    """Remove any text in parentheses from a string."""
    return re.sub(r"\s*\(.*?\)\s*", "", s or "").strip()

def _trunc(s: T.Any, n=420) -> str:
    s = "" if pd.isna(s) else str(s)
    return s if len(s) <= n else s[:n].rstrip()+"…"

def humanize(name: str | None) -> str | None:
    if name is None: return None
    s = str(name).replace("___", " — ").replace("__", " — ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip().title()


# Synonyms (expand as needed)
HOST_SYNONYMS = {
    "bell": "bell pepper",
    "pepper": "bell pepper",
    "pepper bell": "bell pepper",
    "orange": "citrus",
    "caneberries": "caneberry",
    "cane berry": "caneberry",
    "grapes": "grape",
    "apples": "apple",
}

DISEASE_SYNONYMS = {
    "stripe rust": ["yellow rust", "wheat yellow rust"],
    "black spot": ["rose black spot"],
    "leaf mold": ["cladosporium leaf mold"],
    "anthracnose": ["watermelon anthracnose", "bean anthracnose"],
    "powdery mildew": ["oidium"],
}

def norm_host(h): return HOST_SYNONYMS.get(canonicalize_term(h), canonicalize_term(h))
def norm_dis(d):  return canonicalize_term(d)

def humanize(name: str | None) -> str | None:
    if name is None: return None
    s = str(name).replace("___", " — ").replace("__", " — ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip().title()

# I/O helpers (local + S3)
def _is_s3(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")

def _read_s3_bytes(s3_uri: str) -> io.BytesIO:
    # NOTE: Requires AWS credentials (role/instance profile/Env vars). Will raise if missing.
    import boto3
    bucket_key = s3_uri[5:]  # strip "s3://"
    bucket, key = bucket_key.split("/", 1)
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return io.BytesIO(obj["Body"].read())

def _read_any_df(path: str | None) -> pd.DataFrame | None:
    """Read DataFrame from local or S3 path. Supports parquet or CSV by extension."""
    if not path:
        return None
    try:
        if _is_s3(path):
            bio = _read_s3_bytes(path)
            if path.endswith(".parquet"):
                try:
                    bio.seek(0)
                    return pd.read_parquet(bio)
                except Exception:
                    bio.seek(0)
                    return pd.read_csv(bio)
            else:
                bio.seek(0)
                return pd.read_csv(bio)
        else:
            if not os.path.exists(path):
                return None
            if path.endswith(".parquet"):
                try:
                    return pd.read_parquet(path)
                except Exception:
                    # Fall back to CSV with same stem if available
                    alt = os.path.splitext(path)[0] + ".csv"
                    if os.path.exists(alt):
                        return pd.read_csv(alt)
                    raise
            else:
                return pd.read_csv(path)
    except Exception as e:
        # Keep it quiet for Streamlit, you’ll see a clear message if nothing loads at all.
        return None


def _load_frame(parq_path: str | None, csv_path: str | None) -> pd.DataFrame | None:
    df1 = _read_any_df(parq_path)
    if df1 is not None:
        return df1
    df2 = _read_any_df(csv_path)
    if df2 is not None:
        return df2
    return None


# Label map (for NLP endpoint class IDs)
def _load_label_map() -> dict[str, str]:
    global _LABEL_MAP
    if _LABEL_MAP is not None:
        return _LABEL_MAP

    lm: dict[str, str] = {}
    if NLP_LABELS_JSON:
        df = None
        # Try JSON file first
        try:
            if _is_s3(NLP_LABELS_JSON):
                bio = _read_s3_bytes(NLP_LABELS_JSON)
                lm = json.loads(bio.getvalue().decode("utf-8"))
            elif os.path.exists(NLP_LABELS_JSON):
                with open(NLP_LABELS_JSON, "r", encoding="utf-8") as f:
                    lm = json.load(f)
        except Exception:
            lm = {}

        # Also allow CSV with columns [id, label]
        if not lm:
            df = _read_any_df(NLP_LABELS_JSON)
            if isinstance(df, pd.DataFrame) and {"id","label"} <= set(df.columns):
                lm = {str(r["id"]): str(r["label"]) for _, r in df.iterrows()}

    _LABEL_MAP = lm
    return _LABEL_MAP



# Data loading (lazy)
def load() -> bool:
    """Load datasets once, add normalized columns, and build DISEASE_VOCAB."""
    global rec_base, passages, details, DISEASE_VOCAB, _LOADED

    if _LOADED:
        return True

    rb = _load_frame(LENIENT_PARQ, LENIENT_CSV)
    ps = _load_frame(PASS_PARQ, PASS_CSV)
    dt = _load_frame(DETAILS_PARQ, DETAILS_CSV)

    if rb is None and ps is None and (dt is None or dt.empty):
        raise FileNotFoundError(
            "No recommender data found. Set LENIENT_PARQ/CSV, PASS_PARQ/CSV, DETAILS_PARQ/CSV "
            "to local paths or s3:// URIs. Expected columns include 'host', 'disease', "
            "'management_text'/'symptoms_text' (lenient), and 'text' (passages)."
        )

    for df in (rb, ps, dt):
        if df is not None and not df.empty:
            if "host" in df.columns:
                df["_host_norm"] = df["host"].map(norm_host)
            if "disease" in df.columns:
                df["_dis_norm"]  = df["disease"].map(norm_dis)
            if "disease_common" in df.columns and "_dis_norm" not in df.columns:
                df["_dis_norm"]  = df["disease_common"].map(norm_dis)

    rec_base = rb if rb is not None else pd.DataFrame()
    passages = ps if ps is not None else pd.DataFrame()
    details  = dt if dt is not None else pd.DataFrame()

    vocab = set()
    if not rec_base.empty and "_dis_norm" in rec_base:
        vocab |= set(rec_base["_dis_norm"].dropna())
    if not passages.empty and "_dis_norm" in passages:
        vocab |= set(passages["_dis_norm"].dropna())
    if not details.empty and "_dis_norm" in details:
        vocab |= set(details["_dis_norm"].dropna())

    DISEASE_VOCAB = sorted(vocab)
    _LOADED = True
    return True


# Candidate generation
def normalize_disease_alias(name: str) -> str:
    n = norm_dis(name) or ""
    for canon, alts in DISEASE_SYNONYMS.items():
        c = norm_dis(canon) or ""
        if n == c or n in [norm_dis(a) for a in alts]:
            return c
    return n

def disease_candidates(query: str, top_k: int = 6, strong: int = 90, weak: int = 80) -> list[str]:
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



def load() -> bool:
    """Load datasets once, add normalized columns, and build DISEASE_VOCAB."""
    import pandas as _pd
    global rec_base, passages, details, DISEASE_VOCAB, _LOADED

    if _LOADED:
        return True

    rb = _load_frame(LENIENT_PARQ, LENIENT_CSV)     
    ps = _load_frame(PASS_PARQ, PASS_CSV)           
    dt = _load_frame(DETAILS_PARQ, DETAILS_CSV)     

    def _is_empty(df):
        return (df is None) or (isinstance(df, _pd.DataFrame) and df.empty)

    if _is_empty(rb) and _is_empty(ps) and _is_empty(dt):
        raise FileNotFoundError(
            "No recommender data found. Set LENIENT_PARQ/CSV, PASS_PARQ/CSV, DETAILS_PARQ/CSV "
            "to local paths or s3:// URIs."
        )

    # Normalize columns on each non-empty frame
    for df in (rb, ps, dt):
        if isinstance(df, _pd.DataFrame) and not df.empty:
            if "host" in df.columns:
                df["_host_norm"] = df["host"].map(norm_host)
            if "disease" in df.columns:
                df["_dis_norm"] = df["disease"].map(norm_dis)
            if ("disease_common" in df.columns) and ("_dis_norm" not in df.columns):
                df["_dis_norm"] = df["disease_common"].map(norm_dis)

    rec_base = rb if isinstance(rb, _pd.DataFrame) else _pd.DataFrame()
    passages = ps if isinstance(ps, _pd.DataFrame) else _pd.DataFrame()
    details  = dt if isinstance(dt, _pd.DataFrame) else _pd.DataFrame()

    # Build disease vocab from available frames
    vocab = set()
    if isinstance(rec_base, _pd.DataFrame) and (not rec_base.empty) and ("_dis_norm" in rec_base.columns):
        vocab |= set(rec_base["_dis_norm"].dropna())
    if isinstance(passages, _pd.DataFrame) and (not passages.empty) and ("_dis_norm" in passages.columns):
        vocab |= set(passages["_dis_norm"].dropna())
    if isinstance(details, _pd.DataFrame) and (not details.empty) and ("_dis_norm" in details.columns):
        vocab |= set(details["_dis_norm"].dropna())

    DISEASE_VOCAB = sorted(vocab) if vocab else []
    _LOADED = True
    return True

def _ensure_loaded():
    if not _LOADED:
        load()

def _ensure_tfidf():
    """Build TF-IDF index once. Return True if available, else False."""
    import pandas as _pd
    _ensure_loaded()
    global _TFIDF, _X
    if (_TFIDF is not None) and (_X is not None):
        return True
    if (not isinstance(passages, _pd.DataFrame)) or passages.empty or ("text" not in passages.columns):
        return False
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = passages["text"].fillna("").astype(str).tolist()
    _TFIDF = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=150_000)
    _X = _TFIDF.fit_transform(texts)
    return True

def tfidf_fallback(disease: str, host_hint: str | None = None, top_k: int = 1, allow_other_hosts: bool = False) -> list[dict]:
    import numpy as _np
    import pandas as _pd
    from sklearn.metrics.pairwise import linear_kernel

    if not _ensure_tfidf():
        return []

    dns = disease_candidates(disease)
    hn  = norm_host(host_hint) if host_hint else None

    cand = passages[passages["_dis_norm"].isin(dns)]
    scope = "unscoped"
    matched_host = None

    if hn is not None:
        h_cand = cand[cand["_host_norm"] == hn]
        if isinstance(h_cand, _pd.DataFrame) and (not h_cand.empty):
            cand = h_cand
            scope = "host-strict"
            matched_host = hn
        elif not allow_other_hosts:
            return []
        else:
            scope = "other-host"

    if (not isinstance(cand, _pd.DataFrame)) or cand.empty:
        return []

    q = f"{(host_hint or '')} {disease} management control treatment".strip()
    qv = _TFIDF.transform([q])
    sims = linear_kernel(qv, _X[cand.index])[0]
    idx = _np.argsort(-sims)[:top_k]
    picks = cand.iloc[idx]

    out: list[dict] = []
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

def details_fallback(disease: str, host_hint: str | None = None, k: int = 1, allow_other_hosts: bool = False, min_chars: int = 60) -> list[dict]:
    import pandas as _pd
    _ensure_loaded()

    if (not isinstance(details, _pd.DataFrame)) or details.empty:
        return []

    dns = disease_candidates(disease)
    hn  = norm_host(host_hint) if host_hint else None
    cand = details[details["_dis_norm"].isin(dns)].copy()
    scope = "unscoped"
    matched_host = None

    if hn is not None:
        h_cand = cand[cand["_host_norm"] == hn]
        if (isinstance(h_cand, _pd.DataFrame)) and (not h_cand.empty):
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

    out: list[dict] = []
    for _, r in picks.iterrows():
        mg = re.sub(r"\s+"," ", str(r.get("management","") or "")).strip()
        if len(mg) < min_chars:
            mg = re.sub(r"\s+"," ", str(r.get("symptoms","") or "")).strip()
        out.append({
            "host": r.get("host"),
            "matched_host": matched_host or r.get("_host_norm"),
            "disease": r.get("disease_common") or r.get("disease"),
            "matched_disease": r.get("_dis_norm"),
            "detail_url": r.get("detail_url"),
            "status": "raw-details",
            "management_snippet": _trunc(mg),
            "source": "details-fallback",
            "stage": f"details ({scope})",
        })
    return out




# Main API (disease -> recommendations)
def recommend_for_disease(disease: str, host_hint: str | None = None, k: int = 1,
                          min_chars: int = 120, allow_other_hosts: bool = True) -> list[dict]:
    """Order: lenient dataset (strict host when possible) → TF-IDF over passages → raw details."""
    _ensure_loaded()
    dns = disease_candidates(disease)
    hn  = norm_host(host_hint) if host_hint else None

    rows = rec_base[rec_base["_dis_norm"].isin(dns)].copy() if rec_base is not None else pd.DataFrame()
    if hn is not None and not rows.empty:
        strict = rows[rows["_host_norm"] == hn]
        if not strict.empty:
            rows = strict
        elif not allow_other_hosts:
            rows = rows.iloc[0:0]

    if not rows.empty:
        def _score(r):
            mg = len(str(r.get("management_text","") or "")); sy = len(str(r.get("symptoms_text","") or ""))
            bonus = 200 if (hn and r.get("_host_norm")==hn) else 0
            return mg*2 + sy + bonus

        rows["__score"] = rows.apply(_score, axis=1)
        picks = rows.sort_values("__score", ascending=False).head(k)

        out: list[dict] = []
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

    fb = tfidf_fallback(disease, host_hint=host_hint, top_k=k, allow_other_hosts=allow_other_hosts)
    if fb: return fb

    return details_fallback(disease, host_hint=host_hint, k=k, allow_other_hosts=allow_other_hosts)


def recommend(disease: str, host: str | None = None, k: int = 3) -> list[dict]:
    """Convenience wrapper for Streamlit."""
    return recommend_for_disease(disease, host_hint=host, k=k, allow_other_hosts=True)


# NLP: text -> disease (endpoint or fuzzy fallback)
def _classify_text_via_endpoint(text: str, top_k: int = 3) -> list[tuple[str, float]]:
    """Call a SageMaker inference endpoint for text -> label(s). Returns [(label, score), ...]."""
    import boto3
    rt = boto3.client("sagemaker-runtime", region_name=NLP_REGION) if NLP_REGION else boto3.client("sagemaker-runtime")
    if not NLP_ENDPOINT:
        raise RuntimeError("NLP_ENDPOINT is not set. Either set it or use fuzzy fallback.")
    rt = boto3.client("sagemaker-runtime")
    payload = json.dumps({"text": text})
    resp = rt.invoke_endpoint(EndpointName=NLP_ENDPOINT,
                              ContentType="application/json",
                              Body=payload)
    body = resp["Body"].read()
    data = json.loads(body.decode("utf-8"))

    # Heuristic parsers for common shapes:
    pairs: list[tuple[str, float]] = []
    label_map = _load_label_map()

    def _map_label(lbl: str) -> str:
        # Map "class_36" or "36" via label map if present
        if lbl in label_map:
            return label_map[lbl]
        m = re.match(r"class[_\-]?(\d+)$", lbl)
        if m and (m.group(1) in label_map):
            return label_map[m.group(1)]
        return lbl

    if isinstance(data, dict):
        # { "labels": [...], "scores": [...] }
        if "labels" in data and "scores" in data:
            pairs = [(_map_label(l), float(s)) for l, s in zip(data["labels"], data["scores"])]
        # { "label": "...", "score": 0.93 }
        elif "label" in data and "score" in data:
            pairs = [(_map_label(str(data["label"])), float(data["score"]))]
        # { "predictions": [{"label": "...", "score": ...}, ...] }
        elif "predictions" in data and isinstance(data["predictions"], list):
            pairs = [(_map_label(str(p.get("label"))), float(p.get("score", 0.0))) for p in data["predictions"] if "label" in p]
    elif isinstance(data, list):
        # [{"label": "...", "score": ...}, ...] or [["label", score], ...]
        if data and isinstance(data[0], dict) and "label" in data[0]:
            pairs = [(_map_label(str(p["label"])), float(p.get("score", 0.0))) for p in data]
        elif data and isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
            pairs = [(_map_label(str(p[0])), float(p[1])) for p in data]

    # Sort + trim
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:max(1, top_k)]
    return pairs

def _classify_text_fuzzy(text: str, top_k: int = 3) -> list[tuple[str, float]]:
    """Fallback: fuzzy match the input text against known diseases."""
    _ensure_loaded()
    if not DISEASE_VOCAB:
        return []
    q = canonicalize_term(text) or ""
    if _HAS_RF:
        cand = process.extract(q, DISEASE_VOCAB, scorer=fuzz.token_set_ratio, limit=top_k)
        return [(c[0], c[1] / 100.0) for c in cand]
    else:
        import difflib
        scores = [(d, difflib.SequenceMatcher(None, q, d).ratio()) for d in DISEASE_VOCAB]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

def predict_from_text(text: str, top_k: int = 3) -> list[tuple[str, float]]:
    """
    Public API: classify text into disease label(s).
    - If NLP_ENDPOINT is set and reachable, uses it.
    - Otherwise falls back to fuzzy match over DISEASE_VOCAB.
    Returns list of (label, score) pairs.
    """
    try:
        if NLP_ENDPOINT:
            return _classify_text_via_endpoint(text, top_k=top_k)
    except Exception:
        # Don’t crash the app if endpoint has issues—fall back gracefully.
        pass
    return _classify_text_fuzzy(text, top_k=top_k)

def recommend_from_text(text: str, host_hint: str | None = None, k: int = 1) -> list[dict]:
    """
    Classify free text → disease → recommendations.
    - Uses endpoint if configured; otherwise fuzzy.
    - host_hint is optional (e.g., 'rose', 'wheat').
    """
    pairs = predict_from_text(text, top_k=max(1, k))
    if not pairs:
        return []
    # Pick the best disease and recommend
    top_label = pairs[0][0]
    return recommend_for_disease(top_label, host_hint=host_hint, k=k)






