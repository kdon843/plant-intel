# inference.py
# Minimal Recommendations endpoint for SageMaker
# Maps (host, disease) -> top management bullets from a prebuilt parquet file.
import json, os, re, pandas as pd

def _clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()

def model_fn(model_dir):
    # Load the packaged parquet
    path = os.path.join(model_dir, "ucanr_nlp_dataset.parquet")
    df = pd.read_parquet(path)
    # Normalize keys
    df["_host"] = df["host"].astype(str).str.lower().str.strip()
    df["_dis"]  = df["disease"].astype(str).str.lower().str.strip()
    # Build in-memory index for fast lookup
    index = {}
    for rec in df.to_dict(orient="records"):
        key = (rec["_host"], rec["_dis"])
        if key not in index:
            index[key] = rec
    return {"index": index}

def input_fn(body, content_type):
    return json.loads(body)

def _bullets(txt):
    parts = re.split(r"[.;]\s+", _clean(txt))
    # Keep a handful of concise bullets
    return [p for p in parts if p][:8]

def predict_fn(payload, model):
    host = _clean(payload.get("host", "")).lower()
    disease = _clean(payload.get("disease", "")).lower()
    if not host or not disease:
        return {"error": "bad_request", "message": "Expected fields: 'host' and 'disease'."}
    rec = model["index"].get((host, disease))
    if not rec:
        return {"error": "not_found", "message": f"No match for host='{host}', disease='{disease}'."}
    return {
        "host": rec.get("host"),
        "disease": rec.get("disease"),
        "detail_url": rec.get("detail_url"),
        "top_recommendations": _bullets(rec.get("management_text", "")),
        "source": "UCANR"
    }

def output_fn(prediction, accept):
    return json.dumps(prediction)

