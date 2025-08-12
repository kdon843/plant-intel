# models_service.py
import os, io, json
import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def get_runtime():
    region = st.secrets.get("AWS_REGION", os.getenv("AWS_REGION", "us-east-2"))
    ak = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
    sk = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
    tk = st.secrets.get("AWS_SESSION_TOKEN", os.getenv("AWS_SESSION_TOKEN"))
    sess = boto3.Session(region_name=region) if not (ak and sk) else boto3.Session(
        aws_access_key_id=ak, aws_secret_access_key=sk, aws_session_token=tk, region_name=region
    )
    if not sess.get_credentials():
        raise NoCredentialsError()
    return sess.client("sagemaker-runtime")

def _to_bytes(obj):
    if isinstance(obj, (bytes, bytearray)): return bytes(obj)
    if hasattr(obj, "read"):
        data = obj.read()
        try: obj.seek(0)
        except Exception: pass
        return data
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            buf = io.BytesIO(); obj.save(buf, format="PNG"); return buf.getvalue()
    except Exception:
        pass
    raise TypeError("Unsupported image input; pass bytes, file-like, or PIL.Image.")

def _parse_image_response(raw) -> tuple[str, float | None, str]:
    # raw can be bytes or str; try JSON first, then fallbacks
    s = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)

    # Try JSON
    try:
        obj = json.loads(s)
        # Common shapes:
        if isinstance(obj, dict):
            label = obj.get("label") or obj.get("class") or obj.get("prediction")
            score = obj.get("score") or obj.get("probability") or obj.get("confidence")
            rec   = obj.get("recommendation") or obj.get("rec") or obj.get("treatment") or ""
            # Sometimes {predictions:[{label, score, ...}]}
            if not label and isinstance(obj.get("predictions"), list) and obj["predictions"]:
                top = obj["predictions"][0]
                label = top.get("label") or top.get("class") or str(top)
                score = top.get("score") or top.get("probability") or top.get("confidence")
                rec   = top.get("recommendation") or rec
            return str(label), float(score) if score is not None else None, str(rec)
        if isinstance(obj, list) and obj:
            first = obj[0]
            if isinstance(first, dict):
                label = first.get("label") or first.get("class") or first.get("prediction") or str(first)
                score = first.get("score") or first.get("probability") or first.get("confidence")
                rec   = first.get("recommendation") or ""
                return str(label), float(score) if score is not None else None, str(rec)
            if isinstance(first, list) and len(first) >= 2:
                return str(first[0]), float(first[1]), ""
    except json.JSONDecodeError:
        pass

    # CSV-ish "label,score,rec"
    parts = [p.strip() for p in s.split(",")]
    if len(parts) >= 3:
        label, score, rec = parts[0], parts[1], ",".join(parts[2:]).strip()
        try: score = float(score)
        except Exception: score = None
        return label, score, rec

    # Last resort: return the whole string as a "message"
    return s[:64] or "unknown", None, ""
    

def predict_from_image(img, endpoint_name=None):
    rt = get_runtime()
    endpoint = endpoint_name or st.secrets.get("IMG_ENDPOINT", os.getenv("IMG_ENDPOINT"))
    if not endpoint:
        raise ValueError("IMG_ENDPOINT is not set.")
    body_bytes = _to_bytes(img)
    try:
        resp = rt.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/x-image",
            Body=body_bytes,
        )
        raw = resp["Body"].read()
        return _parse_image_response(raw)  # <- always (label, score, rec)
    except ClientError:
        raise




