# models_service.py
import os, io, json
import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def get_runtime():
    # Prefer Streamlit secrets; fall back to env vars
    ak = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
    sk = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
    tk = st.secrets.get("AWS_SESSION_TOKEN", os.getenv("AWS_SESSION_TOKEN"))  # optional
    region = st.secrets.get("AWS_REGION", os.getenv("AWS_REGION", "us-east-1"))

    if not (ak and sk):
        # Keep this lazy: only raised when a predict_* function calls get_runtime()
        raise NoCredentialsError()

    sess = boto3.Session(
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        aws_session_token=tk,
        region_name=region,
    )
    return sess.client("sagemaker-runtime")

def _to_bytes(obj):
    # Accept bytes, file-like (e.g., UploadedFile), or PIL.Image
    if obj is None:
        raise ValueError("No image provided.")
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if hasattr(obj, "read"):
        data = obj.read()
        # Streamlit UploadedFile needs seek(0) so it can be re-read elsewhere
        try:
            obj.seek(0)
        except Exception:
            pass
        return data
    # Fallback: try PIL without importing unless needed
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            buf = io.BytesIO()
            obj.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        pass
    raise TypeError("Unsupported image input; pass bytes, a file-like object, or a PIL image.")

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
        # TODO: parse your model’s response
        return raw
    except ClientError as e:
        # Re-raise so Streamlit shows a concise error
        raise

def predict_from_text(text: str, endpoint_name=None):
    rt = get_runtime()
    endpoint = endpoint_name or st.secrets.get("NLP_ENDPOINT", os.getenv("NLP_ENDPOINT"))
    if not endpoint:
        # Provide a friendly message if you’re only testing images
        raise ValueError("NLP_ENDPOINT is not set. Set it or remove predict_from_text imports/calls.")
    payload = json.dumps({"inputs": text})  # adjust to your model’s expected schema
    try:
        resp = rt.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=payload,
        )
        raw = resp["Body"].read()
        # TODO: parse JSON response as needed
        return raw
    except ClientError:
        raise


