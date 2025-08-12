# models_service.py
import os
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
        raise NoCredentialsError()

    sess = boto3.Session(
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        aws_session_token=tk,
        region_name=region,
    )
    return sess.client("sagemaker-runtime")

def predict_from_image(img_bytes, endpoint_name=None):
    rt = get_runtime()
    endpoint = endpoint_name or st.secrets.get("IMG_ENDPOINT", os.getenv("IMG_ENDPOINT"))
    if not endpoint:
        raise ValueError("IMG_ENDPOINT is not set.")
    try:
        resp = rt.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/x-image",
            Body=img_bytes,
        )
        body = resp["Body"].read()
        # parse your model's response here...
        return body
    except ClientError as e:
        # surface a concise error in the UI
        raise

