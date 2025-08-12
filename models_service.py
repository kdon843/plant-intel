import io, os, json
import boto3
from PIL import Image
from config import SETTINGS

_runtime = boto3.client("sagemaker-runtime", region_name=SETTINGS["AWS_REGION"])

def _recommendation_for(label: str) -> str:
    recs = {
        "tomato_healthy": "No treatment needed. Maintain spacing, irrigate at soil level, and monitor weekly.",
        "tomato_blight": "Prune infected leaves, avoid overhead watering, and apply copper/chlorothalonil per label.",
        "powdery_mildew": "Improve airflow, remove affected leaves, and apply sulfur-based fungicide as directed.",
    }
    return recs.get(label, "Follow IPM: isolate plant, prune, improve airflow, rotate crops, consult extension office.")

def predict_from_image(image: Image.Image):
    if SETTINGS["USE_STUB"]:
        return "tomato_blight", 0.92, _recommendation_for("tomato_blight")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    payload = buf.getvalue()

    resp = _runtime.invoke_endpoint(
        EndpointName=SETTINGS["IMG_ENDPOINT"],
        ContentType="application/x-image",
        Body=payload,
    )
    result = json.loads(resp["Body"].read())
    label = result.get("label", "unknown")
    score = float(result.get("score", 0.0))
    return label, score, _recommendation_for(label)

def predict_from_text(text: str):
    if SETTINGS["USE_STUB"]:
        return "powdery_mildew", 0.88, _recommendation_for("powdery_mildew")

    resp = _runtime.invoke_endpoint(
        EndpointName=SETTINGS["NLP_ENDPOINT"],
        ContentType="application/json",
        Body=json.dumps({"text": text}),
    )
    result = json.loads(resp["Body"].read())
    label = result.get("label", "unknown")
    score = float(result.get("score", 0.0))
    return label, score, _recommendation_for(label)
