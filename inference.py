# inference.py
import os, io, json, glob, base64, re
import boto3
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms
from collections import OrderedDict

# ---- Runtime config ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
S3_CLIENT = boto3.client("s3")

# ---------------- Helpers ----------------
def _unwrap_state_dict(raw):
    """
    Unwrap common checkpoint containers and strip DDP/Lightning prefixes.
    Returns (state_dict_or_model, classes_from_ckpt, img_size_from_ckpt)
    """
    classes = None
    img_size = None

    # If someone saved the whole nn.Module
    if not isinstance(raw, dict):
        return raw, classes, img_size

    # Typical wrappers
    sd = None
    for key in ("state_dict", "model_state_dict", "net", "model"):
        if key in raw and isinstance(raw[key], dict):
            sd = raw[key]
            classes = raw.get("classes", None)
            img_size = raw.get("img_size", None)
            break
    if sd is None:
        sd = raw

    # strip common prefixes
    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        out[nk] = v
    return out, classes, img_size

def _infer_fc_shape(sd):
    """Try to infer classifier layout and number of classes from the state_dict."""
    if not isinstance(sd, dict):
        return (None, None, None)
    # Sequential head e.g. fc.0 -> fc.3
    if "fc.0.weight" in sd and "fc.3.weight" in sd:
        hid = sd["fc.0.weight"].shape[0]
        out = sd["fc.3.weight"].shape[0]
        return ("two", hid, out)
    # Single Linear head e.g. fc.weight
    if "fc.weight" in sd:
        out = sd["fc.weight"].shape[0]
        return ("one", None, out)
    return (None, None, None)

def _load_classes_from_dir(model_dir):
    """
    Try to load classes from common files. Returns list or None if not found.
    Supports list OR class_to_idx dict.
    """
    for name in ("classes.json", "labels.json", "class_to_idx.json"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return list(data)
            if isinstance(data, dict):  # class->idx
                inv = {}
                for k, v in data.items():
                    try:
                        inv[int(v)] = k
                    except Exception:
                        # tolerate string-ified ints
                        inv[int(str(v))] = k
                return [inv[i] for i in sorted(inv)]
    return None

def _canonical_classes(classes, num_classes):
    """
    If classes missing or count mismatch, generate numeric fallback and align shapes.
    """
    if not classes or not isinstance(classes, (list, tuple)) or len(classes) == 0:
        return [f"class_{i}" for i in range(num_classes)]
    if len(classes) != num_classes:
        # Prefer checkpoint-implied count; truncate or pad deterministically
        if len(classes) > num_classes:
            return list(classes)[:num_classes]
        else:
            pad = [f"class_{i}" for i in range(len(classes), num_classes)]
            return list(classes) + pad
    return list(classes)

def _s3_read_bytes(uri: str) -> bytes:
    """Read an object from s3://bucket/key into bytes."""
    m = re.match(r"^s3://([^/]+)/(.+)$", uri.strip())
    if not m:
        raise ValueError("Invalid s3_uri. Expected format s3://bucket/key")
    bucket, key = m.group(1), m.group(2)
    obj = S3_CLIENT.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def _open_image(b: bytes) -> Image.Image:
    """Open bytes as a RGB image with EXIF orientation respected."""
    img = Image.open(io.BytesIO(b)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    return img

# --------------- SageMaker hooks ---------------
def model_fn(model_dir):
    # Find a .pth/.pt inside /opt/ml/model
    ckpts = glob.glob(os.path.join(model_dir, "**", "*.pth"), recursive=True) + \
            glob.glob(os.path.join(model_dir, "**", "*.pt"), recursive=True)
    if not ckpts:
        raise FileNotFoundError("No .pth/.pt checkpoint found in model_dir.")
    ckpt_path = sorted(ckpts)[0]

    # Load checkpoint (prefer weights_only when available)
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(ckpt_path, map_location="cpu")

    state_or_model, sd_classes, sd_img = _unwrap_state_dict(raw)

    # Build base model (ResNet50)
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features

    # If a full model object was saved
    if not isinstance(state_or_model, dict):
        model = state_or_model
        # Try to get num_classes from model.fc if present
        try:
            num_classes = model.fc.out_features
        except Exception:
            num_classes = None
    else:
        # Construct classifier head to match checkpoint
        head_kind, hid_ck, out_ck = _infer_fc_shape(state_or_model)
        # Fall back if we can't infer yet
        num_classes = out_ck if out_ck else None

        if head_kind == "two" and hid_ck and num_classes:
            model.fc = nn.Sequential(
                nn.Linear(in_features, hid_ck, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(hid_ck, num_classes, bias=True),
            )
        elif head_kind == "one" and num_classes:
            model.fc = nn.Linear(in_features, num_classes, bias=True)
        # else keep default until we know classes

        # Load weights (non-strict to tolerate minor key diffs)
        model.load_state_dict(state_or_model, strict=False)

        # If still unknown, try to pull from current head
        if num_classes is None:
            try:
                num_classes = model.fc.out_features
            except Exception:
                pass

    # Determine classes
    file_classes = _load_classes_from_dir(model_dir)
    classes = sd_classes if sd_classes else file_classes

    # If we still don't know num_classes, infer from classes length or last resort 1
    if num_classes is None:
        num_classes = len(classes) if classes else 1

    # Finalize classes alignment
    classes = _canonical_classes(classes, num_classes)

    # If our current head doesn't match the final class count, rebuild it cleanly
    try:
        if model.fc.out_features != len(classes):
            model.fc = nn.Linear(in_features, len(classes), bias=True)
    except Exception:
        # Some custom models may lack .fc
        pass

    model.to(DEVICE).eval()
    torch.set_grad_enabled(False)

    # Use size from checkpoint if present, else env, else default
    img_size = int(sd_img) if (sd_img and str(sd_img).isdigit()) else DEFAULT_IMG_SIZE

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    return {"model": model, "classes": classes, "transform": tfm, "img_size": img_size}

def input_fn(request_body, content_type=None):
    """
    Accepts:
      - image/* or application/x-image or application/octet-stream: raw bytes
      - application/json: {"image_b64": "..."} or {"s3_uri": "s3://..."}
        (If {"text": "..."} is provided, raise a clear message to use NLP endpoint.)
    """
    ct = (content_type or "").lower().split(";")[0].strip()

    # Raw image bytes path
    if ct in ("image/jpeg", "image/jpg", "image/png", "image/webp",
              "application/x-image", "application/octet-stream"):
        b = request_body if isinstance(request_body, (bytes, bytearray)) else bytes(request_body)
        return _open_image(b)

    # JSON path
    if ct == "application/json" or isinstance(request_body, (str, bytes, bytearray)):
        if isinstance(request_body, (bytes, bytearray)):
            body = request_body.decode("utf-8", "ignore")
        else:
            body = request_body
        try:
            data = json.loads(body)
        except Exception:
            raise ValueError("Invalid JSON. Provide {'image_b64': '...'} or {'s3_uri': 's3://...'}.")

        # Guard: users sometimes send text intended for an NLP endpoint
        if "text" in data:
            raise ValueError(
                "This endpoint expects an IMAGE, but received a 'text' field. "
                "Send raw image bytes, or JSON with 'image_b64' or 's3_uri'. "
                "Use your NLP endpoint for text classification."
            )

        if "image_b64" in data:
            try:
                img_bytes = base64.b64decode(data["image_b64"], validate=True)
            except Exception:
                raise ValueError("image_b64 is not valid base64.")
            return _open_image(img_bytes)

        if "s3_uri" in data:
            img_bytes = _s3_read_bytes(data["s3_uri"])
            return _open_image(img_bytes)

        raise ValueError("JSON must include 'image_b64' or 's3_uri' for image classification.")

    # Fallback attempt: treat as raw image bytes
    try:
        b = request_body if isinstance(request_body, (bytes, bytearray)) else bytes(request_body)
        return _open_image(b)
    except Exception:
        raise ValueError(
            f"Unsupported ContentType '{content_type}'. "
            "Use image/*, application/x-image, application/octet-stream, or application/json "
            "with 'image_b64' or 's3_uri'."
        )

def predict_fn(input_object, model_artifacts):
    img: Image.Image = input_object
    model = model_artifacts["model"]
    classes = model_artifacts["classes"]
    tfm = model_artifacts["transform"]

    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_p, top_i = probs.topk(min(5, probs.shape[1]), dim=1)

    top_i = top_i[0].cpu().tolist()
    top_p = top_p[0].cpu().tolist()
    top5 = [{"label": classes[i], "score": float(p)} for i, p in zip(top_i, top_p)]

    return {
        "label": top5[0]["label"],
        "score": top5[0]["score"],
        "top5": top5
    }

def output_fn(prediction, accept="application/json"):
    # Normalize accept header
    a = (accept or "application/json").lower()
    if "application/json" in a or "*/*" in a:
        return json.dumps(prediction), "application/json"
    # Default to JSON
    return json.dumps(prediction), "application/json"
