# inference.py
import os, io, json, glob
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from collections import OrderedDict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# ---------- helpers ----------
def _unwrap_state_dict(raw):
    """Unwrap common checkpoint containers and strip DDP/Lightning prefixes."""
    classes = None
    img_size = None
    if not isinstance(raw, dict):
        return raw, classes, img_size  # whole model object saved
    for key in ["state_dict", "model_state_dict", "net", "model"]:
        if key in raw and isinstance(raw[key], dict):
            sd = raw[key]
            classes = raw.get("classes", None)
            img_size = raw.get("img_size", None)
            break
    else:
        sd = raw

    # strip common prefixes
    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."): nk = nk[len("module."):]
        if nk.startswith("model."):  nk = nk[len("model."):]
        out[nk] = v
    return out, classes, img_size

def _infer_fc_shape(sd):
    if not isinstance(sd, dict):
        return (None, None, None)
    k0 = next((k for k in sd.keys() if k.startswith("fc.0.weight")), None)
    k3 = next((k for k in sd.keys() if k.startswith("fc.3.weight")), None)
    k1 = sd.get("fc.weight")
    if k0 is not None and k3 is not None:
        hid = sd["fc.0.weight"].shape[0]
        out = sd["fc.3.weight"].shape[0]
        return ("two", hid, out)
    if k1 is not None:
        out = sd["fc.weight"].shape[0]
        return ("one", None, out)
    return (None, None, None)

def _load_classes(model_dir, sd_classes):
    # Prefer classes embedded in checkpoint; else look for a json file in model_dir
    if sd_classes and isinstance(sd_classes, (list, tuple)):
        return list(sd_classes)
    for name in ("classes.json", "labels.json", "class_to_idx.json"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return list(data)
            if isinstance(data, dict):  # class->idx
                inv = {int(v): k for k, v in data.items()}
                return [inv[i] for i in sorted(inv)]
    raise RuntimeError("No class names found in checkpoint or model_dir.")

# ---------- SageMaker model server hooks ----------
def model_fn(model_dir):
    # Find a .pth/.pt inside /opt/ml/model
    ckpts = glob.glob(os.path.join(model_dir, "**", "*.pth"), recursive=True) + \
            glob.glob(os.path.join(model_dir, "**", "*.pt"), recursive=True)
    if not ckpts:
        raise FileNotFoundError("No .pth/.pt checkpoint found in model_dir")
    ckpt_path = sorted(ckpts)[0]

    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict, sd_classes, sd_img = _unwrap_state_dict(raw)
    classes = _load_classes(model_dir, sd_classes)

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features

    head_kind, hid_ck, out_ck = _infer_fc_shape(state_dict)
    num_classes = len(classes)
    if head_kind == "two":
        if out_ck and out_ck != num_classes:
            num_classes = out_ck
            classes = classes[:num_classes]
        model.fc = nn.Sequential(
            nn.Linear(in_features, hid_ck, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hid_ck, num_classes, bias=True),
        )
    elif head_kind == "one":
        if out_ck and out_ck != num_classes:
            num_classes = out_ck
            classes = classes[:num_classes]
        model.fc = nn.Linear(in_features, num_classes, bias=True)
    else:
        model.fc = nn.Linear(in_features, num_classes, bias=True)

    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
    else:
        model = state_dict

    model.to(DEVICE).eval()

    # Save transforms and classes into a dict we return
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return {"model": model, "classes": classes, "transform": tfm}

def input_fn(request_body, content_type=None):
    # Accept raw image bytes
    if isinstance(request_body, (bytes, bytearray)):
        b = bytes(request_body)
    else:
        # Some runtimes pass str
        b = request_body if isinstance(request_body, bytes) else bytes(request_body, "utf-8", "ignore")

    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        raise ValueError("Unable to decode image. Ensure ContentType is image/* or application/x-image.")

    return img  # PIL.Image

def predict_fn(input_object, model_artifacts):
    img: Image.Image = input_object
    model = model_artifacts["model"]
    classes = model_artifacts["classes"]
    tfm = model_artifacts["transform"]

    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_p, top_i = probs.topk(5, dim=1)

    top_i = top_i[0].cpu().tolist()
    top_p = top_p[0].cpu().tolist()

    top5 = [{"label": classes[i], "score": float(p)} for i, p in zip(top_i, top_p)]
    out = {
        "label": top5[0]["label"],
        "score": top5[0]["score"],
        "top5": top5
    }
    return out

def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
