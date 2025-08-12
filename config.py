import os
SETTINGS = {
    "AWS_REGION":  os.getenv("AWS_REGION", "us-east-1"),
    "IMG_ENDPOINT": os.getenv("IMG_ENDPOINT", "plant-resnet50-prod"),
    "NLP_ENDPOINT": os.getenv("NLP_ENDPOINT", "plant-nlp-prod"),
    "USE_STUB":    os.getenv("USE_STUB", "false").lower() == "true",
}
