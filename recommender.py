# recommender.py
import os, pandas as pd
from typing import Optional, List, Dict

# === paste your whole recommender code here ===
# (keep all helpers as-is)

# Add two convenience wrappers:
def load():
    # returns nothing; just ensures TF-IDF is ready on first call
    # your globals (rec_base, passages, details) are already loaded by import
    return True

def recommend(disease: str, host: Optional[str] = None, k: int = 3) -> List[Dict]:
    return recommend_for_disease(disease, host_hint=host, k=k, allow_other_hosts=True)
