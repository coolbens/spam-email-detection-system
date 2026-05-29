from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

# Kuhaon ang root directory sa project.
#
# Example:
# project/
# ├── artifacts/
# ├── model/
# └── utils/
#
# Kung naa ni nga file sulod sa utils/,
# ang BASE_DIR mahimong project/
BASE_DIR = Path(__file__).resolve().parents[1]

# Folder diin i-save ang tanan artifacts sama sa:
# - model.pkl
# - vectorizer.pkl
# - metrics.json
ARTIFACT_DIR = BASE_DIR / "artifacts"

# Siguraduhon nga naa ang artifacts folder.
# Kung wala pa, automatic nga himuon.
ARTIFACT_DIR.mkdir(exist_ok=True)


# I-save ang Python object ngadto sa pickle file.
#
# Example:
# save_pickle(model, "model.pkl")
#
# Kasagaran gigamit para sa:
# - trained models
# - vectorizers
# - preprocessors
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# I-load balik ang object gikan sa pickle file.
#
# Example:
# model = load_pickle("model.pkl")
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# I-save ang Python object ngadto sa JSON file.
#
# Example:
# save_json(metrics, "metrics.json")
#
# indent=2 aron mas readable ang JSON file.
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# I-load ang JSON file ug himuon balik nga Python object.
#
# Type hint:
# os.PathLike -> mahimong Path object o file path
# Any         -> bisan unsang klase sa data ang mabalik
#
# Example:
# metrics = load_json("metrics.json")
def load_json(path: os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)