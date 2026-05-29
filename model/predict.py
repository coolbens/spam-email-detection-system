from __future__ import annotations

# Gigamit para sa pagdumala sa file paths sa mas limpyo ug cross-platform nga paagi
from pathlib import Path

# Type hints para mas klaro unsay expected nga data types
from typing import Dict, List, Tuple

# NumPy gigamit para sa array operations ug feature analysis
import numpy as np

# Mga function para sa text preprocessing ug tokenization
from nlp.preprocess import preprocess_text, preprocess_tokens

# Mga helper function ug artifact directory
from utils.helpers import ARTIFACT_DIR, load_json, load_pickle

# Default path sa trained model
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "model.pkl"

# Folder diin gitago ang lain-laing trained models
MODELS_DIR = ARTIFACT_DIR / "models"

# Path sa vectorizer artifact
VECTORIZER_PATH = ARTIFACT_DIR / "vectorizer.pkl"

# Path sa metrics file nga adunay impormasyon sa training results
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


# Function nga mo-load sa model, vectorizer, ug metrics gikan sa artifacts
def load_artifacts(model_name: str | None = None):

    # Sigurohon nga naa ang vectorizer file
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Vectorizer artifact not found. Run `python -m model.train` first."
        )

    # I-load ang metrics file kung naa
    metrics = load_json(METRICS_PATH) if METRICS_PATH.exists() else {}

    # Pilion ang model nga gamiton
    # Priority:
    # 1. model_name nga gipasa
    # 2. best_model gikan sa metrics
    chosen_model = model_name or metrics.get("best_model")

    # Himoon ang path sa piniling model
    model_path = (
        MODELS_DIR / f"{chosen_model}.pkl"
        if chosen_model
        else DEFAULT_MODEL_PATH
    )

    # Kung wala ang model nga gipili, gamiton ang default model
    if not model_path.exists():
        model_path = DEFAULT_MODEL_PATH

    # Kung wala gihapon model file, mohatag ug error
    if not model_path.exists():
        raise FileNotFoundError(
            "Model artifact not found. Run `python -m model.train` first."
        )

    # I-load ang model ug vectorizer
    model = load_pickle(model_path)
    vectorizer = load_pickle(VECTORIZER_PATH)

    # Ibalik ang loaded artifacts
    return model, vectorizer, metrics


# Function nga mokuha sa spam ug ham probabilities
def _get_probability(model, vector):

    # Predict probabilities sa input vector
    probs = model.predict_proba(vector)[0]

    # Lista sa available classes (ham/spam)
    classes = list(model.classes_)

    # Pangitaon ang index sa spam ug ham
    spam_idx = classes.index("spam")
    ham_idx = classes.index("ham")

    # Ibalik ang probabilities isip float
    return float(probs[spam_idx]), float(probs[ham_idx])


# Function nga mokuha sa mga words nga adunay pinakadako nga epekto sa prediction
def _top_feature_influence(
    model,
    vectorizer,
    processed_text: str,
    top_n: int = 8,
) -> List[Dict]:

    # Kuhaon tanan feature names gikan sa vectorizer
    feature_names = np.array(vectorizer.get_feature_names_out())

    # I-convert ang processed text ngadto sa vector
    vector = vectorizer.transform([processed_text])

    # Kuhaon ang indices sa active words nga naa sa text
    active_indices = vector.nonzero()[1]

    # Kung walay active words, walay ma-analyze
    if len(active_indices) == 0:
        return []

    contributions = []

    # Para sa models nga adunay coef_ (e.g. Logistic Regression)
    if hasattr(model, "coef_"):

        # Kuhaon ang feature weights
        weights = model.coef_[0]

        # Kwentahon ang contribution sa matag active feature
        for idx in active_indices:
            score = float(vector[0, idx] * weights[idx])

            contributions.append(
                (feature_names[idx], score)
            )

    # Para sa Naive Bayes nga adunay feature_log_prob_
    elif hasattr(model, "feature_log_prob_"):

        spam_idx = list(model.classes_).index("spam")
        ham_idx = list(model.classes_).index("ham")

        spam_log_prob = model.feature_log_prob_[spam_idx]
        ham_log_prob = model.feature_log_prob_[ham_idx]

        # Kwentahon ang kalainan sa spam ug ham probabilities
        for idx in active_indices:

            score = float(
                vector[0, idx] *
                (spam_log_prob[idx] - ham_log_prob[idx])
            )

            contributions.append(
                (feature_names[idx], score)
            )

    # Kung unsupported model
    else:
        return []

    # I-sort base sa pinakadako nga impact
    contributions = sorted(
        contributions,
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    # Himoon nga mas readable ang output
    return [
        {
            "term": term,

            # Kung positive ang score, mas pabor sa spam
            # Kung negative, mas pabor sa ham
            "impact": "spam" if score > 0 else "ham",

            # Rounded score para limpyo tan-awon
            "score": round(score, 4),
        }
        for term, score in contributions
    ]


# Function para mag-predict sa usa ka text
def predict_text(
    text: str,
    model_name: str | None = None,
) -> Dict:

    # I-load ang model ug vectorizer
    model, vectorizer, _ = load_artifacts(
        model_name=model_name
    )

    # I-preprocess ang text
    processed = preprocess_text(text)

    # I-convert ang processed text ngadto sa vector
    vector = vectorizer.transform([processed])

    # Predict sa label (spam o ham)
    pred = model.predict(vector)[0]

    # Kuhaon ang spam ug ham probabilities
    spam_prob, ham_prob = _get_probability(
        model,
        vector,
    )

    # Confidence sa final prediction
    confidence = (
        spam_prob
        if pred == "spam"
        else ham_prob
    )

    # Kuhaon ang pinaka-importanteng words
    highlights = _top_feature_influence(
        model,
        vectorizer,
        processed,
    )

    # Ibalik tanan resulta
    return {
        "label": pred,
        "confidence": float(confidence),
        "spam_probability": float(spam_prob),
        "ham_probability": float(ham_prob),

        # Processed version sa text
        "processed_text": processed,

        # Mga words nga nakaapekto sa prediction
        "highlights": highlights,

        # Original tokens gikan sa text
        "tokens": preprocess_tokens(text),

        # Model nga gigamit
        "model_name": model_name,
    }


# Function para mag-predict sa daghang texts
def predict_many(
    texts: List[str],
    model_name: str | None = None,
) -> List[Dict]:

    # Tawagon ang predict_text() sa matag text
    return [
        predict_text(
            text,
            model_name=model_name,
        )
        for text in texts
    ]