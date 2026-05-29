from __future__ import annotations

import re
import string
from typing import Iterable, List

import nltk
from nltk.stem import PorterStemmer


# Siguraduhon nga available ang stopwords.
# Kung wala pa na-download sa NLTK, i-download niya automatic.
# Kung naay error sa NLTK, mogamit siya sa fallback stopwords sa scikit-learn.
def _ensure_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return set(ENGLISH_STOP_WORDS)


# Global stop words list, example: "the", "is", "and", "a"
STOP_WORDS = _ensure_stopwords()

# Stemmer nga mopamubo sa words:
# example: "running" -> "run", "played" -> "play"
STEMMER = PorterStemmer()


# Limpyo-on ang text para ready na siya sa NLP processing.
def clean_text(text: str) -> str:
    # I-convert ang input into string dayon himuong lowercase.
    text = str(text).lower()

    # Ilisan ang website links og word nga "URL".
    text = re.sub(r"http\S+|www\S+", " URL ", text)

    # Ilisan ang email address og word nga "EMAIL".
    text = re.sub(r"\S+@\S+", " EMAIL ", text)

    # Ilisan ang numbers og word nga "NUMBER".
    text = re.sub(r"\d+", " NUMBER ", text)

    # Tangtangon ang punctuation marks like comma, period, !, ?
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tangtangon ang extra spaces para usa ra ka space ang mabilin.
    text = re.sub(r"\s+", " ", text).strip()

    # Ibalik ang limpyo nga text.
    return text


# Kuhaon ang words/tokens gikan sa text.
def tokenize(text: str) -> List[str]:
    # Mangita ra siya og alphabet characters.
    # Dili niya apilon ang numbers or symbols.
    return re.findall(r"[a-zA-Z]+", text)


# I-normalize ang tokens:
# remove stopwords, apply stemming, ug tangtang short words.
def normalize_tokens(tokens: Iterable[str]) -> List[str]:
    normalized = []

    for token in tokens:
        # Kung ang token kay stopword, skip lang.
        if token in STOP_WORDS:
            continue

        # I-stem ang token para makuha ang root form.
        stem = STEMMER.stem(token)

        # I-apil lang kung ang stemmed word mas taas pa sa 1 character.
        if len(stem) > 1:
            normalized.append(stem)

    # Ibalik ang normalized tokens.
    return normalized


# Full preprocessing nga mo-return og list of tokens.
def preprocess_tokens(text: str) -> List[str]:
    # Limpyo-on una ang raw text.
    cleaned = clean_text(text)

    # I-tokenize ang cleaned text.
    tokens = tokenize(cleaned)

    # I-normalize ang tokens.
    return normalize_tokens(tokens)


# Full preprocessing nga mo-return og final string.
def preprocess_text(text: str) -> str:
    # I-join ang processed tokens balik into one string.
    # Example: ["free", "win", "prize"] -> "free win prize"
    return " ".join(preprocess_tokens(text))