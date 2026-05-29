from __future__ import annotations  # Gitugotan ang paggamit sa modern Python type hints bisan sa mas daang versions

import argparse  # Para modawat ug command-line arguments
from pathlib import Path  # Para mas limpyo ug portable nga file path handling

import pandas as pd  # Para sa data manipulation ug CSV reading
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.model_selection import train_test_split  # Para pagbahin sa training ug testing data
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes model

from model.evaluate import evaluate_model, compare_models  # Mga function para sa model evaluation
from nlp.preprocess import preprocess_text  # Function para limpyo ug preprocess sa text
from nlp.vectorizer import build_vectorizer  # Function para paghimo ug TF-IDF o Bag-of-Words vectorizer
from utils.helpers import ARTIFACT_DIR, save_json, save_pickle  # Utilities para save sa files

# Root folder sa project
BASE_DIR = Path(__file__).resolve().parents[1]

# Default location sa dataset
DEFAULT_DATASET = BASE_DIR / "data" / "dataset.csv"

# Folder diin i-save ang mga trained models
MODELS_DIR = ARTIFACT_DIR / "models"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Magbasa sa CSV dataset ug mo-validate sa required columns.
    """

    # Basaha ang CSV file ngadto sa DataFrame
    df = pd.read_csv(csv_path)

    # Required columns sa dataset
    expected = {"label", "text"}

    # Siguroha nga naa ang label ug text columns
    if not expected.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {expected}")

    # Tangtanga ang rows nga walay label o text
    df = df.dropna(subset=["label", "text"]).copy()

    # Limpyoha ang label column
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Siguroha nga string tanan text values
    df["text"] = df["text"].astype(str)

    return df


def train_models(dataset_path: Path = DEFAULT_DATASET, vectorizer_kind: str = "tfidf") -> dict:
    """
    Main training pipeline.
    Mag-train ug Naive Bayes ug Logistic Regression,
    pilion ang pinakamaayo base sa F1 score,
    ug i-save ang artifacts.
    """

    # Load dataset
    df = load_dataset(dataset_path)

    # Preprocess tanan text
    df["processed_text"] = df["text"].apply(preprocess_text)

    # Bahina ang data ngadto sa training ug testing set
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"],   # Features
        df["label"],            # Target labels
        test_size=0.2,          # 20% testing data
        random_state=42,        # Fixed random seed
        stratify=df["label"],   # Balanced ang labels sa train ug test
    )

    # Buhata ang vectorizer (TF-IDF o BOW)
    vectorizer = build_vectorizer(kind=vectorizer_kind)

    # Convert text ngadto sa numerical vectors
    X_train_vec = vectorizer.fit_transform(X_train)

    # Transform lamang ang test data gamit ang training vocabulary
    X_test_vec = vectorizer.transform(X_test)

    # Mga models nga atong i-compare
    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(
            max_iter=2000,          # Daghang iterations para dili dali mo-timeout
            class_weight="balanced" # Mas maayo kung imbalance ang dataset
        ),
    }

    # Storage sa evaluation results
    results = {}

    # Storage sa trained models
    trained_models = {}

    # Loop sa matag model
    for name, model in models.items():

        # Train ang model
        model.fit(X_train_vec, y_train)

        # Predict sa test data
        preds = model.predict(X_test_vec)

        # Compute metrics
        metrics = evaluate_model(y_test, preds)

        # Save metrics
        results[name] = {"metrics": metrics}

        # Save trained model
        trained_models[name] = model

    # Pilia ang model nga adunay pinakataas nga F1 score
    best_model_name = max(
        results,
        key=lambda name: results[name]["metrics"]["f1_score"]
    )

    # Retrain ang vectorizer gamit ang tibuok dataset
    X_full_vec = vectorizer.fit_transform(df["processed_text"])

    # Siguroha nga naa ang models folder
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Retrain ug save ang matag model gamit ang full dataset
    for name, model in trained_models.items():

        model.fit(X_full_vec, df["label"])

        # Save model file
        save_pickle(model, MODELS_DIR / f"{name}.pkl")

        # Save artifact path sa results
        results[name]["artifact_path"] = str(
            MODELS_DIR / f"{name}.pkl"
        )

    # Siguroha nga naa ang artifacts folder
    ARTIFACT_DIR.mkdir(exist_ok=True)

    # Save vectorizer
    save_pickle(
        vectorizer,
        ARTIFACT_DIR / "vectorizer.pkl"
    )

    # Save best model para dali ma-load sa prediction
    save_pickle(
        trained_models[best_model_name],
        ARTIFACT_DIR / "model.pkl"
    )

    # Save metadata ug evaluation results
    save_json(
        {
            "dataset_path": str(dataset_path),
            "vectorizer_kind": vectorizer_kind,
            "best_model": best_model_name,
            "available_models": list(trained_models.keys()),
            "results": results,
            "summary": compare_models(results),
        },
        ARTIFACT_DIR / "metrics.json",
    )

    # Ibalik ang summary sa training
    return {
        "best_model": best_model_name,
        "available_models": list(trained_models.keys()),
        "results": results,
        "summary": compare_models(results),
    }


# Entry point kung direkta daganon ang file
if __name__ == "__main__":

    # Command line parser
    parser = argparse.ArgumentParser(
        description="Train spam detection models."
    )

    # Dataset path argument
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET)
    )

    # Vectorizer type argument
    parser.add_argument(
        "--vectorizer",
        type=str,
        default="tfidf",
        choices=["tfidf", "bow"]
    )

    # Basaha ang command-line arguments
    args = parser.parse_args()

    # Tawaga ang training function
    output = train_models(
        Path(args.dataset),
        args.vectorizer
    )

    # I-print ang final summary
    print(output["summary"])