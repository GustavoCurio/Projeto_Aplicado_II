"""Modelo reproduzível com textos deduplicados antes da separação treino/teste."""
from pathlib import Path
import re
import unicodedata

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def review_examples(reviews):
    selected = reviews.loc[reviews["review_score"].isin([1, 2, 4, 5]) & reviews["review_comment_message"].notna()].copy()
    selected["text"] = selected["review_comment_message"].map(normalize_text)
    selected = selected.loc[selected["text"].str.len().ge(3)].copy()
    selected["label"] = selected["review_score"].map(lambda score: "positiva" if score >= 4 else "negativa")
    ambiguous = selected.groupby("text")["label"].transform("nunique").gt(1)
    selected = selected.loc[~ambiguous].drop_duplicates("text")
    return selected[["text", "label"]].reset_index(drop=True)


def train_model(reviews, output_path=None):
    examples = review_examples(reviews)
    counts = examples["label"].value_counts()
    if len(counts) != 2 or counts.min() < 5:
        raise ValueError("São necessários ao menos 5 textos únicos por classe, sem rótulos conflitantes.")
    train, test = train_test_split(examples, test_size=0.2, random_state=42, stratify=examples["label"])
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2), max_features=20000, min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline.fit(train["text"], train["label"])
    prediction = pipeline.predict(test["text"])
    baseline = DummyClassifier(strategy="most_frequent").fit(train[["text"]], train["label"])
    baseline_prediction = baseline.predict(test[["text"]])
    metrics = {
        "amostras_antes_da_limpeza": len(reviews), "textos_unicos_utilizados": len(examples),
        "treino": len(train), "teste": len(test), "seed": 42,
        "acuracia": round(accuracy_score(test["label"], prediction), 4),
        "f1_macro": round(f1_score(test["label"], prediction, average="macro"), 4),
        "baseline_f1_macro": round(f1_score(test["label"], baseline_prediction, average="macro"), 4),
        "classes": ["negativa", "positiva"],
        "matriz_confusao": confusion_matrix(test["label"], prediction, labels=["negativa", "positiva"]).tolist(),
        "por_classe": classification_report(test["label"], prediction, output_dict=True, zero_division=0),
        "limitacoes": "Avaliação aleatória em dados históricos. Não valida generalização temporal. Notas são rótulos aproximados de sentimento; nota 3 foi excluída.",
    }
    artifact = {"pipeline": pipeline, "metrics": metrics}
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, destination)
    return artifact


def predict_review(artifact, text):
    if not text.strip() or len(text) > 3000:
        raise ValueError("Digite uma avaliação de 1 a 3.000 caracteres.")
    pipeline = artifact["pipeline"]
    features = pipeline.named_steps["tfidf"].transform([text])
    if features.nnz == 0:
        return {"classe": None, "escore": None, "aviso": "Texto sem palavras conhecidas pelo modelo; não é possível classificar."}
    probabilities = pipeline.predict_proba([text])[0]
    index = int(probabilities.argmax())
    return {"classe": str(pipeline.classes_[index]), "escore": round(float(probabilities[index]), 4),
            "aviso": "Escore do modelo, não uma probabilidade calibrada. Exemplo acadêmico; pode errar, especialmente em ironia e contexto novo."}
