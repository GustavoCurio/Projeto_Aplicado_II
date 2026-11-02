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
    """Padroniza texto em NFKC, minúsculas e espaços, retirando pontuação.

    A mesma transformação é usada na deduplicação e no vetorizador. Acentos
    são preservados; a função não realiza stemming ou remoção de stopwords.
    """
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def review_examples(reviews):
    """Prepara exemplos únicos de sentimento sem alterar a tabela recebida.

    Args:
        reviews: DataFrame com review_score e review_comment_message.

    Returns:
        DataFrame text/label: notas 1–2 são negativas e 4–5 positivas.
        Exclui notas 3, comentários nulos, textos normalizados muito curtos e
        todas as ocorrências de textos com classes conflitantes.

    A deduplicação precede a separação treino/teste para evitar que textos
    normalizados idênticos apareçam nas duas partições.
    """
    selected = reviews.loc[reviews["review_score"].isin([1, 2, 4, 5]) & reviews["review_comment_message"].notna()].copy()
    selected["text"] = selected["review_comment_message"].map(normalize_text)
    selected = selected.loc[selected["text"].str.len().ge(3)].copy()
    selected["label"] = selected["review_score"].map(lambda score: "positiva" if score >= 4 else "negativa")
    ambiguous = selected.groupby("text")["label"].transform("nunique").gt(1)
    selected = selected.loc[~ambiguous].drop_duplicates("text")
    return selected[["text", "label"]].reset_index(drop=True)


def train_model(reviews, output_path=None):
    """Treina e avalia um classificador binário com divisão estratificada fixa.

    Args:
        reviews: Avaliações brutas com nota e comentário, usadas por review_examples.
        output_path: Arquivo joblib opcional; None mantém o artefato só em memória.

    Returns:
        Dicionário com pipeline ajustado e metrics, incluindo matriz de
        confusão, métricas por classe e F1 do baseline majoritário.

    Raises:
        ValueError: Restam menos de cinco textos únicos em alguma classe,
            ou a base não contém exatamente as duas classes esperadas.

    Reserva 20% para teste com seed 42. O TF-IDF aprende apenas no treino;
    as métricas descrevem uma divisão aleatória, não generalização temporal.
    Quando solicitado, cria a pasta de destino e grava o artefato no disco.
    """
    examples = review_examples(reviews)
    counts = examples["label"].value_counts()
    if len(counts) != 2 or counts.min() < 5:
        raise ValueError("São necessários ao menos 5 textos únicos por classe, sem rótulos conflitantes.")
    # A limpeza já ocorreu: textos normalizados idênticos não cruzam a divisão.
    train, test = train_test_split(examples, test_size=0.2, random_state=42, stratify=examples["label"])
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2), max_features=20000, min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    # Aprender o vocabulário aqui mantém o conjunto de teste fora do ajuste.
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
    """Estima sentimento para uma avaliação usando um artefato já treinado.

    Args:
        artifact: Dicionário retornado por train_model, contendo pipeline.
        text: Avaliação com conteúdo e no máximo 3.000 caracteres.

    Returns:
        Dicionário classe/escore/aviso. Classe e escore são None quando o
        texto não ativa nenhum termo conhecido. O escore não é calibrado.

    Raises:
        ValueError: O texto está vazio ou ultrapassa o tamanho aceito.
    """
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
