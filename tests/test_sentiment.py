import joblib
import pandas as pd

from olist_insights.sentiment import normalize_text, predict_review, review_examples, train_model


def examples():
    return pd.DataFrame([
        *[{"review_comment_message": f"ótimo produto entrega rápida recomendo qualidade {i}", "review_score": 5} for i in range(20)],
        *[{"review_comment_message": f"péssimo produto não chegou atraso defeito {i}", "review_score": 1} for i in range(20)],
    ])


def test_deduplication_and_conflicting_labels_removed_before_split():
    raw = pd.DataFrame({"review_comment_message": ["Muito bom!", "muito bom", "igual", "igual", "neutro", None],
                        "review_score": [5, 4, 1, 5, 3, 5]})
    clean = review_examples(raw)
    assert clean["text"].tolist() == ["muito bom"]
    assert "não" in normalize_text("NÃO chegou!")


def test_train_persist_and_predict_unseen_review(tmp_path):
    path = tmp_path / "reviews.joblib"
    artifact = train_model(examples(), path)
    loaded = joblib.load(path)
    assert artifact["metrics"]["treino"] + artifact["metrics"]["teste"] == 40
    assert artifact["metrics"]["f1_macro"] > artifact["metrics"]["baseline_f1_macro"]
    assert predict_review(loaded, "ótimo produto entrega rápida")["classe"] == "positiva"
    assert predict_review(loaded, "não chegou péssimo produto")["classe"] == "negativa"
    assert predict_review(loaded, "zyxwvuts")["classe"] is None
