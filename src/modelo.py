from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================
# CAMINHO DO PROJETO
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

ARQUIVO_DADOS = (
    BASE_DIR
    / "data"
    / "processed"
    / "avaliacoes_tratadas.csv"
)


# =========================
# LEITURA DA BASE
# =========================

dados = pd.read_csv(
    ARQUIVO_DADOS
)


# =========================
# PROTEÇÃO CONTRA NULOS
# =========================

# Remove textos e classes nulas
dados = dados.dropna(
    subset=[
        "texto_processado",
        "classe"
    ]
)

# Remove textos vazios
dados = dados[
    dados["texto_processado"]
    .str.strip()
    != ""
].copy()


# =========================
# VERIFICAÇÃO DA BASE
# =========================

print("\n===== BASE CARREGADA =====")

print(
    "Quantidade de registros:",
    len(dados)
)

print(
    "Textos nulos:",
    dados["texto_processado"]
    .isnull()
    .sum()
)

print(
    "Classes nulas:",
    dados["classe"]
    .isnull()
    .sum()
)

print("\nDistribuição das classes:")

print(
    dados["classe"]
    .value_counts()
)


# =========================
# DEFINIÇÃO DE X E Y
# =========================

# X = texto usado pelo modelo
X = dados[
    "texto_processado"
]

# y = classe que queremos prever
y = dados[
    "classe"
]


# =========================
# DIVISÃO TREINO E TESTE
# =========================

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n===== DIVISÃO DOS DADOS =====")

print(
    "Treino:",
    len(X_treino)
)

print(
    "Teste:",
    len(X_teste)
)


# =========================
# TF-IDF
# =========================

vetorizador = TfidfVectorizer(
    max_features=10000
)

# Aprende o vocabulário no treino
X_treino_tfidf = vetorizador.fit_transform(
    X_treino
)

# Usa o mesmo vocabulário no teste
X_teste_tfidf = vetorizador.transform(
    X_teste
)


print("\n===== TF-IDF =====")

print(
    "Formato treino:",
    X_treino_tfidf.shape
)

print(
    "Formato teste:",
    X_teste_tfidf.shape
)


# =========================
# MODELO
# =========================

modelo = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


# =========================
# TREINAMENTO
# =========================

modelo.fit(
    X_treino_tfidf,
    y_treino
)


# =========================
# PREVISÕES
# =========================

previsoes = modelo.predict(
    X_teste_tfidf
)


# =========================
# ACURÁCIA
# =========================

acuracia = accuracy_score(
    y_teste,
    previsoes
)


print("\n===== RESULTADOS =====")

print(
    "Acurácia:",
    acuracia
)

print(
    "Acurácia em percentual:",
    f"{acuracia * 100:.2f}%"
)


# =========================
# RELATÓRIO
# =========================

print("\n===== RELATÓRIO DE CLASSIFICAÇÃO =====")

print(
    classification_report(
        y_teste,
        previsoes
    )
)


# =========================
# MATRIZ DE CONFUSÃO
# =========================

print("\n===== MATRIZ DE CONFUSÃO =====")

print(
    confusion_matrix(
        y_teste,
        previsoes,
        labels=[
            "negativa",
            "positiva"
        ]
    )
)
# =========================
# GRÁFICO DA MATRIZ DE CONFUSÃO
# =========================

matriz = confusion_matrix(
    y_teste,
    previsoes,
    labels=[
        "negativa",
        "positiva"
    ]
)

fig, ax = plt.subplots()

imagem = ax.imshow(
    matriz
)

ax.set_title(
    "Matriz de confusão - Classificação das avaliações"
)

ax.set_xlabel(
    "Classe prevista"
)

ax.set_ylabel(
    "Classe real"
)

ax.set_xticks(
    [0, 1]
)

ax.set_yticks(
    [0, 1]
)

ax.set_xticklabels(
    [
        "Negativa",
        "Positiva"
    ]
)

ax.set_yticklabels(
    [
        "Negativa",
        "Positiva"
    ]
)

# Coloca os valores dentro da matriz
for i in range(matriz.shape[0]):

    for j in range(matriz.shape[1]):

        ax.text(
            j,
            i,
            matriz[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    BASE_DIR
    / "reports"
    / "matriz_confusao.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# GRÁFICO DAS MÉTRICAS
# =========================

precisao, recall, f1, suporte = precision_recall_fscore_support(
    y_teste,
    previsoes,
    labels=[
        "negativa",
        "positiva"
    ]
)

metricas = pd.DataFrame(
    {
        "Precisão": precisao,
        "Recall": recall,
        "F1-score": f1
    },
    index=[
        "Negativa",
        "Positiva"
    ]
)

print("\n===== MÉTRICAS POR CLASSE =====")

print(metricas)

metricas.plot(
    kind="bar"
)

plt.title(
    "Desempenho do modelo por classe"
)

plt.xlabel(
    "Classe"
)

plt.ylabel(
    "Valor da métrica"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Métrica"
)

plt.tight_layout()

plt.savefig(
    BASE_DIR
    / "reports"
    / "metricas_modelo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()