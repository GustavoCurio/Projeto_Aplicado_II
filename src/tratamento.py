from pathlib import Path
import re

import pandas as pd
import nltk
from nltk.corpus import stopwords


# =========================
# CAMINHO DO PROJETO
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

ARQUIVO_AVALIACOES = (
    BASE_DIR
    / "data"
    / "raw"
    / "olist_order_reviews_dataset.csv"
)

PASTA_PROCESSADOS = (
    BASE_DIR
    / "data"
    / "processed"
)

ARQUIVO_SAIDA = (
    PASTA_PROCESSADOS
    / "avaliacoes_tratadas.csv"
)


# =========================
# NLTK
# =========================

nltk.download(
    "stopwords",
    quiet=True
)


# =========================
# LEITURA DA BASE
# =========================

avaliacoes = pd.read_csv(
    ARQUIVO_AVALIACOES
)


# =========================
# SELEÇÃO DOS COMENTÁRIOS
# =========================

# Remove avaliações sem comentário
avaliacoes_texto = avaliacoes.dropna(
    subset=["review_comment_message"]
).copy()


# =========================
# REMOÇÃO DA NOTA 3
# =========================

# 1 e 2 = negativa
# 4 e 5 = positiva
# 3 = neutra e será removida
avaliacoes_texto = avaliacoes_texto[
    avaliacoes_texto["review_score"] != 3
].copy()


# =========================
# CRIAÇÃO DAS CLASSES
# =========================

def criar_classe(nota):

    if nota >= 4:
        return "positiva"

    return "negativa"


avaliacoes_texto["classe"] = (
    avaliacoes_texto["review_score"]
    .apply(criar_classe)
)


# =========================
# LIMPEZA DO TEXTO
# =========================

def limpar_texto(texto):

    # Coloca tudo em minúsculas
    texto = texto.lower()

    # Remove pontuação e caracteres especiais
    # Mantém letras, números, acentos e espaços
    texto = re.sub(
        r"[^a-zA-ZÀ-ÿ0-9\s]",
        " ",
        texto
    )

    # Remove espaços repetidos
    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


avaliacoes_texto["texto_limpo"] = (
    avaliacoes_texto["review_comment_message"]
    .apply(limpar_texto)
)


# =========================
# STOPWORDS
# =========================

stop_words = set(
    stopwords.words("portuguese")
)

# Mantém palavras de negação
palavras_importantes = {
    "não",
    "nem"
}

stop_words = (
    stop_words
    - palavras_importantes
)


def remover_stopwords(texto):

    palavras = texto.split()

    palavras_filtradas = [
        palavra
        for palavra in palavras
        if palavra not in stop_words
    ]

    return " ".join(
        palavras_filtradas
    )


avaliacoes_texto["texto_processado"] = (
    avaliacoes_texto["texto_limpo"]
    .apply(remover_stopwords)
)


# =========================
# REMOVE TEXTOS VAZIOS
# =========================

# Remove valores nulos
avaliacoes_texto = avaliacoes_texto.dropna(
    subset=["texto_processado"]
)

# Remove textos que ficaram vazios
avaliacoes_texto = avaliacoes_texto[
    avaliacoes_texto["texto_processado"]
    .str.strip()
    != ""
].copy()


# =========================
# RESULTADOS
# =========================

print("\n===== BASE TEXTUAL =====")

print(
    "Quantidade de comentários:",
    len(avaliacoes_texto)
)

print("\nDistribuição das classes:")

print(
    avaliacoes_texto["classe"]
    .value_counts()
)

print("\nTextos nulos:")

print(
    avaliacoes_texto[
        "texto_processado"
    ]
    .isnull()
    .sum()
)


# =========================
# EXEMPLOS DE LIMPEZA
# =========================

print("\n===== EXEMPLOS DE LIMPEZA =====")

for i in range(
    min(5, len(avaliacoes_texto))
):

    print("\nOriginal:")

    print(
        avaliacoes_texto.iloc[i][
            "review_comment_message"
        ]
    )

    print("Limpo:")

    print(
        avaliacoes_texto.iloc[i][
            "texto_limpo"
        ]
    )

    print("Processado:")

    print(
        avaliacoes_texto.iloc[i][
            "texto_processado"
        ]
    )


# =========================
# SALVAR BASE TRATADA
# =========================

PASTA_PROCESSADOS.mkdir(
    parents=True,
    exist_ok=True
)

avaliacoes_texto.to_csv(
    ARQUIVO_SAIDA,
    index=False,
    encoding="utf-8"
)

print("\n===== ARQUIVO SALVO =====")

print(
    "Base tratada salva em:",
    ARQUIVO_SAIDA
)