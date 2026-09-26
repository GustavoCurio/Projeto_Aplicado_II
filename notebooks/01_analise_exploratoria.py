from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CAMINHO DO PROJETO
# =========================

# Este arquivo está dentro da pasta notebooks
# parent.parent volta para a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

PASTA_DADOS = BASE_DIR / "data" / "raw"
PASTA_REPORTS = BASE_DIR / "reports"


# =========================
# LEITURA DOS ARQUIVOS
# =========================

pedidos = pd.read_csv(
    PASTA_DADOS / "olist_orders_dataset.csv"
)

avaliacoes = pd.read_csv(
    PASTA_DADOS / "olist_order_reviews_dataset.csv"
)

itens = pd.read_csv(
    PASTA_DADOS / "olist_order_items_dataset.csv"
)

produtos = pd.read_csv(
    PASTA_DADOS / "olist_products_dataset.csv"
)

clientes = pd.read_csv(
    PASTA_DADOS / "olist_customers_dataset.csv"
)

traducao = pd.read_csv(
    PASTA_DADOS / "product_category_name_translation.csv"
)


# =========================
# TAMANHO DAS BASES
# =========================

print("\n===== TAMANHO DAS BASES =====")

print("Pedidos:", pedidos.shape)
print("Avaliações:", avaliacoes.shape)
print("Itens:", itens.shape)
print("Produtos:", produtos.shape)
print("Clientes:", clientes.shape)


# =========================
# INFORMAÇÕES DOS PEDIDOS
# =========================

print("\n===== PEDIDOS =====")

pedidos.info()

print("\nValores nulos:")

print(
    pedidos.isnull().sum()
)


# =========================
# INFORMAÇÕES DAS AVALIAÇÕES
# =========================

print("\n===== AVALIAÇÕES =====")

avaliacoes.info()

print("\nValores nulos:")

print(
    avaliacoes.isnull().sum()
)

print("\nDistribuição das notas:")

print(
    avaliacoes["review_score"]
    .value_counts()
    .sort_index()
)


# =========================
# PERÍODO DOS PEDIDOS
# =========================

pedidos["order_purchase_timestamp"] = pd.to_datetime(
    pedidos["order_purchase_timestamp"]
)

print("\n===== PERÍODO DOS PEDIDOS =====")

print(
    "Primeiro pedido:",
    pedidos["order_purchase_timestamp"].min()
)

print(
    "Último pedido:",
    pedidos["order_purchase_timestamp"].max()
)


# =========================
# STATUS DOS PEDIDOS
# =========================

print("\n===== STATUS DOS PEDIDOS =====")

print(
    pedidos["order_status"]
    .value_counts()
)


# =========================
# ANÁLISE DE ATRASOS
# =========================

pedidos["order_delivered_customer_date"] = pd.to_datetime(
    pedidos["order_delivered_customer_date"]
)

pedidos["order_estimated_delivery_date"] = pd.to_datetime(
    pedidos["order_estimated_delivery_date"]
)


# Seleciona apenas pedidos entregues
pedidos_entregues = pedidos[
    pedidos["order_status"] == "delivered"
].copy()


# Remove registros sem data real de entrega
pedidos_entregues = pedidos_entregues.dropna(
    subset=["order_delivered_customer_date"]
)


# Cria coluna informando se houve atraso
pedidos_entregues["atrasado"] = (
    pedidos_entregues["order_delivered_customer_date"]
    > pedidos_entregues["order_estimated_delivery_date"]
)


print("\n===== ATRASOS NAS ENTREGAS =====")

print(
    "Pedidos entregues analisados:",
    len(pedidos_entregues)
)

print(
    "Quantidade atrasada:",
    pedidos_entregues["atrasado"].sum()
)

print(
    "Quantidade no prazo ou antecipada:",
    (~pedidos_entregues["atrasado"]).sum()
)

print(
    "Percentual de atrasos:",
    pedidos_entregues["atrasado"].mean() * 100
)


# =========================
# RELAÇÃO ENTRE ATRASO
# E AVALIAÇÃO
# =========================

base_atraso_avaliacao = pedidos_entregues.merge(
    avaliacoes[
        [
            "order_id",
            "review_score"
        ]
    ],
    on="order_id",
    how="inner"
)


print("\n===== RELAÇÃO ENTRE ATRASO E AVALIAÇÃO =====")

print(
    "Quantidade de registros após a junção:",
    len(base_atraso_avaliacao)
)


# Nota média dos pedidos no prazo
media_no_prazo = base_atraso_avaliacao[
    base_atraso_avaliacao["atrasado"] == False
]["review_score"].mean()


# Nota média dos pedidos atrasados
media_atrasados = base_atraso_avaliacao[
    base_atraso_avaliacao["atrasado"] == True
]["review_score"].mean()


print(
    "Nota média - pedidos no prazo:",
    media_no_prazo
)

print(
    "Nota média - pedidos atrasados:",
    media_atrasados
)


# =========================
# DISTRIBUIÇÃO DAS NOTAS
# POR ATRASO
# =========================

tabela_notas = pd.crosstab(
    base_atraso_avaliacao["review_score"],
    base_atraso_avaliacao["atrasado"],
    normalize="columns"
) * 100


print("\n===== DISTRIBUIÇÃO DAS NOTAS POR ATRASO =====")

print(
    tabela_notas
)


# =========================
# GRÁFICO:
# NOTAS POR ATRASO
# =========================

tabela_notas.plot(
    kind="bar"
)

plt.title(
    "Distribuição das notas por atraso na entrega"
)

plt.xlabel(
    "Nota da avaliação"
)

plt.ylabel(
    "Percentual"
)

plt.xticks(
    rotation=0
)

plt.legend(
    [
        "No prazo",
        "Atrasado"
    ]
)

plt.tight_layout()

plt.savefig(
    PASTA_REPORTS / "distribuicao_notas_atraso.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================
# CATEGORIAS DE PRODUTOS
# =========================

base_categorias = itens.merge(
    produtos[
        [
            "product_id",
            "product_category_name"
        ]
    ],
    on="product_id",
    how="left"
)


base_categorias = base_categorias.merge(
    traducao,
    on="product_category_name",
    how="left"
)


top_categorias = (
    base_categorias[
        "product_category_name_english"
    ]
    .value_counts()
    .head(10)
)


print("\n===== CATEGORIAS COM MAIOR VOLUME DE ITENS =====")

print(
    top_categorias
)


# =========================
# GRÁFICO:
# TOP 10 CATEGORIAS
# =========================

top_categorias.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 10 categorias com maior volume de itens"
)

plt.xlabel(
    "Quantidade de itens"
)

plt.ylabel(
    "Categoria"
)

plt.tight_layout()

plt.savefig(
    PASTA_REPORTS / "top_10_categorias.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================
# PEDIDOS POR MÊS
# =========================

pedidos["ano_mes"] = (
    pedidos["order_purchase_timestamp"]
    .dt.to_period("M")
)


pedidos_por_mes = (
    pedidos["ano_mes"]
    .value_counts()
    .sort_index()
)


print("\n===== PEDIDOS POR MÊS =====")

print(
    pedidos_por_mes
)


# =========================
# GRÁFICO:
# PEDIDOS POR MÊS
# =========================

pedidos_por_mes.plot(
    kind="line",
    marker="o"
)

plt.title(
    "Quantidade de pedidos por mês"
)

plt.xlabel(
    "Ano e mês"
)

plt.ylabel(
    "Quantidade de pedidos"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    PASTA_REPORTS / "pedidos_por_mes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================
# CRIAÇÃO DE ANO E MÊS
# =========================

pedidos["ano"] = (
    pedidos["order_purchase_timestamp"]
    .dt.year
)

pedidos["mes"] = (
    pedidos["order_purchase_timestamp"]
    .dt.month
)


# =========================
# PEDIDOS POR MÊS - 2017
# =========================

pedidos_2017 = pedidos[
    pedidos["ano"] == 2017
]


pedidos_mes_2017 = (
    pedidos_2017["mes"]
    .value_counts()
    .sort_index()
)


print("\n===== PEDIDOS POR MÊS - 2017 =====")

print(
    pedidos_mes_2017
)


# =========================
# GRÁFICO:
# PEDIDOS POR MÊS - 2017
# =========================

pedidos_mes_2017.plot(
    kind="bar"
)

plt.title(
    "Quantidade de pedidos por mês - 2017"
)

plt.xlabel(
    "Mês"
)

plt.ylabel(
    "Quantidade de pedidos"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    PASTA_REPORTS / "pedidos_por_mes_2017.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================
# COMPARAÇÃO 2017 X 2018
# =========================

pedidos_ano_mes = (
    pedidos
    .groupby(
        [
            "ano",
            "mes"
        ]
    )
    .size()
    .reset_index(
        name="quantidade"
    )
)


comparacao = pedidos_ano_mes[
    pedidos_ano_mes["ano"]
    .isin(
        [
            2017,
            2018
        ]
    )
].copy()


# Usa apenas janeiro até agosto
comparacao = comparacao[
    comparacao["mes"] <= 8
]


comparacao_pivot = comparacao.pivot(
    index="mes",
    columns="ano",
    values="quantidade"
)


print("\n===== COMPARAÇÃO 2017 X 2018 =====")

print(
    comparacao_pivot
)


# =========================
# GRÁFICO:
# COMPARAÇÃO 2017 X 2018
# =========================

comparacao_pivot.plot(
    kind="line",
    marker="o"
)

plt.title(
    "Comparação mensal de pedidos - 2017 x 2018"
)

plt.xlabel(
    "Mês"
)

plt.ylabel(
    "Quantidade de pedidos"
)

plt.xticks(
    range(1, 9)
)

plt.legend(
    title="Ano"
)

plt.tight_layout()

plt.savefig(
    PASTA_REPORTS / "comparacao_2017_2018.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================
# FIM DA ANÁLISE
# =========================

print("\n===== ANÁLISE EXPLORATÓRIA FINALIZADA =====")