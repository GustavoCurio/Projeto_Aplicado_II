"""Indicadores no grão de um pedido, com denominadores explícitos."""
from pathlib import Path
import sqlite3

import pandas as pd

DATE_COLUMNS = [
    "order_purchase_timestamp", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

QUERIES = {
    "states": """SELECT customer_state AS estado, COUNT(*) AS pedidos,
        SUM(delivery_eligible) AS entregas_validas, SUM(late) AS atrasados,
        ROUND(100.0 * SUM(late) / NULLIF(SUM(delivery_eligible), 0), 2) AS atraso_pct,
        ROUND(AVG(review_score), 2) AS nota_media
        FROM orders GROUP BY customer_state ORDER BY pedidos DESC, estado""",
    "monthly": """SELECT substr(order_purchase_timestamp, 1, 7) AS mes,
        COUNT(*) AS pedidos, SUM(delivery_eligible) AS entregas_validas,
        SUM(late) AS atrasados,
        ROUND(100.0 * SUM(late) / NULLIF(SUM(delivery_eligible), 0), 2) AS atraso_pct
        FROM orders WHERE order_purchase_timestamp IS NOT NULL
        GROUP BY mes ORDER BY mes""",
    "satisfaction": """SELECT CASE late WHEN 1 THEN 'Atrasado'
        ELSE 'No prazo' END AS entrega, COUNT(*) AS pedidos_avaliados,
        ROUND(AVG(review_score), 2) AS nota_media
        FROM orders WHERE delivery_eligible = 1 AND review_score IS NOT NULL
        GROUP BY late ORDER BY late""",
}


def load_orders(data_dir: str | Path) -> pd.DataFrame:
    """Carrega só as colunas usadas e evita multiplicar pedidos em junções."""
    directory = Path(data_dir)
    files = {
        "orders": ("olist_orders_dataset.csv", ["order_id", "customer_id", "order_status", *DATE_COLUMNS]),
        "customers": ("olist_customers_dataset.csv", ["customer_id", "customer_state"]),
        "items": ("olist_order_items_dataset.csv", ["order_id", "price", "freight_value"]),
        "reviews": ("olist_order_reviews_dataset.csv", ["order_id", "review_id", "review_score", "review_answer_timestamp"]),
    }
    missing = [filename for filename, _ in files.values() if not (directory / filename).exists()]
    if missing:
        raise FileNotFoundError("Arquivos ausentes em data/raw: " + ", ".join(missing))
    tables = {name: pd.read_csv(directory / filename, usecols=columns)
              for name, (filename, columns) in files.items()}
    return prepare_orders(**tables)


def prepare_orders(orders, customers, items, reviews) -> pd.DataFrame:
    """Escolhe a avaliação respondida mais recentemente por pedido.

    Empates são resolvidos por review_id. Valor de itens exclui frete e não é
    receita reconhecida. Entrega é comparada por dia de calendário.
    """
    frame = orders.copy()
    if frame["order_id"].isna().any() or frame["order_id"].duplicated().any():
        raise ValueError("A base de pedidos deve ter order_id único e preenchido.")
    for column in DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce", format="mixed")
    reviews = reviews.copy()
    reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"], errors="coerce", format="mixed")
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")
    reviews.loc[~reviews["review_score"].between(1, 5), "review_score"] = float("nan")
    reviews = reviews.sort_values(
        ["order_id", "review_answer_timestamp", "review_id"], na_position="first"
    ).drop_duplicates("order_id", keep="last")
    items = items.copy()
    for column in ["price", "freight_value"]:
        items[column] = pd.to_numeric(items[column], errors="coerce")
    totals = items.groupby("order_id")[["price", "freight_value"]].sum(min_count=1).reset_index()
    frame = frame.merge(customers, on="customer_id", how="left", validate="many_to_one")
    frame = frame.merge(totals, on="order_id", how="left", validate="one_to_one")
    frame = frame.merge(reviews[["order_id", "review_score"]], on="order_id", how="left", validate="one_to_one")
    frame["customer_state"] = frame["customer_state"].fillna("Não informado")
    purchased = frame[DATE_COLUMNS[0]].dt.normalize()
    delivered = frame[DATE_COLUMNS[1]].dt.normalize()
    estimated = frame[DATE_COLUMNS[2]].dt.normalize()
    frame["delivery_eligible"] = (
        frame["order_status"].eq("delivered") & delivered.notna()
        & estimated.notna() & purchased.notna()
        & delivered.ge(purchased) & estimated.ge(purchased)
    )
    frame["late"] = delivered.gt(estimated).astype("boolean").where(frame["delivery_eligible"])
    frame["delivery_days"] = (delivered - purchased).dt.days.where(frame["delivery_eligible"])
    return frame


def filter_orders(frame, start=None, end=None, states=None):
    selected = frame.copy()
    purchased = selected["order_purchase_timestamp"].dt.normalize()
    if start is not None:
        selected = selected.loc[purchased.ge(pd.Timestamp(start).normalize())]
    if end is not None:
        selected = selected.loc[selected["order_purchase_timestamp"].dt.normalize().le(pd.Timestamp(end).normalize())]
    if states is not None:
        selected = selected.loc[selected["customer_state"].isin(states)]
    return selected.copy()


def optional_number(value):
    return None if pd.isna(value) else round(float(value), 4)


def summarize(frame):
    eligible = frame.loc[frame["delivery_eligible"]]
    late_count = int(eligible["late"].fillna(False).sum())
    period = frame["order_purchase_timestamp"]
    return {
        "pedidos": len(frame), "entregas_validas": len(eligible),
        "fora_do_calculo_de_atraso": len(frame) - len(eligible),
        "atrasados": late_count,
        "atraso_pct": round(100 * late_count / len(eligible), 4) if len(eligible) else None,
        "nota_media": optional_number(frame["review_score"].mean()),
        "pedidos_avaliados": int(frame["review_score"].notna().sum()),
        "valor_itens_brl": optional_number(frame["price"].sum(min_count=1)),
        "ticket_itens_brl": optional_number(frame["price"].mean()),
        "pedidos_com_valor_itens": int(frame["price"].notna().sum()),
        "dias_entrega_media": optional_number(eligible["delivery_days"].mean()),
        "inicio": period.min().strftime("%Y-%m-%d") if period.notna().any() else None,
        "fim": period.max().strftime("%Y-%m-%d") if period.notna().any() else None,
        "estados": sorted(frame["customer_state"].unique().tolist()),
    }


def query_tables(frame):
    """SQL fixo; nenhuma pergunta ou saída do LLM é executada como código."""
    columns = ["customer_state", "order_purchase_timestamp", "delivery_eligible", "late", "review_score"]
    export = frame[columns].copy()
    export["order_purchase_timestamp"] = export["order_purchase_timestamp"].dt.strftime("%Y-%m-%d")
    export["late"] = export["late"].astype("Int64")
    with sqlite3.connect(":memory:") as connection:
        export.to_sql("orders", connection, index=False)
        return {name: pd.read_sql_query(sql, connection) for name, sql in QUERIES.items()}


def build_evidence(frame):
    metrics = summarize(frame)
    tables = query_tables(frame)
    return {
        "M1": {"fonte": "Pedidos únicos no recorte; métricas calculadas em Python", "dados": metrics},
        "M2": {"fonte": "SQL por estado", "dados": records(tables["states"])},
        "M3": {"fonte": "SQL por mês da compra", "dados": records(tables["monthly"])},
        "M4": {"fonte": "SQL: entrega e nota, só pedidos avaliados com entrega válida", "dados": records(tables["satisfaction"])},
        "regras": [
            "Atraso: entrega após o dia previsto, só entregas com datas válidas.",
            "Uma avaliação por pedido: resposta mais recente; desempate por review_id.",
            "Valor de itens não inclui frete e não equivale a lucro ou receita reconhecida.",
            "Associação entre atraso e nota não demonstra causalidade.",
            "Dados históricos públicos da Olist; não descrevem a operação atual de nenhuma empresa.",
        ],
    }


def records(frame):
    # JSON puro, sem NaN/NA, inclusive quando um grupo não tem entregas válidas.
    import json
    return json.loads(frame.to_json(orient="records", force_ascii=False))
