import json

import pandas as pd
import pytest

from olist_insights.analytics import build_evidence, filter_orders, load_orders, query_tables, summarize


def test_items_and_reviews_do_not_multiply_orders(orders_frame):
    assert len(orders_frame) == 6
    a = orders_frame.set_index("order_id").loc["a"]
    assert a["price"] == 30
    assert a["review_score"] == 5
    assert summarize(orders_frame)["valor_itens_brl"] == 70


def test_deadline_day_and_missing_dates_have_correct_denominator(orders_frame):
    metrics = summarize(orders_frame)
    assert metrics["entregas_validas"] == 2
    assert metrics["atrasados"] == 1
    assert metrics["atraso_pct"] == 50
    assert metrics["fora_do_calculo_de_atraso"] == 4
    assert not orders_frame.set_index("order_id").loc["a", "late"]
    assert pd.isna(orders_frame.set_index("order_id").loc["c", "late"])


def test_end_date_includes_entire_day_and_empty_states_mean_empty(orders_frame):
    assert filter_orders(orders_frame, "2018-01-01", "2018-01-01")["order_id"].tolist() == ["a"]
    assert filter_orders(orders_frame, states=[]).empty
    assert len(filter_orders(orders_frame, states=["SP"])) == 3


def test_invalid_scores_not_treated_as_real_reviews(orders_frame):
    metrics = summarize(orders_frame)
    assert metrics["pedidos_avaliados"] == 2
    assert metrics["nota_media"] == 3.5


def test_sql_matches_global_counts(orders_frame):
    tables = query_tables(orders_frame)
    for name in ["states", "monthly"]:
        assert tables[name]["pedidos"].sum() == 6
        assert tables[name]["entregas_validas"].sum() == 2
        assert tables[name]["atrasados"].sum() == 1
    assert dict(zip(tables["satisfaction"]["entrega"], tables["satisfaction"]["nota_media"])) == {"No prazo": 5, "Atrasado": 2}


def test_no_valid_deliveries_produce_null_not_zero(orders_frame):
    pending = orders_frame.loc[orders_frame["order_id"].eq("c")]
    assert summarize(pending)["atraso_pct"] is None
    json.dumps(build_evidence(pending), allow_nan=False)


def test_empty_evidence_is_valid_json(orders_frame):
    empty = orders_frame.iloc[:0]
    evidence = build_evidence(empty)
    assert evidence["M1"]["dados"]["pedidos"] == 0
    assert evidence["M1"]["dados"]["valor_itens_brl"] is None
    json.dumps(evidence, allow_nan=False)


def test_missing_files_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="olist_orders_dataset.csv"):
        load_orders(tmp_path)
