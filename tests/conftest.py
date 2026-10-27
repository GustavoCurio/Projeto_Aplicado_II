import pandas as pd
import pytest

from olist_insights.analytics import prepare_orders


@pytest.fixture
def orders_frame():
    orders = pd.DataFrame([
        ["a", "c1", "delivered", "2018-01-01 23:59:00", "2018-01-05 23:00:00", "2018-01-05"],
        ["b", "c2", "delivered", "2018-01-02", "2018-01-07", "2018-01-05"],
        ["c", "c3", "delivered", "2018-01-02", None, "2018-01-05"],
        ["d", "c4", "shipped", "2018-01-02", "2018-01-07", "2018-01-05"],
        ["e", "c5", "delivered", "2018-01-02", "2018-01-07", None],
        ["f", "c6", "delivered", "2018-01-02", "2017-12-20", "2018-01-05"],
    ], columns=["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
    customers = pd.DataFrame({"customer_id": [f"c{i}" for i in range(1, 7)], "customer_state": ["SP", "RJ", "SP", "RJ", "SP", "RJ"]})
    items = pd.DataFrame([["a", 10, 2], ["a", 20, 3], ["b", 40, 5]], columns=["order_id", "price", "freight_value"])
    reviews = pd.DataFrame([
        ["a", "r1", 1, "2018-01-05"], ["a", "r2", 5, "2018-01-08"],
        ["b", "r3", 2, "2018-01-08"], ["c", "r4", 9, "2018-01-08"],
    ], columns=["order_id", "review_id", "review_score", "review_answer_timestamp"])
    return prepare_orders(orders, customers, items, reviews)
