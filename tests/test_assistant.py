import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olist_insights.assistant import answer_question


@pytest.mark.parametrize("question, expected", [
    ("Qual é a taxa de atraso?", "50,00%"),
    ("Qual o valor dos itens?", "70,00"),
    ("Como se relacionam nota e atraso?", "não demonstra causalidade"),
    ("Qual mês tem maior volume?", "2018-01"),
    ("Qual o estado com maior taxa de atrasos?", "20 entregas"),
    ("Qual o estado com mais atrasos?", "RJ tem o maior número de atrasos: 1"),
    ("Resuma os indicadores", "6 pedidos"),
    ("Qual é o clima hoje?", "não contém dados"),
])
def test_local_answers_grounded(orders_frame, question, expected):
    answer = answer_question(question, orders_frame)
    assert expected in answer.text
    assert answer.mode == "Consulta local (sem LLM)"


def test_llm_only_receives_aggregates_and_does_not_store_response(orders_frame):
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=SimpleNamespace(output_text="50% [M1]"))))
    answer = answer_question("Qual é o atraso?", orders_frame, use_llm=True, client=client, model="modelo-de-teste")
    arguments = client.responses.create.call_args.kwargs
    payload = json.loads(arguments["input"])
    assert payload["evidencias"]["M1"]["dados"]["pedidos"] == 6
    assert "order_id" not in arguments["input"]
    assert "customer_id" not in arguments["input"]
    assert "review_comment_message" not in arguments["input"]
    assert arguments["store"] is False
    assert "[M1]" in answer.text


def test_blank_answer_and_missing_model_are_explicit(orders_frame, monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        answer_question("Resumo", orders_frame, use_llm=True)
    with pytest.raises(ValueError, match="1 a 1.500"):
        answer_question(" ", orders_frame)
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=SimpleNamespace(output_text=""))))
    with pytest.raises(ValueError, match="não retornou texto"):
        answer_question("Resumo", orders_frame, use_llm=True, client=client, model="modelo-de-teste")


def test_empty_selection_does_not_invent_indicators(orders_frame):
    answer = answer_question("Qual o lucro?", orders_frame.iloc[:0])
    assert "vazio" in answer.text
    assert "R$" not in answer.text
