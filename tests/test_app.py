from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_panel_local_question_and_empty_filter(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.metric[0].label == "Pedidos únicos"
    next(r for r in app.radio if r.label == "Área de análise").set_value("Pergunte aos dados").run()
    next(b for b in app.button if b.label == "Consultar indicadores").click().run()
    assert not app.exception
    assert any("[M1]" in item.value for item in app.markdown)
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("Nenhum pedido" in item.value for item in app.info)


def test_generative_mode_without_configuration_cannot_call_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    next(r for r in app.radio if r.label == "Área de análise").set_value("Pergunte aos dados").run()
    next(r for r in app.radio if r.label == "Modo de resposta").set_value("IA generativa (OpenAI)").run()
    assert not app.exception
    assert next(b for b in app.button if b.label == "Consultar indicadores").disabled
    assert any("OPENAI_API_KEY" in item.value for item in app.info)


def test_navigation_preserves_analytics_and_method():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 0
    next(r for r in app.radio if r.label == "Área de análise").set_value("Como foi construído").run()
    assert not app.exception
    assert any("Preparação" in m.value for m in app.markdown)
    next(r for r in app.radio if r.label == "Área de análise").set_value("Teste o classificador").run()
    assert not app.exception
    assert any("Uma avaliação" in item.value for item in app.subheader)
