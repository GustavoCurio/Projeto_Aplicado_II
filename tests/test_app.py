from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_panel_local_question_and_empty_filter(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.metric[0].label == "Pedidos únicos"
    app.button[0].click().run()
    assert not app.exception
    assert any("[M1]" in item.value for item in app.markdown)
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("Nenhum pedido" in item.value for item in app.info)


def test_generative_mode_without_configuration_cannot_call_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.radio[0].set_value("IA generativa (OpenAI)").run()
    assert not app.exception
    assert app.button[0].disabled
    assert any("OPENAI_API_KEY" in item.value for item in app.info)
