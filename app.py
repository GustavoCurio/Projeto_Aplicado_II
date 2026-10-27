"""Painel Olist: execute com python -m streamlit run app.py."""
import os
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from olist_insights.analytics import filter_orders, load_orders, query_tables, summarize
from olist_insights.assistant import answer_question, number
from olist_insights.sentiment import predict_review, train_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "reviews.joblib"

st.set_page_config(page_title="Olist | Entregas & experiência", page_icon="📦", layout="wide")
st.markdown("""<style>
    .stApp {background: #f5f7fa;}
    [data-testid="stSidebar"] {background: #eaf0f5;}
    [data-testid="stMetric"] {background: white; border-radius: 12px;}
    .eyebrow {color:#087f8c;font-size:13px;font-weight:700;letter-spacing:2px;}
    .subtitle {color:#526477;font-size:17px;max-width:850px;}
    h1, h2, h3 {color:#17324d;}
    .block-container {padding-top:4rem;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_orders():
    return load_orders(DATA_DIR)


try:
    orders = cached_orders()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.info("Confira a seção de instalação do README e os CSVs em data/raw.")
    st.stop()

valid_dates = orders["order_purchase_timestamp"].dropna()
if valid_dates.empty:
    st.error("Não há datas de compra válidas na base.")
    st.stop()

with st.sidebar:
    st.markdown("### Seu recorte")
    st.caption("Os filtros valem para os indicadores e para as respostas do assistente.")
    dates = st.date_input("Período da compra", value=(valid_dates.min().date(), valid_dates.max().date()),
                          min_value=valid_dates.min().date(), max_value=valid_dates.max().date())
    states = st.multiselect("Estados", options=sorted(orders["customer_state"].unique()),
                            default=sorted(orders["customer_state"].unique()))
    st.divider()
    st.markdown("**Dataset público da Olist**")
    st.caption("Estudo acadêmico com dados históricos. Sem conexão com operações em tempo real.")
    st.link_button("Ver o projeto no GitHub", "https://github.com/GustavoCurio/Projeto_Aplicado_II")

st.markdown('<div class="eyebrow">OLIST · LABORATÓRIO DE DADOS</div>', unsafe_allow_html=True)
st.title("Da entrega à experiência do cliente")
st.markdown('<p class="subtitle">Explore os pedidos, investigue atrasos e consulte os indicadores com evidências. Um projeto acadêmico de dados, aprendizado de máquina e IA generativa opcional.</p>', unsafe_allow_html=True)
if len(dates) != 2:
    st.info("Selecione a data inicial e a final do período.")
    st.stop()
selected = filter_orders(orders, dates[0], dates[1], states)
metrics = summarize(selected)
st.caption(f"{dates[0]:%d/%m/%Y} a {dates[1]:%d/%m/%Y} · {len(states)} estados selecionados · {len(selected):,} pedidos no recorte".replace(",", "."))
if selected.empty:
    st.info("Nenhum pedido neste recorte. Selecione pelo menos um estado ou amplie o período.")
    st.stop()

columns = st.columns(4)
columns[0].metric("Pedidos únicos", f"{metrics['pedidos']:,}".replace(",", "."), border=True)
columns[1].metric("Entregas com atraso", f"{number(metrics['atraso_pct'])}%" if metrics["atraso_pct"] is not None else "Sem dados", border=True)
columns[2].metric("Nota média", f"{number(metrics['nota_media'])} / 5" if metrics["nota_media"] is not None else "Sem dados", border=True)
columns[3].metric("Valor dos itens", f"R$ {number(metrics['valor_itens_brl'], 0)}" if metrics["valor_itens_brl"] is not None else "Sem dados", border=True)
st.caption(f"Atraso: {metrics['atrasados']} de {metrics['entregas_validas']} entregas válidas. Nota: {metrics['pedidos_avaliados']} pedidos avaliados. Valor dos itens exclui frete; não representa lucro.")

overview, assistant, classifier, method = st.tabs(["Visão dos pedidos", "Pergunte aos dados", "Teste o classificador", "Como foi construído"])
tables = query_tables(selected)
with overview:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Volume de compras ao longo do tempo")
        st.line_chart(tables["monthly"].set_index("mes")[["pedidos"]], color="#087f8c", height=270)
        st.caption("Contagem por mês da compra. Meses nas bordas do período podem estar incompletos.")
    with right:
        st.subheader("Onde investigar os atrasos")
        ranked = tables["states"].loc[tables["states"]["entregas_validas"].ge(20)].sort_values("atraso_pct", ascending=False).head(10)
        if ranked.empty:
            st.info("Não há estado com pelo menos 20 entregas válidas neste recorte.")
        else:
            chart = alt.Chart(ranked).mark_bar(color="#087f8c", cornerRadiusEnd=3).encode(
                x=alt.X("atraso_pct:Q", title="Entregas com atraso (%)"),
                y=alt.Y("estado:N", sort="-x", title=None),
                tooltip=["estado", "entregas_validas", "atrasados", "atraso_pct"],
            ).properties(height=270)
            st.altair_chart(chart, width="stretch")
        st.caption("Até 10 maiores taxas; mínimo de 20 entregas válidas por estado. Volumes diferentes exigem cuidado na comparação.")
    st.subheader("Entregas e avaliações")
    st.dataframe(tables["satisfaction"], hide_index=True, width="stretch")
    st.caption("Uma avaliação por pedido. A associação entre atraso e nota não prova causalidade.")
    with st.expander("Conferir os números por estado"):
        st.dataframe(tables["states"], hide_index=True, width="stretch")
    st.download_button("Baixar indicadores por estado (CSV)", tables["states"].to_csv(index=False).encode("utf-8-sig"),
                       "olist_indicadores_estados.csv", mime="text/csv")

with assistant:
    st.subheader("Perguntas com números que você pode conferir")
    st.caption("Exemplos: Qual é a taxa de atraso? · Qual estado tem mais atrasos? · Como o atraso se relaciona com a nota? · Qual mês teve mais pedidos?")
    configured = bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"))
    mode = st.radio("Modo de resposta", ["Consulta local (sem LLM)", "IA generativa (OpenAI)"], horizontal=True)
    use_llm = mode == "IA generativa (OpenAI)"
    if use_llm and not configured:
        st.info("O modo generativo precisa de OPENAI_API_KEY e OPENAI_MODEL configurados no servidor. Você pode usar a consulta local agora.")
    context_key = (str(dates), tuple(states), mode)
    if st.session_state.get("answer_context") != context_key:
        st.session_state.pop("last_answer", None)
        st.session_state["answer_context"] = context_key
    with st.form("question_form"):
        question = st.text_input("Sua pergunta", value="Qual é a taxa de atraso?", max_chars=1500)
        approved = True
        if use_llm:
            approved = st.checkbox("Enviar esta pergunta e os indicadores agregados para a OpenAI. A chamada pode consumir créditos da minha API.")
        submitted = st.form_submit_button("Consultar indicadores", disabled=use_llm and not configured)
    if submitted:
        if not approved:
            st.warning("Confirme o envio para usar IA generativa ou escolha a consulta local.")
        else:
            try:
                with st.spinner("Consultando o recorte selecionado…"):
                    st.session_state["last_answer"] = answer_question(question, selected, use_llm=use_llm)
            except ValueError as error:
                st.session_state.pop("last_answer", None)
                st.error(str(error))
            except Exception:
                st.session_state.pop("last_answer", None)
                st.error("Não foi possível consultar o provedor. Confira a configuração e os créditos da API, ou use o modo local.")
    if "last_answer" in st.session_state:
        answer = st.session_state["last_answer"]
        with st.chat_message("assistant"):
            st.caption(answer.mode)
            st.markdown(answer.text)
        with st.expander("Ver evidências da resposta"):
            st.json(answer.evidence)
    st.caption("A consulta local usa regras e SQL fixo. No modo generativo, o LLM redige a resposta a partir dos mesmos agregados; confira as evidências, pois ele pode errar. Nenhum comentário de cliente é enviado.")

with classifier:
    st.subheader("Uma avaliação, duas classes")
    st.write("Modelo TF-IDF + regressão logística treinado para distinguir avaliações positivas e negativas. Notas 1–2 representam a classe negativa; 4–5, a positiva. Nota 3 não entra no treino.")
    st.caption("Este modelo usa a base histórica completa de avaliações; os filtros do painel não alteram o treinamento.")
    if st.button("Treinar e avaliar o modelo localmente"):
        with st.spinner("Separando textos únicos, treinando e avaliando…"):
            reviews = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv", usecols=["review_score", "review_comment_message"])
            train_model(reviews, MODEL_PATH)
        st.success("Modelo treinado. A avaliação usa textos separados do treino.")
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        model_metrics = artifact["metrics"]
        a, b, c = st.columns(3)
        a.metric("F1 macro · teste", number(model_metrics["f1_macro"], 3))
        b.metric("F1 macro · baseline", number(model_metrics["baseline_f1_macro"], 3))
        c.metric("Textos no teste", model_metrics["teste"])
        with st.form("review_form"):
            review = st.text_area("Escreva uma avaliação para testar", max_chars=3000, placeholder="Ex.: O produto chegou antes do prazo e em boas condições.")
            predict = st.form_submit_button("Classificar avaliação")
        if predict:
            try:
                result = predict_review(artifact, review)
                if result["classe"] is None:
                    st.info(result["aviso"])
                else:
                    st.write(f"**Classe estimada: {result['classe']}** · escore {number(result['escore'], 3)}")
                    st.caption(result["aviso"])
            except ValueError as error:
                st.warning(str(error))
        with st.expander("Conferir a avaliação do modelo"):
            st.json(model_metrics)
    else:
        st.info("Treine o modelo pelo botão acima ou execute python -m olist_insights --train antes de abrir o painel.")

with method:
    st.subheader("Do CSV à resposta")
    st.markdown("""
1. **Preparação:** pedidos e clientes são unidos com validação; itens são somados antes da junção. A avaliação mais recente representa cada pedido.
2. **Indicadores:** o cálculo de atraso usa só entregas com datas válidas, comparadas por dia de calendário. As consultas por estado, mês e nota usam SQL fixo.
3. **Consulta:** no modo local, regras selecionam a resposta. No modo generativo, a pergunta e agregados seguem para o LLM, sem identificadores ou textos de clientes.
4. **Classificação:** textos normalizados são deduplicados antes de dividir treino/teste; textos idênticos com rótulos conflitantes são retirados. O teste compara o modelo com uma previsão majoritária.
""")
    st.info("Estudo de e-commerce com dados públicos históricos da Olist. A metodologia documenta os critérios de cálculo dos indicadores e de avaliação das previsões.")
    st.markdown("O repositório reúne o estudo de caso, a metodologia, os testes e os códigos de análise em `src/` e `notebooks/`.")
