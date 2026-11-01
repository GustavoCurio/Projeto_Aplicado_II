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

st.set_page_config(page_title="Olist · Commerce Analytics", page_icon="📦", layout="wide")
st.markdown("""<style>
.stApp{background:#0d1424;color:#dce5f3}
.block-container{padding-top:3.6rem;max-width:1600px}
h1,h2,h3{color:#f0f4fc;letter-spacing:-.035em}
h1{font-size:2rem!important;padding:.25rem 0!important}\n.block-container [data-testid="stVerticalBlock"]{gap:.65rem}\n.block-container hr{margin:.3rem 0}h3{font-size:1.18rem!important}
.terminal-brand{font-family:monospace;font-size:12px;letter-spacing:3px;color:#6de1b7}
.subtitle{color:#97a7bd;font-size:15px}
[data-testid="stMetric"]{background:#141f32;border:0;border-left:3px solid #6de1b7;border-radius:0;padding:8px 14px}
[data-testid="stMetricValue"]{color:#f2f5fb;font-size:26px}
[data-testid="stMetricLabel"]{color:#9caec6}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:#29364e!important}
div[data-testid="stRadio"] [role="radiogroup"]{gap:25px}
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

st.markdown('<div class="terminal-brand">OLIST / COMMERCE ANALYTICS</div>', unsafe_allow_html=True)
heading, repo = st.columns([4,1])
heading.title("Panorama da operação")
repo.link_button("Repositório ↗", "https://github.com/GustavoCurio/Projeto_Aplicado_II")
st.markdown('<p class="subtitle">Pedidos, entregas e satisfação · análise histórica de e-commerce</p>', unsafe_allow_html=True)
with st.container(border=True):
    period, geography = st.columns([1, 2])
    with period:
        dates = st.date_input("Período da compra", value=(valid_dates.min().date(), valid_dates.max().date()),
                              min_value=valid_dates.min().date(), max_value=valid_dates.max().date())
    with geography:
        states = st.multiselect("Estados", options=sorted(orders["customer_state"].unique()),
                                default=sorted(orders["customer_state"].unique()))
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
columns[0].metric("Pedidos únicos", f"{metrics['pedidos']:,}".replace(",", "."))
columns[1].metric("Entregas com atraso", f"{number(metrics['atraso_pct'])}%" if metrics["atraso_pct"] is not None else "Sem dados")
columns[2].metric("Nota média", f"{number(metrics['nota_media'])} / 5" if metrics["nota_media"] is not None else "Sem dados")
columns[3].metric("Valor dos itens", f"R$ {number(metrics['valor_itens_brl'], 0)}" if metrics["valor_itens_brl"] is not None else "Sem dados")
st.caption(f"Atraso: {metrics['atrasados']} de {metrics['entregas_validas']} entregas válidas. Nota: {metrics['pedidos_avaliados']} pedidos avaliados. Valor dos itens exclui frete; não representa lucro.")

view = st.radio("Área de análise", ["Operação", "Pergunte aos dados", "Teste o classificador", "Como foi construído"], horizontal=True, label_visibility="collapsed")
st.divider()
tables = query_tables(selected)
if view == "Operação":
    st.subheader("Pulso das compras")
    monthly = tables["monthly"]
    volume = alt.Chart(monthly).mark_area(
        line={"color":"#6de1b7","strokeWidth":2.5},
        color=alt.Gradient(gradient="linear", stops=[alt.GradientStop(color="#6de1b7",offset=0),alt.GradientStop(color="#0d1424",offset=1)], x1=1,x2=1,y1=0,y2=1)
    ).encode(
        x=alt.X("mes:N",title=None,axis=alt.Axis(labelAngle=-45,labelColor="#95a7c0",tickCount=12)),
        y=alt.Y("pedidos:Q",title="Pedidos",axis=alt.Axis(labelColor="#95a7c0",titleColor="#95a7c0",gridColor="#243149")),
        tooltip=["mes","pedidos","atrasados","atraso_pct"]
    ).properties(height=310,background="#0d1424").configure_view(strokeWidth=0)
    st.altair_chart(volume, width="stretch")
    st.caption("Contagem por mês da compra. Os meses nas bordas do período podem estar incompletos.")
    left, right = st.columns([1.25,1],gap="large")
    with left:
        st.subheader("Atrasos por estado")
        ranked = tables["states"].loc[tables["states"]["entregas_validas"].ge(20)].sort_values("atraso_pct", ascending=False).head(10)
        if ranked.empty:
            st.info("Não há estado com pelo menos 20 entregas válidas neste recorte.")
        else:
            chart = alt.Chart(ranked).mark_bar(color="#f3ad69",cornerRadiusEnd=3).encode(
                x=alt.X("atraso_pct:Q",title="Entregas com atraso (%)",axis=alt.Axis(labelColor="#95a7c0",titleColor="#95a7c0",gridColor="#243149")),
                y=alt.Y("estado:N",sort="-x",title=None,axis=alt.Axis(labelColor="#dce5f3")),
                tooltip=["estado","entregas_validas","atrasados","atraso_pct"]
            ).properties(height=280,background="#0d1424").configure_view(strokeWidth=0)
            st.altair_chart(chart,width="stretch")
        st.caption("10 maiores taxas · mínimo de 20 entregas válidas por estado.")
    with right:
        st.subheader("A entrega aparece na avaliação?")
        satisfaction = tables["satisfaction"]
        score = alt.Chart(satisfaction).mark_bar(size=48,cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("entrega:N",title=None,axis=alt.Axis(labelAngle=0,labelColor="#dce5f3")),
            y=alt.Y("nota_media:Q",scale=alt.Scale(domain=[0,5]),title="Nota média",axis=alt.Axis(labelColor="#95a7c0",titleColor="#95a7c0",gridColor="#243149")),
            color=alt.Color("entrega:N",scale=alt.Scale(domain=["No prazo","Atrasado"],range=["#6de1b7","#f3ad69"]),legend=None),
            tooltip=["entrega","pedidos_avaliados","nota_media"]
        ).properties(height=230,background="#0d1424").configure_view(strokeWidth=0)
        st.altair_chart(score,width="stretch")
        st.caption("Uma avaliação por pedido. A associação entre atraso e nota não prova causalidade.")
        with st.expander("Ver tabela de avaliações"):
            st.dataframe(satisfaction,hide_index=True,width="stretch")
    with st.expander("Conferir os números por estado"):
        st.dataframe(tables["states"],hide_index=True,width="stretch")
    st.download_button("Baixar indicadores por estado (CSV)",tables["states"].to_csv(index=False).encode("utf-8-sig"),"olist_indicadores_estados.csv",mime="text/csv")

if view == "Pergunte aos dados":
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

if view == "Teste o classificador":
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

if view == "Como foi construído":
    st.subheader("Do CSV à resposta")
    st.markdown("""
1. **Preparação:** pedidos e clientes são unidos com validação; itens são somados antes da junção. A avaliação mais recente representa cada pedido.
2. **Indicadores:** o cálculo de atraso usa só entregas com datas válidas, comparadas por dia de calendário. As consultas por estado, mês e nota usam SQL fixo.
3. **Consulta:** no modo local, regras selecionam a resposta. No modo generativo, a pergunta e agregados seguem para o LLM, sem identificadores ou textos de clientes.
4. **Classificação:** textos normalizados são deduplicados antes de dividir treino/teste; textos idênticos com rótulos conflitantes são retirados. O teste compara o modelo com uma previsão majoritária.
""")
    st.info("Estudo de e-commerce com dados públicos históricos da Olist. A metodologia documenta os critérios de cálculo dos indicadores e de avaliação das previsões.")
    st.markdown("O repositório reúne o estudo de caso, a metodologia, os testes e os códigos de análise em `src/` e `notebooks/`.")

st.divider()
st.caption("Dataset público da Olist · estudo acadêmico com dados históricos · sem conexão com operações em tempo real.")
