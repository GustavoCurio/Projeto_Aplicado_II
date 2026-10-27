# Olist — entregas, satisfação e IA aplicada

Projeto em Python que combina análise de e-commerce, classificação de avaliações e um painel interativo para explorar entregas e satisfação do cliente com dados públicos da Olist.

**Abrir o painel:** `python -m streamlit run app.py` · [Estudo de caso](docs/PORTFOLIO.md) · [Metodologia](docs/METODOLOGIA.md)

## Problema e solução

Como explorar a relação entre atrasos de entrega e satisfação sem confundir quantidade de pedidos, número de itens e avaliações repetidas?

O painel permite filtrar período e estados, comparar indicadores e consultar suas evidências. Também há um classificador para testar avaliações novas. A consulta local funciona sem API; o modo generativo é uma integração opcional, configurada no servidor.

| Componente | Funcionalidades |
| --- | --- |
| Dados e SQL | Junções validadas, um registro por pedido, indicadores e consultas SQL por estado, mês e nota. |
| Aprendizado de máquina | TF-IDF + regressão logística, inferência de texto novo e comparação com baseline. |
| Interface | Painel Streamlit com filtros, gráficos, tabelas e exportação dos indicadores. |
| Consulta local | Respostas por regras com referências aos números calculados. Não utiliza LLM. |
| IA generativa opcional | Responses API da OpenAI para redigir respostas com agregados e instruções documentadas. Requer chave e modelo. |

## Instalação e execução

Use **Python 3.12**, na raiz do repositório. As versões diretas testadas estão fixadas em requirements.txt; dependências transitivas podem variar.

```bash
python -m venv .venv
```

Ative o ambiente:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
python -m olist_insights --train
python -m streamlit run app.py
```

Abra a URL local mostrada pelo Streamlit. O comando com --train gera reports/portfolio/results.json e models/reviews.joblib. O classificador também pode ser treinado pelo painel, sem usar uma API.

### Dados

Fonte: [Brazilian E-Commerce Public Dataset by Olist, no Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Os CSVs já presentes no projeto ficam em data/raw/. Para o painel são necessários:

- olist_orders_dataset.csv
- olist_customers_dataset.csv
- olist_order_items_dataset.csv
- olist_order_reviews_dataset.csv

Consulte os termos e a licença na página da fonte antes de redistribuir ou usar os dados para outra finalidade. São dados históricos, sem conexão com a operação atual da Olist ou de outras empresas.

## Usar o painel

1. Em **Visão dos pedidos**, filtre as compras e compare atraso, nota e volume. Valor dos itens exclui frete e não equivale a lucro.
2. Em **Pergunte aos dados**, consulte “Qual é a taxa de atraso?”, “Qual estado tem mais atrasos?” ou “Qual estado tem a maior taxa de atraso?”. Quantidade e taxa usam critérios distintos.
3. Abra **Ver evidências da resposta** para conferir os indicadores M1–M4. Os filtros alteram as evidências.
4. Em **Teste o classificador**, escreva uma avaliação nova e compare a previsão com as limitações do modelo.

### Modo generativo (opcional)

Configure OPENAI_API_KEY pelo gerenciador de segredos do ambiente e OPENAI_MODEL com o nome exato de um modelo disponível na sua conta. **Não publique chaves no código, no GitHub ou no chat.** O app lê variáveis do ambiente; não carrega .env automaticamente.

```powershell
# Depois de configurar OPENAI_API_KEY no ambiente:
$env:OPENAI_MODEL = 'nome-do-modelo-na-sua-conta'
python -m streamlit run app.py
```

No painel, selecione **IA generativa (OpenAI)** e confirme o envio da pergunta e dos agregados. A chamada pode consumir créditos da API. O cliente envia store=False e não transmite identificadores, comentários de clientes ou os CSVs. Isso não implica ausência de toda retenção pelo provedor. Consulte o [prompt e seus limites](docs/PROMPTS.md).

**Validação:** painel, consulta local e classificador executados com os dados do repositório. A integração generativa é coberta por testes de contrato com cliente simulado; a avaliação com um modelo externo depende da configuração de credenciais e de testes específicos, descritos na [documentação do assistente](docs/PROMPTS.md).

## Avaliação reproduzível

Uma execução em 27/10/2026, em Python 3.12, produziu:

| Indicador | Resultado |
| --- | ---: |
| Pedidos únicos | 99.441 |
| Entregas válidas para atraso | 96.470 |
| Entregas após o dia previsto | 6.534 (6,7731%) |
| Textos únicos usados no classificador | 30.475 |
| Textos no teste | 6.095 |
| Acurácia no teste | 0,9229 |
| F1 macro no teste | 0,9152 |
| F1 macro do baseline majoritário | 0,4004 |

Textos normalizados são deduplicados **antes** da divisão treino/teste. Textos idênticos com rótulos conflitantes são excluídos; o vocabulário TF-IDF é aprendido somente no treino. O teste usa separação aleatória estratificada com semente 42 e não valida generalização temporal ou desempenho em produção.

Os números podem diferir dos scripts originais: atraso agora usa o dia previsto, há uma avaliação por pedido e o modelo tem um protocolo próprio de deduplicação. Não compare métricas de protocolos distintos como se fossem uma melhoria direta. Veja a [metodologia](docs/METODOLOGIA.md) e a [execução de exemplo](reports/portfolio/example_results.json).

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Os testes cobrem datas, junções, denominadores, filtros vazios, respostas locais, contrato da API e inferência do classificador. A CI executa os testes sem chave de API.

## Estrutura

```text
app.py                   # Interface Streamlit
olist_insights/
  analytics.py           # Preparação, indicadores e SQL fixo
  assistant.py           # Consulta local e integração generativa
  sentiment.py           # Treino, avaliação e inferência
  __main__.py            # Relatório reexecutável e treino opcional
tests/                   # Testes de dados, consultas, modelo e painel
docs/                    # Estudo de caso, metodologia e prompts
src/                     # Scripts acadêmicos originais de NLP/modelo
notebooks/               # Análise exploratória original
reports/                 # Gráficos originais e relatório demonstrativo
data/raw/                # Dataset público
```

## Sobre o projeto

O projeto surgiu na disciplina Projeto Aplicado II do curso de Ciência de Dados e evoluiu para uma aplicação interativa de análise de e-commerce. Reúne preparação de dados, consultas SQL, classificação de avaliações, respostas com evidências e testes automatizados, com apoio de IA no desenvolvimento.

O [estudo de caso](docs/PORTFOLIO.md) apresenta o problema, as decisões de implementação e os resultados. A [metodologia](docs/METODOLOGIA.md) documenta os critérios de cálculo e de avaliação dos modelos.
