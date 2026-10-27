# Metodologia da extensão Olist

## Grão e junções

O painel mantém **uma linha por order_id**. Itens são somados antes da junção para evitar que um pedido com vários produtos seja contado várias vezes. A relação com clientes é validada como muitos-para-um. Duplicidades de order_id na tabela principal ou customer_id em clientes interrompem o carregamento em vez de inflar indicadores.

Avaliações são ordenadas por review_answer_timestamp. A resposta mais recente representa o pedido; empates são resolvidos pelo review_id. Notas fora de 1–5 viram ausentes. Esse critério é uma escolha analítica explícita.

## Indicadores e filtros

| Medida | Definição |
| --- | --- |
| Pedidos | Quantidade de pedidos únicos no recorte. |
| Recorte | Data de compra, incluindo todo o último dia, e estados selecionados. Seleção vazia de estados produz zero pedidos. |
| Entrega válida para atraso | Status delivered; datas de compra, entrega e estimativa válidas; entrega e estimativa não anteriores ao dia da compra. |
| Atraso | Dia da entrega real posterior ao dia estimado. Uma entrega às 23h do próprio dia previsto não é atraso nesta extensão. |
| Taxa de atraso | Atrasados / entregas válidas. Sem entregas válidas, o resultado é ausente, não zero. |
| Nota média | Média da avaliação selecionada por pedido, excluindo notas ausentes. |
| Valor dos itens | Soma de price, excluindo frete. Pedidos sem itens têm valor ausente. Não mede lucro, pagamentos ou receita reconhecida. |
| Ticket dos itens | Média dos valores de itens disponíveis por pedido. |
| Dias para entrega | Diferença em dias de calendário entre compra e entrega, apenas nas entregas válidas. |

Consultas fixas em SQLite calculam tabelas por estado, mês e relação entre entrega e nota. O banco é temporário, em memória. Perguntas e saídas do modelo nunca são executadas como SQL ou Python.

A comparação da **maior taxa por estado** exige ao menos 20 entregas válidas por grupo. É um corte exploratório, não um teste de significância. A consulta pela **maior quantidade de atrasos** usa contagem absoluta e informa também o denominador.

## Classificador

1. Selecionar comentários preenchidos e notas 1, 2, 4 e 5.
2. Normalizar Unicode, caixa, pontuação e espaços; preservar palavras de negação.
3. Excluir textos normalizados com menos de três caracteres e textos idênticos com rótulos conflitantes.
4. Deduplicar textos antes da separação treino/teste.
5. Separar 80%/20%, com estratificação e semente 42.
6. Aprender TF-IDF no treino, com unigramas e bigramas; treinar regressão logística balanceada.
7. Comparar com a previsão constante da classe majoritária. Registrar acurácia, F1 macro, relatório por classe e matriz de confusão.

O artefato salvo contém pipeline e métricas. A inferência aplica a mesma normalização. Um texto sem termos conhecidos não recebe classe. O escore exibido não é uma probabilidade calibrada.

## Limitações

- Notas são aproximações de sentimento; podem divergir do texto, conter ironia ou discutir vários assuntos.
- O teste aleatório não mede dados futuros. Ainda não há separação temporal, validação externa ou monitoramento de deriva.
- Textos parecidos, embora não idênticos, podem existir nas duas partições. A deduplicação reduz uma fonte de vazamento, mas não elimina todas.
- Indicadores descritivos não estabelecem causalidade entre atraso e nota.
- Não foram medidos ROI, economia de horas ou resultados de negócio.
- O modo generativo pode errar mesmo recebendo evidências corretas. Os testes de contrato não substituem avaliação de respostas reais.

## Reproduzir

```bash
python -m olist_insights --train
python -m pytest -q
```

reports/portfolio/results.json contém evidências, métricas e duração local. example_results.json registra uma execução com os dados do repositório. O tempo depende do computador e **não equivale a tempo de trabalho economizado**.
