# Como apresentar o projeto

## Demonstração em três minutos

1. **Problema:** explicar por que entregas e avaliações importam para o e-commerce. Mostrar que os dados são públicos e históricos.
2. **Painel:** escolher um estado e período e mostrar a mudança dos indicadores.
3. **Pergunta:** consultar a taxa de atraso, abrir as evidências e explicar o denominador. Comparar maior quantidade com maior taxa por estado.
4. **Modelo:** classificar uma avaliação nova. Explicar TF-IDF, regressão logística e a separação dos dados antes de interpretar a métrica.
5. **Limites:** dizer que é um projeto acadêmico, que a consulta local usa regras e que o modo generativo depende de configuração e avaliação com um provedor real.

## Descrição factual para o portfólio

Projeto acadêmico de análise de e-commerce com dados públicos da Olist. Inclui preparação de dados em Python, consultas SQL, painel interativo de entregas e satisfação, classificação de avaliações com TF-IDF e regressão logística e integração opcional com LLM para redigir respostas a partir de indicadores agregados. A extensão foi implementada com apoio de IA, com testes e documentação dos critérios e limitações.

Antes de usar o texto em uma candidatura, execute o projeto e ajuste a descrição ao que efetivamente revisou, compreende e consegue demonstrar. Ele não comprova implantação em empresa, usuários em produção, orquestração de múltiplos agentes ou economia de horas.

## Perguntas para estudar

- Por que somar itens antes de juntá-los com pedidos? O que aconteceria se um pedido tivesse várias cópias de sua avaliação?
- Por que um pedido sem data de entrega não deve ser contado como pontual?
- Qual a diferença entre comparar o dia previsto e o horário exato da entrega?
- Qual a diferença entre quantidade de atrasos e taxa de atraso?
- Por que deduplicar os textos antes da divisão treino/teste?
- Como TF-IDF representa o texto? O que a regressão logística aprende?
- O que o F1 macro acrescenta em relação à acurácia? Por que comparar com um baseline?
- Por que a nota é um rótulo imperfeito para sentimento?
- Que dados seguem para o LLM? Como conferir uma resposta errada?
- Por que a consulta local não deve ser chamada de agente generativo?

## Próximos passos ainda não concluídos

- Avaliar o modo generativo com os casos de PROMPTS.md, registrando modelo, custo e erros reais.
- Publicar uma demonstração na hospedagem escolhida pelo autor e configurar segredos apenas no servidor.
- Adicionar validação temporal e análise de erros do classificador.
- Testar o painel com pessoas e registrar feedback real.
- Havendo um processo manual real a substituir, medir o tempo antes e depois. Até lá, não declarar economia de horas.

Este repositório é independente de outros projetos pessoais e não requer divulgá-los.
