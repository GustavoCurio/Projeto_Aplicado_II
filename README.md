# Projeto Aplicado II - Olist

Projeto desenvolvido para a disciplina Projeto Aplicado II do curso de Ciência de Dados.

## Objetivo

Analisar dados do e-commerce brasileiro utilizando o dataset público da Olist, com foco em:

- análise exploratória dos pedidos;
- atrasos de entrega;
- relação entre atraso e satisfação do cliente;
- comportamento mensal dos pedidos;
- análise de categorias de produtos;
- tratamento de avaliações textuais;
- classificação de avaliações positivas e negativas.

## Dataset

Foi utilizado o dataset:

Brazilian E-Commerce Public Dataset by Olist

Disponível no Kaggle:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Os arquivos CSV originais devem ser colocados em:

data/raw/

## Estrutura do projeto

```text
trabalho/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_analise_exploratoria.py
├── reports/
├── src/
│   ├── tratamento.py
│   └── modelo.py
├── README.MD
└── requirements.txt