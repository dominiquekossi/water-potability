# Classificação de Potabilidade da Água

## Descrição

Projeto de Ciência de Dados desenvolvido para a disciplina **Tópicos Especiais em Computação I (2026.1)** da Universidade Federal do Ceará, Campus Sobral, curso de Engenharia de Computação.

O objetivo é construir e avaliar modelos de classificação supervisionada para determinar se uma amostra de água é potável ou não, utilizando o dataset [Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability) do Kaggle. O sistema implementa um pipeline completo de Machine Learning com rastreamento de experimentos via MLflow.

## Autores

| Nome                          | Matrícula |
| ----------------------------- | --------- |
| Kossi Sedjro Mawuli Dominique | 422707    |

## Pré-requisitos

- Python >= 3.10
- pip (gerenciador de pacotes)

## Instalação

```bash
git clone https://github.com/dominiquekossi/water-potability.git
cd water-potability
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Dataset

O dataset Water Potability está incluído no repositório (`water_potability.csv`) e contém 3276 amostras com 9 atributos físico-químicos (pH, Dureza, Sólidos, Cloraminas, Sulfato, Condutividade, Carbono Orgânico, Trihalometanos, Turbidez) e a variável alvo binária Potability (0 = não potável, 1 = potável).

Fonte: [Water Potability (Kaggle)](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

## Execução

```bash
python main.py
```

Este comando executa sequencialmente:

1. Coleta e validação dos dados
2. Análise Exploratória de Dados (EDA)
3. Preprocessamento (imputação, escalonamento, balanceamento)
4. Treinamento de modelos com busca de hiperparâmetros
5. Avaliação e comparação dos modelos

## MLflow UI

Para visualizar os experimentos, métricas e artefatos:

```bash
mlflow ui --port 5000
```

Acesse: http://localhost:5000

## Estrutura do Projeto

```
├── main.py                     Ponto de entrada do pipeline
├── requirements.txt            Dependências com versões fixadas
├── configs/
│   └── config.yaml             Configurações do pipeline
├── src/
│   ├── data_collection.py      Coleta e validação do dataset
│   ├── eda.py                  Análise exploratória de dados
│   ├── preprocessing.py        Imputação, escalonamento, balanceamento
│   ├── training.py             Treinamento e otimização de modelos
│   ├── evaluation.py           Avaliação e relatório comparativo
│   ├── mlflow_utils.py         Integração com MLflow
│   ├── logger.py               Configuração de logging
│   └── types.py                Tipos de dados compartilhados
├── tests/                      Testes unitários e de propriedade
├── models/                     Modelos treinados (.joblib)
├── reports/figures/            Visualizações geradas (PNG)
├── data/                       Dataset (não versionado)
└── notebooks/                  Notebooks exploratórios
```

## Resultados

| Modelo              | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Random Forest       | 0.6402   | 0.5490    | 0.4375 | 0.4870   | 0.6594  |
| XGBoost             | 0.6280   | 0.5275    | 0.4492 | 0.4852   | 0.6491  |
| Logistic Regression | 0.5320   | 0.4220    | 0.5391 | 0.4734   | 0.5469  |

**Melhor modelo:** Random Forest (F1-Score: 0.4870, AUC-ROC: 0.6594)

### Configurações de Preprocessamento

| Parâmetro            | Valor                     |
| -------------------- | ------------------------- |
| Imputação            | Mediana                   |
| Escalonamento        | StandardScaler            |
| Balanceamento        | SMOTE (razão 1:1)         |
| Divisão treino/teste | 80% / 20% (estratificada) |
| Random state         | 42                        |

## Testes

```bash
pytest tests/ -v
```

## Tecnologias

- Python 3.10+
- pandas, scikit-learn, XGBoost
- imbalanced-learn (SMOTE)
- matplotlib, seaborn
- MLflow
- Hypothesis (testes de propriedade)
- pytest
