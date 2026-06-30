# Classificação de Potabilidade da Água

## Aprendizado de Máquina Aplicado à Qualidade Hídrica

**Disciplina:** Tópicos Especiais em Computação I (2026.1)
**Instituição:** Universidade Federal do Ceará — Campus Sobral
**Curso:** Engenharia de Computação
**Autor:** Dominique Kossi

---

# Slide 1 — Introdução ao Problema

## Por que classificar potabilidade da água?

- A Organização Mundial da Saúde estima que 2 bilhões de pessoas utilizam fontes de água contaminada
- A análise manual de parâmetros físico-químicos é custosa e demorada
- Modelos de Machine Learning podem automatizar a triagem de potabilidade com base em atributos mensuráveis

## Objetivo do Projeto

Construir um pipeline completo de Machine Learning para classificação binária (potável vs. não potável) utilizando 9 parâmetros físico-químicos da água.

---

# Slide 2 — Dataset Water Potability

## Características do Dataset

| Propriedade         | Valor                     |
| ------------------- | ------------------------- |
| Fonte               | Kaggle (Water Potability) |
| Total de amostras   | 3.276                     |
| Atributos numéricos | 9                         |
| Variável alvo       | Potability (0 ou 1)       |
| Classe majoritária  | Não potável (61%)         |
| Classe minoritária  | Potável (39%)             |

## Atributos

pH, Dureza, Sólidos Dissolvidos, Cloraminas, Sulfato, Condutividade, Carbono Orgânico, Trihalometanos, Turbidez

## Desafios Identificados

- Desbalanceamento de classes (61% vs 39%)
- Valores ausentes em 3 atributos (pH: 15%, Sulfato: 24%, Trihalometanos: 5%)

---

# Slide 3 — Arquitetura do Pipeline

## Pipeline Modular em 5 Etapas

```
main.py (Orquestrador)
    │
    ├── 1. Coleta e Validação
    ├── 2. Análise Exploratória (EDA)
    ├── 3. Preprocessamento
    ├── 4. Treinamento de Modelos
    └── 5. Avaliação e Comparação
```

## Decisões Arquiteturais

- Execução com comando único: `python main.py`
- Módulos independentes em `src/`
- Configuração centralizada via YAML
- Rastreamento de experimentos com MLflow
- Seeds fixas para reprodutibilidade total

---

# Slide 4 — Análise Exploratória de Dados

## Estatísticas Descritivas

- pH varia de 0 a 14 (distribuição aproximadamente normal)
- Sólidos dissolvidos apresentam alta variância (320 a 61.227 ppm)
- Nenhuma correlação forte entre atributos (max: 0.08)

## Valores Ausentes

| Atributo       | Ausentes | Percentual |
| -------------- | -------- | ---------- |
| Sulfato        | 781      | 23,84%     |
| pH             | 491      | 14,99%     |
| Trihalometanos | 162      | 4,95%      |

## Visualizações Geradas

- Histogramas de distribuição por atributo
- Matriz de correlação (heatmap)
- Distribuição da variável alvo

---

# Slide 5 — Preprocessamento

## Pipeline de Preprocessamento (4 etapas)

### 1. Imputação de Valores Ausentes

- Estratégia: mediana (robusta a outliers)
- 1.434 valores imputados no total

### 2. Divisão Treino/Teste

- 80% treino (2.620 amostras) / 20% teste (656 amostras)
- Estratificação pela variável alvo

### 3. Escalonamento

- StandardScaler ajustado exclusivamente no treino
- Prevenção de data leakage

### 4. Balanceamento de Classes

- SMOTE aplicado apenas no treino
- Resultado: razão 1:1 (1.598 amostras por classe)
- Treino final: 3.196 amostras

---

# Slide 6 — Modelos de Classificação

## Algoritmos Selecionados

| Modelo              | Justificativa                              |
| ------------------- | ------------------------------------------ |
| Logistic Regression | Baseline linear, interpretável             |
| Random Forest       | Ensemble robusto, captura não-linearidades |
| XGBoost             | Gradient boosting de alto desempenho       |

## Otimização de Hiperparâmetros

- Método: GridSearchCV
- Validação cruzada: 5 folds estratificados
- Métrica de otimização: F1-Score
- Busca exaustiva em grids predefinidos

## Melhores Hiperparâmetros Encontrados

- **Logistic Regression:** C=0.01, solver=liblinear, penalty=l2
- **Random Forest:** max_depth=None, min_samples_leaf=2, n_estimators=100
- **XGBoost:** learning_rate=0.1, max_depth=7, n_estimators=300, subsample=0.8

---

# Slide 7 — Resultados: Métricas de Avaliação

## Relatório Comparativo (Conjunto de Teste)

| Modelo              | Accuracy   | Precision  | Recall     | F1-Score   | AUC-ROC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **Random Forest**   | **0.6402** | **0.5490** | 0.4375     | **0.4870** | **0.6594** |
| XGBoost             | 0.6280     | 0.5275     | 0.4492     | 0.4852     | 0.6491     |
| Logistic Regression | 0.5320     | 0.4220     | **0.5391** | 0.4734     | 0.5469     |

## Melhor Modelo: Random Forest

- Maior F1-Score (0.4870) e AUC-ROC (0.6594)
- Melhor equilíbrio entre precisão e recall
- Critério de desempate: AUC-ROC

---

# Slide 8 — Análise dos Resultados

## Interpretação das Métricas

- **Random Forest** obteve melhor desempenho geral com F1=0.487
- **Logistic Regression** teve o melhor Recall (0.539) mas baixa Precision
- **XGBoost** ficou próximo do Random Forest em todas as métricas

## Por que F1-Score moderado?

- Dataset com baixa separabilidade entre classes
- Atributos com correlação fraca com a variável alvo (max 0.08)
- Problema inerentemente difícil: parâmetros físico-químicos isolados não determinam potabilidade com alta confiança
- Resultados consistentes com a literatura para este dataset

## Visualizações

- Matrizes de confusão por modelo
- Curvas ROC comparativas

---

# Slide 9 — Rastreamento com MLflow

## Funcionalidades Implementadas

- Experimento nomeado: "water-potability-classification"
- Registro automático de hiperparâmetros, métricas e artefatos
- Versionamento de bibliotecas e timestamp ISO 8601
- Interface web para comparação visual de runs

## Benefícios para Reprodutibilidade

- Todos os parâmetros registrados permitem reconstrução exata
- Seeds fixas garantem resultados idênticos entre execuções
- Fallback automático para armazenamento local se servidor indisponível

## Acesso

```bash
mlflow ui --port 5000
```

---

# Slide 10 — Testes e Qualidade de Código

## Abordagem Dual de Testes

### Testes de Propriedade (Hypothesis)

- 12 propriedades formais de corretude
- Geração aleatória de dados para validação universal
- Exemplos: completude da imputação, validade das métricas, ordenação do relatório

### Testes Unitários (pytest)

- Cenários específicos e edge cases
- Tratamento de erros e resiliência

## Resultados

- **31 testes** no total, todos passando
- Cobertura das 5 etapas do pipeline
- Docstrings em português (formato Google)

---

# Slide 11 — Estrutura do Repositório

## Organização do Código

```
├── main.py                 Orquestrador (ponto de entrada)
├── configs/config.yaml     Configurações centralizadas
├── src/
│   ├── data_collection.py  Coleta e validação
│   ├── eda.py              Análise exploratória
│   ├── preprocessing.py    Preprocessamento
│   ├── training.py         Treinamento
│   ├── evaluation.py       Avaliação
│   ├── mlflow_utils.py     Integração MLflow
│   ├── logger.py           Logging padronizado
│   └── types.py            Contratos de dados
├── tests/                  31 testes automatizados
├── models/                 Modelos salvos (.joblib)
└── reports/figures/        Visualizações (PNG)
```

## Boas Práticas Adotadas

- Código modular e desacoplado
- Type hints e TypedDicts
- Logging estruturado por nível (INFO, WARNING, ERROR, CRITICAL)
- Tratamento de erros resiliente (falha individual não interrompe pipeline)

---

# Slide 12 — Demonstração

## Execução Completa

```bash
git clone https://github.com/dominiquekossi/water-potability.git
cd water-potability
pip install -r requirements.txt
python main.py
```

## Tempo de Execução

- Pipeline completo: ~5 minutos
- Etapa mais longa: treinamento do Random Forest (~4.5 min)

## Artefatos Gerados

- 9 histogramas de distribuição
- 1 matriz de correlação
- 3 matrizes de confusão
- 3 curvas ROC
- 3 modelos salvos em joblib
- Relatório comparativo CSV
- Métricas registradas no MLflow

---

# Slide 13 — Conclusões

## Contribuições do Projeto

1. Pipeline completo e reprodutível de classificação de potabilidade
2. Comparação sistemática de 3 algoritmos com busca de hiperparâmetros
3. Tratamento rigoroso de desbalanceamento e valores ausentes
4. Rastreamento de experimentos com MLflow
5. Suite de testes com propriedades formais de corretude

## Limitações

- F1-Score moderado (~0.49) devido à baixa separabilidade do dataset
- Apenas 9 atributos disponíveis para classificação
- Dataset relativamente pequeno (3.276 amostras)

## Trabalhos Futuros

- Feature engineering (interações entre atributos)
- Ensemble de modelos (stacking)
- Aumento do dataset com fontes complementares
- Deploy do modelo como API REST

---

# Slide 14 — Referências

1. Dataset Water Potability — Kaggle
   https://www.kaggle.com/datasets/adityakadiwal/water-potability

2. Scikit-learn: Machine Learning in Python — Pedregosa et al., JMLR 12, 2011

3. XGBoost: A Scalable Tree Boosting System — Chen & Guestrin, KDD 2016

4. SMOTE: Synthetic Minority Over-sampling Technique — Chawla et al., JAIR 16, 2002

5. MLflow: A Machine Learning Lifecycle Platform — Zaharia et al., 2018

6. Hypothesis: Property-based testing for Python
   https://hypothesis.readthedocs.io/

---

# Obrigado!

## Perguntas?

**Repositório:** https://github.com/dominiquekossi/water-potability

**Contato:** houessoudominique@gmail.com
