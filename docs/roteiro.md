# Roteiro de Apresentação — Classificação de Potabilidade da Água

## Instruções Gerais

- Tempo total estimado: 15 a 20 minutos
- Fale com confiança e domine os números (métricas, tamanhos, percentuais)
- Mantenha contato visual com a plateia
- Cada slide tem pontos-chave para enfatizar e possíveis perguntas do avaliador

---

## Slide 1 — Introdução ao Problema (1 min)

### O que falar:

"Boa tarde. Hoje vou apresentar o projeto de classificação de potabilidade da água usando Machine Learning."

"A motivação é simples: 2 bilhões de pessoas no mundo usam fontes de água potencialmente contaminadas. Analisar cada amostra manualmente é caro e demorado. O nosso objetivo é automatizar essa triagem usando 9 parâmetros físico-químicos da água."

"O projeto implementa um pipeline completo — da coleta de dados até a avaliação dos modelos — executável com um único comando."

### Pontos-chave:

- Enfatizar a relevância prática do problema
- Mencionar que é end-to-end (pipeline completo)
- Usar "classificação binária" (potável vs. não potável)

---

## Slide 2 — Dataset Water Potability (1.5 min)

### O que falar:

"Utilizamos o dataset Water Potability do Kaggle com 3.276 amostras e 9 atributos físico-químicos como pH, dureza, sólidos dissolvidos, sulfato, entre outros."

"Dois desafios importantes: primeiro, o desbalanceamento — 61% das amostras são não potáveis e apenas 39% são potáveis. Segundo, temos valores ausentes em 3 colunas, sendo sulfato com quase 24% de dados faltantes."

"Esses desafios guiaram nossas decisões de preprocessamento, como veremos adiante."

### Pontos-chave:

- Citar os números exatos de desbalanceamento (61/39)
- Destacar que valores ausentes afetam 3 dos 9 atributos
- Isso justifica o uso de SMOTE e imputação

### Possível pergunta do avaliador:

- "Por que não remover amostras com valores ausentes?" → "Perderíamos até 24% dos dados, reduzindo significativamente o dataset já pequeno. A imputação preserva todas as amostras."

---

## Slide 3 — Arquitetura do Pipeline (1.5 min)

### O que falar:

"A arquitetura foi projetada para ser modular e reprodutível. O ponto de entrada é o main.py que orquestra 5 etapas sequenciais."

"Cada módulo em src/ é independente e testável isoladamente. As configurações ficam centralizadas em um arquivo YAML, e todos os experimentos são rastreados pelo MLflow."

"Uma decisão importante: usamos seeds fixas em todas as operações estocásticas. Isso garante que qualquer pessoa que execute o pipeline obtenha exatamente os mesmos resultados numéricos."

### Pontos-chave:

- Enfatizar a modularidade (cada módulo tem responsabilidade única)
- Reprodutibilidade como princípio de design
- Comando único para execução

### Possível pergunta do avaliador:

- "Como garantem reprodutibilidade?" → "Seeds fixas (random_state=42) em split, SMOTE, GridSearch. MLflow registra todos os parâmetros. O requirements.txt tem versões exatas."

---

## Slide 4 — Análise Exploratória de Dados (1.5 min)

### O que falar:

"Na EDA, geramos estatísticas descritivas para todos os 9 atributos e produzimos visualizações automáticas."

"Um achado importante: a correlação entre os atributos e a variável alvo é muito baixa — no máximo 0.08. Isso já nos indica que o problema é inerentemente difícil e que não devemos esperar acurácias altíssimas."

"Identificamos os valores ausentes por coluna e geramos histogramas, matriz de correlação e gráfico de distribuição da variável alvo. Tudo é salvo como PNG e registrado no MLflow."

### Pontos-chave:

- Correlação fraca (max 0.08) — prepara o avaliador para métricas moderadas
- Todas as visualizações são automatizadas e salvas
- EDA serve como base para decisões de preprocessamento

---

## Slide 5 — Preprocessamento (2 min)

### O que falar:

"O preprocessamento segue 4 etapas rigorosas."

"Primeiro, imputação pela mediana — escolhemos mediana por ser robusta a outliers, diferente da média. Imputamos 1.434 valores no total."

"Segundo, dividimos 80/20 com estratificação. A divisão acontece ANTES do balanceamento — isso é fundamental para evitar data leakage."

"Terceiro, StandardScaler ajustado APENAS no treino. Os parâmetros de média e desvio padrão vêm exclusivamente do treino. Isso previne que informação do teste vaze para o modelo."

"Quarto, SMOTE no treino para balancear as classes em razão 1:1. O conjunto de teste permanece intacto."

### Pontos-chave:

- ENFATIZAR a ordem: split antes do balanceamento
- ENFATIZAR: scaler fit apenas no treino
- SMOTE apenas no treino, teste intocado
- Esses são os erros mais comuns em projetos de ML — mostrar que vocês sabem evitar

### Possível pergunta do avaliador:

- "O que acontece se aplicar SMOTE antes do split?" → "Data leakage. Amostras sintéticas baseadas no teste contaminam o treino, inflando métricas artificialmente."
- "Por que mediana e não média?" → "A mediana é robusta a outliers. No nosso dataset, atributos como Sólidos e Sulfato têm alta variância."

---

## Slide 6 — Modelos de Classificação (1.5 min)

### O que falar:

"Treinamos 3 classificadores: Logistic Regression como baseline linear, Random Forest como ensemble robusto, e XGBoost como método de gradient boosting."

"Para cada modelo, fizemos GridSearchCV com 5 folds estratificados, otimizando pelo F1-Score. Escolhemos F1 porque é a métrica mais adequada para datasets desbalanceados — ela combina precisão e recall."

"O treinamento é resiliente: se um modelo falha, os demais continuam normalmente. Todos os hiperparâmetros e modelos são registrados no MLflow."

### Pontos-chave:

- Justificar a escolha do F1-Score (não accuracy!)
- 3 modelos de complexidade crescente
- GridSearchCV com validação cruzada garante generalização

### Possível pergunta do avaliador:

- "Por que F1 e não Accuracy?" → "Com classes desbalanceadas, um modelo que prevê sempre 'não potável' teria 61% de accuracy mas seria inútil. F1 penaliza modelos com baixo recall."

---

## Slide 7 — Resultados: Métricas (2 min)

### O que falar:

"Aqui temos o relatório comparativo no conjunto de teste — dados que o modelo nunca viu durante o treinamento."

"O Random Forest obteve o melhor F1-Score de 0.487 e AUC-ROC de 0.659. XGBoost ficou muito próximo com F1 de 0.485. A Logistic Regression ficou abaixo, como esperado para um modelo linear."

"O critério de seleção é: maior F1-Score, com desempate por AUC-ROC. O Random Forest vence em ambos."

"Importante notar que o Logistic Regression teve o melhor Recall (0.539), ou seja, detecta mais amostras potáveis, mas com muitos falsos positivos."

### Pontos-chave:

- Citar números com segurança
- Explicar o trade-off precision vs recall
- Random Forest vence no equilíbrio geral

---

## Slide 8 — Análise dos Resultados (1.5 min)

### O que falar:

"Por que o F1-Score ficou em torno de 0.49? Isso é esperado para este dataset."

"A correlação máxima entre qualquer atributo e a potabilidade é de apenas 0.08. Na prática, isso significa que os parâmetros físico-químicos isoladamente não determinam potabilidade com alta confiança."

"Nossos resultados são consistentes com outros trabalhos na literatura que utilizam este mesmo dataset — F1-Scores entre 0.45 e 0.55 são típicos."

"O valor está na metodologia rigorosa: preprocessamento correto, validação cruzada, prevenção de data leakage, e reprodutibilidade garantida."

### Pontos-chave:

- Antecipar a crítica de "métricas baixas"
- Justificar com a natureza do dataset (baixa separabilidade)
- O mérito está na metodologia, não apenas nos números

### Possível pergunta do avaliador:

- "Como melhorar esses resultados?" → "Feature engineering (interações entre pH e cloraminas, por exemplo), ensemble stacking, ou dados adicionais de outras fontes."

---

## Slide 9 — Rastreamento com MLflow (1 min)

### O que falar:

"Integramos o MLflow para rastreamento completo dos experimentos. Cada execução registra parâmetros, métricas e artefatos."

"Isso permite comparar diferentes configurações — por exemplo, testar KNN ao invés de mediana para imputação e ver imediatamente o impacto nas métricas."

"Implementamos também um fallback: se o MLflow não estiver disponível, o pipeline continua normalmente salvando dados localmente."

"Para acessar a interface, basta executar mlflow ui --port 5000."

### Pontos-chave:

- MLflow como ferramenta profissional de gestão de experimentos
- Fallback resiliente (pipeline nunca quebra por causa do MLflow)

---

## Slide 10 — Testes e Qualidade de Código (1.5 min)

### O que falar:

"Adotamos uma abordagem dual de testes. Primeiro, testes de propriedade com Hypothesis — que geram dados aleatórios para verificar se propriedades universais sempre valem."

"Por exemplo: para qualquer DataFrame com NaN, a imputação deve resultar em zero NaN. Para qualquer lista de resultados, o relatório deve estar ordenado por F1-Score decrescente."

"Segundo, testes unitários clássicos para edge cases: arquivo não encontrado, CSV corrompido, modelo que falha durante treinamento."

"No total, 31 testes automatizados cobrem todas as etapas do pipeline."

### Pontos-chave:

- Property-based testing é diferencial (não é só pytest convencional)
- 12 propriedades formais de corretude
- Testes verificam comportamento universal, não exemplos isolados

### Possível pergunta do avaliador:

- "O que é property-based testing?" → "Ao invés de testar com dados fixos, geramos milhares de entradas aleatórias e verificamos que uma propriedade sempre vale. Por exemplo: após imputação, nunca deve sobrar NaN."

---

## Slide 11 — Estrutura do Repositório (1 min)

### O que falar:

"O repositório segue boas práticas de engenharia de software. Código-fonte separado em src/, testes em tests/, configurações centralizadas, e artefatos em diretórios dedicados."

"Cada módulo tem docstrings em português no formato Google, type hints, e logging estruturado. O código é limpo e seguiu princípios de responsabilidade única."

"O .gitignore exclui artefatos gerados — modelos e figuras são recriados a cada execução."

### Pontos-chave:

- Organização profissional
- Separação clara de responsabilidades
- Documentação integrada no código

---

## Slide 12 — Demonstração (1 min)

### O que falar:

"Para reproduzir o projeto: clone o repositório, instale as dependências com pip, e execute python main.py. O dataset já está incluído."

"O pipeline completo roda em aproximadamente 5 minutos. A etapa mais longa é o GridSearchCV do Random Forest."

"Ao final, todos os artefatos são gerados automaticamente: visualizações, modelos salvos, e métricas no MLflow."

### Pontos-chave:

- Se possível, mostrar a execução ao vivo ou um vídeo
- Destacar que é um único comando
- Mencionar que o dataset já está no repositório

---

## Slide 13 — Conclusões (1.5 min)

### O que falar:

"Em resumo, entregamos um pipeline completo, modular e reprodutível para classificação de potabilidade da água."

"As contribuições principais são: primeiro, a comparação sistemática de 3 algoritmos com busca otimizada de hiperparâmetros. Segundo, o tratamento rigoroso de desbalanceamento e valores ausentes prevenindo data leakage. Terceiro, o rastreamento de experimentos com MLflow. E quarto, uma suite de testes com propriedades formais."

"Como limitação, o F1-Score moderado reflete a natureza do dataset, não falha metodológica."

"Para trabalhos futuros: feature engineering, ensemble stacking, e possivelmente deploy como API."

### Pontos-chave:

- Resumir as 4 contribuições
- Reconhecer limitações (mostra maturidade)
- Sugerir trabalhos futuros (mostra visão)

---

## Slide 14 — Referências (30 seg)

### O que falar:

"As referências incluem o dataset original do Kaggle, as bibliotecas scikit-learn, XGBoost, imbalanced-learn para SMOTE, MLflow para rastreamento, e Hypothesis para testes de propriedade."

(Não precisa ler todas — apenas mencionar que estão disponíveis)

---

## Slide Final — Obrigado (30 seg)

### O que falar:

"O repositório está disponível no GitHub para consulta. Fico à disposição para perguntas."

### Dicas para perguntas:

Se perguntarem algo que não sabe:

- "Essa é uma boa pergunta. No escopo deste trabalho não investigamos isso, mas seria um ponto interessante para trabalhos futuros."

Se perguntarem sobre métricas baixas:

- Redirecionar para a metodologia: "O valor deste trabalho está na execução rigorosa do pipeline, prevenção de data leakage, e reprodutibilidade."

Se perguntarem sobre outro algoritmo (SVM, redes neurais):

- "SVM é uma alternativa válida. Optamos por Random Forest e XGBoost pela eficácia comprovada em datasets tabulares de pequeno porte. Redes neurais geralmente requerem mais dados."

---

## Resumo de Tempos

| Slide     | Tema             | Tempo       |
| --------- | ---------------- | ----------- |
| 1         | Introdução       | 1 min       |
| 2         | Dataset          | 1.5 min     |
| 3         | Arquitetura      | 1.5 min     |
| 4         | EDA              | 1.5 min     |
| 5         | Preprocessamento | 2 min       |
| 6         | Modelos          | 1.5 min     |
| 7         | Resultados       | 2 min       |
| 8         | Análise          | 1.5 min     |
| 9         | MLflow           | 1 min       |
| 10        | Testes           | 1.5 min     |
| 11        | Repositório      | 1 min       |
| 12        | Demonstração     | 1 min       |
| 13        | Conclusões       | 1.5 min     |
| 14        | Referências      | 0.5 min     |
| Final     | Perguntas        | 0.5 min     |
| **Total** |                  | **~18 min** |
