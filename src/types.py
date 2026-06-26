"""Tipos de dados compartilhados do pipeline de classificação de potabilidade da água.

Este módulo define os TypedDicts utilizados como contratos de dados entre os
módulos do pipeline, garantindo consistência e documentação dos formatos
esperados em cada interface.
"""

from typing import TypedDict


class TrainedModelResult(TypedDict):
    """Resultado do treinamento de um modelo de classificação.

    Attributes:
        model_name: Nome identificador do modelo (ex: "random_forest").
        model: Instância do modelo sklearn/xgboost treinado.
        best_params: Dicionário com os melhores hiperparâmetros encontrados.
        training_time_seconds: Tempo de treinamento em segundos.
        status: Status do treinamento ("success" ou "failed").
        error_message: Mensagem de erro caso o treinamento falhe, None caso contrário.
    """

    model_name: str
    model: object
    best_params: dict
    training_time_seconds: float
    status: str
    error_message: str | None


class EvaluationResult(TypedDict):
    """Resultado da avaliação de um modelo no conjunto de teste.

    Attributes:
        model_name: Nome identificador do modelo avaliado.
        accuracy: Acurácia do modelo (0.0 a 1.0).
        precision: Precisão do modelo (0.0 a 1.0).
        recall: Recall/Sensibilidade do modelo (0.0 a 1.0).
        f1_score: F1-Score do modelo (0.0 a 1.0).
        auc_roc: Área sob a curva ROC (0.0 a 1.0).
        confusion_matrix_path: Caminho do arquivo PNG da matriz de confusão.
        roc_curve_path: Caminho do arquivo PNG da curva ROC.
    """

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix_path: str
    roc_curve_path: str


class PreprocessingParams(TypedDict):
    """Parâmetros de preprocessamento registrados no MLflow para reprodutibilidade.

    Attributes:
        imputation_strategy: Estratégia de imputação utilizada ("median" ou "knn").
        knn_neighbors: Número de vizinhos para KNN (None se estratégia for mediana).
        scaling_method: Método de escalonamento ("standard" ou "minmax").
        balancing_technique: Técnica de balanceamento ("smote", "oversampling" ou "undersampling").
        test_size: Proporção do conjunto de teste (ex: 0.2).
        random_state: Seed utilizada para reprodutibilidade.
        train_size_before_balance: Tamanho do conjunto de treino antes do balanceamento.
        train_size_after_balance: Tamanho do conjunto de treino após o balanceamento.
        class_distribution_before: Distribuição de classes antes do balanceamento.
        class_distribution_after: Distribuição de classes após o balanceamento.
        total_imputed_values: Contagem de valores imputados por atributo.
    """

    imputation_strategy: str
    knn_neighbors: int | None
    scaling_method: str
    balancing_technique: str
    test_size: float
    random_state: int
    train_size_before_balance: int
    train_size_after_balance: int
    class_distribution_before: dict
    class_distribution_after: dict
    total_imputed_values: dict
