"""Módulo de preprocessamento de dados para classificação de potabilidade da água.

Este módulo implementa o pipeline de preprocessamento completo, incluindo
imputação de valores ausentes, divisão treino/teste estratificada,
escalonamento de features e balanceamento de classes. Todas as operações
são registradas no MLflow para garantir reprodutibilidade.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import mlflow

from src.logger import get_logger
from src.mlflow_utils import safe_mlflow_operation

logger = get_logger(__name__)


def impute_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    knn_neighbors: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Imputa valores ausentes usando a estratégia especificada.

    Realiza a imputação de valores NaN no DataFrame utilizando a estratégia
    escolhida (mediana ou KNN). Emite alerta para atributos com mais de 80%
    de valores ausentes antes da imputação.

    Args:
        df: DataFrame com valores ausentes.
        strategy: Estratégia de imputação. Aceita "median" ou "knn".
        knn_neighbors: Número de vizinhos para KNN (deve estar entre 3 e 10).

    Returns:
        Tupla contendo:
            - DataFrame com valores imputados (sem NaN).
            - Dicionário com contagem de valores imputados por atributo.

    Raises:
        ValueError: Se a estratégia não for "median" ou "knn".
        ValueError: Se knn_neighbors estiver fora do intervalo [3, 10].
    """
    if strategy not in ("median", "knn"):
        raise ValueError(
            f"Estratégia de imputação inválida: '{strategy}'. "
            "Use 'median' ou 'knn'."
        )

    if strategy == "knn" and not (3 <= knn_neighbors <= 10):
        raise ValueError(
            f"knn_neighbors deve estar entre 3 e 10. Recebido: {knn_neighbors}"
        )

    # Verificar atributos com >80% de valores ausentes e emitir alerta
    total_rows = len(df)
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        if missing_pct > 80:
            logger.warning(
                f"Atributo '{col}' possui {missing_pct:.1f}% de valores ausentes "
                f"({missing_count}/{total_rows} registros). "
                "Prosseguindo com imputação normalmente."
            )

    # Registrar contagem de valores ausentes antes da imputação
    imputed_counts = {}
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            imputed_counts[col] = int(missing)

    # Aplicar imputação
    df_imputed = df.copy()

    if strategy == "median":
        imputer = SimpleImputer(strategy="median")
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_imputed),
            columns=df.columns,
            index=df.index,
        )
    elif strategy == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_imputed),
            columns=df.columns,
            index=df.index,
        )

    # Validar que nenhum NaN permanece após imputação
    remaining_nans = df_imputed.isna().sum().sum()
    if remaining_nans > 0:
        logger.error(
            f"Imputação incompleta: {remaining_nans} valores NaN permanecem."
        )
    else:
        logger.info(
            "Imputação concluída com sucesso. Zero valores NaN no dataset."
        )

    # Log de valores imputados por atributo
    for col, count in imputed_counts.items():
        logger.info(f"Atributo '{col}': {count} valores imputados.")

    logger.info(
        f"Estratégia de imputação: {strategy}"
        + (f" (k={knn_neighbors})" if strategy == "knn" else "")
    )

    return df_imputed, imputed_counts


def split_data(
    df: pd.DataFrame,
    target_col: str = "Potability",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide dados em treino/teste com estratificação pela variável alvo.

    Realiza a divisão do dataset em conjuntos de treinamento (80%) e teste
    (20%), garantindo que a proporção de classes seja preservada em ambos
    os conjuntos através de estratificação.

    Args:
        df: DataFrame completo com features e variável alvo.
        target_col: Nome da coluna alvo para estratificação.
        test_size: Proporção do conjunto de teste (padrão 0.2 = 20%).
        random_state: Seed para reprodutibilidade.

    Returns:
        Tupla contendo:
            - X_train: Features de treinamento (DataFrame).
            - X_test: Features de teste (DataFrame).
            - y_train: Labels de treinamento (Series).
            - y_test: Labels de teste (Series).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        f"Divisão treino/teste concluída: "
        f"treino={len(X_train)} amostras ({(1 - test_size) * 100:.0f}%), "
        f"teste={len(X_test)} amostras ({test_size * 100:.0f}%)"
    )
    logger.info(
        f"Distribuição no treino - Classe 0: {(y_train == 0).sum()}, "
        f"Classe 1: {(y_train == 1).sum()}"
    )
    logger.info(
        f"Distribuição no teste - Classe 0: {(y_test == 0).sum()}, "
        f"Classe 1: {(y_test == 1).sum()}"
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "standard",
) -> tuple[np.ndarray, np.ndarray, object]:
    """Aplica escalonamento ajustado no treino e transformado no teste.

    O scaler é ajustado (fit) exclusivamente no conjunto de treinamento e
    a transformação (transform) é aplicada em ambos os conjuntos, evitando
    data leakage do conjunto de teste.

    Args:
        X_train: Features de treinamento.
        X_test: Features de teste.
        method: Método de escalonamento. Aceita "standard" (StandardScaler)
            ou "minmax" (MinMaxScaler).

    Returns:
        Tupla contendo:
            - X_train_scaled: Array com features de treino escalonadas.
            - X_test_scaled: Array com features de teste escalonadas.
            - scaler: Objeto scaler ajustado (para possível reutilização).

    Raises:
        ValueError: Se o método não for "standard" ou "minmax".
    """
    if method not in ("standard", "minmax"):
        raise ValueError(
            f"Método de escalonamento inválido: '{method}'. "
            "Use 'standard' ou 'minmax'."
        )

    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    # Fit APENAS no treino, transform em ambos
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Escalonamento aplicado: método '{method}'.")
    logger.info(
        f"Scaler ajustado no treino ({X_train.shape[0]} amostras) "
        f"e transformação aplicada no teste ({X_test.shape[0]} amostras)."
    )

    return X_train_scaled, X_test_scaled, scaler


def balance_classes(
    X_train: np.ndarray,
    y_train: pd.Series,
    technique: str = "smote",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica técnica de reamostragem no conjunto de treinamento.

    Realiza balanceamento de classes para atingir razão 1:1 entre classes,
    aplicando a técnica exclusivamente no conjunto de treinamento.

    Args:
        X_train: Array de features de treinamento.
        y_train: Series ou array com labels de treinamento.
        technique: Técnica de reamostragem. Aceita "smote", "oversampling"
            ou "undersampling".
        random_state: Seed para reprodutibilidade.

    Returns:
        Tupla contendo:
            - X_resampled: Array com features de treino reamostradas.
            - y_resampled: Array com labels de treino reamostradas.

    Raises:
        ValueError: Se a técnica não for "smote", "oversampling" ou
            "undersampling".
    """
    if technique not in ("smote", "oversampling", "undersampling"):
        raise ValueError(
            f"Técnica de balanceamento inválida: '{technique}'. "
            "Use 'smote', 'oversampling' ou 'undersampling'."
        )

    # Converter y_train para array numpy se necessário
    y_array = np.array(y_train)

    # Distribuição antes do balanceamento
    unique, counts = np.unique(y_array, return_counts=True)
    dist_before = dict(zip(unique.astype(int), counts.astype(int)))
    logger.info(f"Distribuição antes do balanceamento: {dist_before}")

    # Aplicar técnica de reamostragem com razão 1:1
    sampling_strategy = 1.0  # Razão 1:1

    if technique == "smote":
        resampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    elif technique == "oversampling":
        resampler = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    else:  # undersampling
        resampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

    X_resampled, y_resampled = resampler.fit_resample(X_train, y_array)

    # Distribuição após balanceamento
    unique_after, counts_after = np.unique(y_resampled, return_counts=True)
    dist_after = dict(
        zip(unique_after.astype(int), counts_after.astype(int))
    )
    logger.info(f"Distribuição após balanceamento: {dist_after}")
    logger.info(
        f"Técnica utilizada: '{technique}'. "
        f"Amostras: {len(y_array)} -> {len(y_resampled)}"
    )

    return X_resampled, y_resampled


def run_preprocessing(
    df: pd.DataFrame,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Executa pipeline de preprocessamento completo.

    Orquestra todas as etapas de preprocessamento na ordem correta:
    1. Imputação de valores ausentes
    2. Divisão treino/teste com estratificação
    3. Escalonamento de features
    4. Balanceamento de classes no treino

    Registra todos os parâmetros no MLflow para reprodutibilidade.

    Args:
        df: DataFrame bruto com features e variável alvo.
        config: Dicionário de configuração com chaves:
            - imputation_strategy: "median" ou "knn"
            - knn_neighbors: int (3-10)
            - scaling_method: "standard" ou "minmax"
            - balancing_technique: "smote", "oversampling" ou "undersampling"
            - test_size: float (padrão 0.2)
            - random_state: int (padrão 42)

    Returns:
        Tupla contendo:
            - X_train: Array de features de treino (balanceado e escalonado).
            - X_test: Array de features de teste (escalonado).
            - y_train: Array de labels de treino (balanceado).
            - y_test: Array de labels de teste.
            - params: Dicionário com todos os parâmetros de preprocessamento.
    """
    # Extrair parâmetros da configuração
    preprocessing_config = config.get("preprocessing", {})
    random_state = config.get("random_state", 42)

    imputation_strategy = preprocessing_config.get(
        "imputation_strategy", "median"
    )
    knn_neighbors = preprocessing_config.get("knn_neighbors", 5)
    scaling_method = preprocessing_config.get("scaling_method", "standard")
    balancing_technique = preprocessing_config.get(
        "balancing_technique", "smote"
    )
    test_size = preprocessing_config.get("test_size", 0.2)

    logger.info("=" * 60)
    logger.info("INÍCIO DO PIPELINE DE PREPROCESSAMENTO")
    logger.info("=" * 60)

    # 1. Imputação de valores ausentes
    logger.info("Etapa 1/4: Imputação de valores ausentes")
    df_imputed, imputed_counts = impute_missing_values(
        df=df,
        strategy=imputation_strategy,
        knn_neighbors=knn_neighbors,
    )

    # 2. Divisão treino/teste com estratificação (ANTES do balanceamento)
    logger.info("Etapa 2/4: Divisão treino/teste estratificada")
    X_train, X_test, y_train, y_test = split_data(
        df=df_imputed,
        target_col="Potability",
        test_size=test_size,
        random_state=random_state,
    )

    # Registrar distribuição de classes antes do balanceamento
    train_class_dist_before = {
        "class_0": int((y_train == 0).sum()),
        "class_1": int((y_train == 1).sum()),
    }
    train_size_before = len(y_train)

    # 3. Escalonamento de features
    logger.info("Etapa 3/4: Escalonamento de features")
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train=X_train,
        X_test=X_test,
        method=scaling_method,
    )

    # 4. Balanceamento de classes (APENAS no treino)
    logger.info("Etapa 4/4: Balanceamento de classes")
    X_train_balanced, y_train_balanced = balance_classes(
        X_train=X_train_scaled,
        y_train=y_train,
        technique=balancing_technique,
        random_state=random_state,
    )

    # Distribuição de classes após balanceamento
    unique_after, counts_after = np.unique(
        y_train_balanced, return_counts=True
    )
    train_class_dist_after = {
        f"class_{int(c)}": int(n)
        for c, n in zip(unique_after, counts_after)
    }
    train_size_after = len(y_train_balanced)

    # Montar dicionário de parâmetros
    params = {
        "imputation_strategy": imputation_strategy,
        "knn_neighbors": knn_neighbors if imputation_strategy == "knn" else None,
        "scaling_method": scaling_method,
        "balancing_technique": balancing_technique,
        "test_size": test_size,
        "random_state": random_state,
        "train_size_before_balance": train_size_before,
        "train_size_after_balance": train_size_after,
        "class_distribution_before": train_class_dist_before,
        "class_distribution_after": train_class_dist_after,
        "total_imputed_values": imputed_counts,
    }

    # Registrar parâmetros no MLflow
    _log_preprocessing_params(params)

    logger.info("=" * 60)
    logger.info("PIPELINE DE PREPROCESSAMENTO CONCLUÍDO")
    logger.info(f"  Treino: {X_train_balanced.shape[0]} amostras")
    logger.info(f"  Teste: {X_test_scaled.shape[0]} amostras")
    logger.info(f"  Features: {X_train_balanced.shape[1]}")
    logger.info("=" * 60)

    return (
        X_train_balanced,
        X_test_scaled,
        y_train_balanced,
        np.array(y_test),
        params,
    )


def _log_preprocessing_params(params: dict) -> None:
    """Registra parâmetros de preprocessamento no MLflow.

    Registra todos os parâmetros necessários para reconstruir o pipeline
    de preprocessamento com resultados idênticos.

    Args:
        params: Dicionário com parâmetros de preprocessamento.
    """
    try:
        mlflow.log_param("imputation_strategy", params["imputation_strategy"])
        if params["knn_neighbors"] is not None:
            mlflow.log_param("knn_neighbors", params["knn_neighbors"])
        mlflow.log_param("scaling_method", params["scaling_method"])
        mlflow.log_param("balancing_technique", params["balancing_technique"])
        mlflow.log_param("test_size", params["test_size"])
        mlflow.log_param("random_state", params["random_state"])
        mlflow.log_param(
            "train_size_before_balance", params["train_size_before_balance"]
        )
        mlflow.log_param(
            "train_size_after_balance", params["train_size_after_balance"]
        )
        mlflow.log_param(
            "class_distribution_before",
            str(params["class_distribution_before"]),
        )
        mlflow.log_param(
            "class_distribution_after",
            str(params["class_distribution_after"]),
        )
        mlflow.log_param(
            "total_imputed_values",
            str(params["total_imputed_values"]),
        )
        logger.info("Parâmetros de preprocessamento registrados no MLflow.")
    except Exception as e:
        logger.warning(
            f"Falha ao registrar parâmetros no MLflow: {e}. "
            "Parâmetros disponíveis no dicionário de retorno."
        )
