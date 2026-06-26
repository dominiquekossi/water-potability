"""Fixtures compartilhadas para os testes do pipeline de classificação de potabilidade da água.

Este módulo contém:
- Configuração do profile Hypothesis para o projeto
- Fixture de DataFrame válido com dados sintéticos
- Fixture de DataFrame com valores ausentes
- Fixture de configuração padrão do pipeline

**Validates: Requirements 7.5**
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "water-potability",
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
settings.load_profile("water-potability")

FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

ALL_COLUMNS = FEATURE_COLUMNS + ["Potability"]


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Retorna um DataFrame válido com dados sintéticos representativos.

    O DataFrame possui 100 registros, 9 colunas numéricas de features
    e a coluna alvo Potability com valores binários (0 ou 1).
    Utiliza seed fixa para garantir reprodutibilidade.
    """
    rng = np.random.default_rng(seed=42)

    data = {
        "ph": rng.uniform(0.0, 14.0, size=100),
        "Hardness": rng.uniform(47.0, 323.0, size=100),
        "Solids": rng.uniform(320.0, 61228.0, size=100),
        "Chloramines": rng.uniform(0.35, 13.13, size=100),
        "Sulfate": rng.uniform(129.0, 481.0, size=100),
        "Conductivity": rng.uniform(181.0, 753.0, size=100),
        "Organic_carbon": rng.uniform(2.2, 28.3, size=100),
        "Trihalomethanes": rng.uniform(0.74, 124.0, size=100),
        "Turbidity": rng.uniform(1.45, 6.74, size=100),
        "Potability": rng.choice([0, 1], size=100, p=[0.61, 0.39]),
    }

    df = pd.DataFrame(data)
    # Garantir tipo int64 para Potability
    df["Potability"] = df["Potability"].astype(np.int64)
    return df


@pytest.fixture
def dataframe_with_missing_values() -> pd.DataFrame:
    """Retorna um DataFrame com valores ausentes em colunas específicas.

    Simula o padrão real do dataset Water Potability onde as colunas
    ph, Sulfate e Trihalomethanes possuem valores ausentes.
    Utiliza seed fixa para garantir reprodutibilidade.
    """
    rng = np.random.default_rng(seed=42)
    n_rows = 100

    data = {
        "ph": rng.uniform(0.0, 14.0, size=n_rows),
        "Hardness": rng.uniform(47.0, 323.0, size=n_rows),
        "Solids": rng.uniform(320.0, 61228.0, size=n_rows),
        "Chloramines": rng.uniform(0.35, 13.13, size=n_rows),
        "Sulfate": rng.uniform(129.0, 481.0, size=n_rows),
        "Conductivity": rng.uniform(181.0, 753.0, size=n_rows),
        "Organic_carbon": rng.uniform(2.2, 28.3, size=n_rows),
        "Trihalomethanes": rng.uniform(0.74, 124.0, size=n_rows),
        "Turbidity": rng.uniform(1.45, 6.74, size=n_rows),
        "Potability": rng.choice([0, 1], size=n_rows, p=[0.61, 0.39]),
    }

    df = pd.DataFrame(data)
    df["Potability"] = df["Potability"].astype(np.int64)

    # Introduzir valores ausentes nas colunas que tipicamente possuem NaN
    # ph: ~15% ausentes
    ph_missing_idx = rng.choice(n_rows, size=15, replace=False)
    df.loc[ph_missing_idx, "ph"] = np.nan

    # Sulfate: ~24% ausentes
    sulfate_missing_idx = rng.choice(n_rows, size=24, replace=False)
    df.loc[sulfate_missing_idx, "Sulfate"] = np.nan

    # Trihalomethanes: ~5% ausentes
    thm_missing_idx = rng.choice(n_rows, size=5, replace=False)
    df.loc[thm_missing_idx, "Trihalomethanes"] = np.nan

    return df


@pytest.fixture
def default_pipeline_config() -> dict:
    """Retorna a configuração padrão do pipeline com seed fixa.

    Reflete a estrutura do arquivo configs/config.yaml, garantindo
    que todas as operações estocásticas utilizem random_state=42
    para reprodutibilidade (Requisito 7.5).
    """
    return {
        "random_state": 42,
        "dataset_path": "water_potability.csv",
        "preprocessing": {
            "imputation_strategy": "median",
            "knn_neighbors": 5,
            "scaling_method": "standard",
            "balancing_technique": "smote",
            "test_size": 0.2,
        },
        "training": {
            "cv_folds": 5,
            "scoring_metric": "f1",
            "models": [
                "logistic_regression",
                "random_forest",
                "xgboost",
            ],
        },
        "mlflow": {
            "experiment_name": "water-potability-classification",
            "tracking_uri": "mlruns",
            "port": 5000,
        },
        "output": {
            "figures_dir": "reports/figures/",
            "models_dir": "models/",
        },
    }
