# Feature: water-potability-classification, Property 2: Dataset Summary Accuracy
"""Testes de propriedade para o módulo de coleta de dados.

Utiliza Hypothesis para verificar que a função log_dataset_summary retorna
contagens e percentuais matematicamente corretos para qualquer DataFrame
válido com coluna Potability binária.

**Validates: Requirements 1.6**
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data_collection import log_dataset_summary


# Estratégia para gerar DataFrames válidos com as 10 colunas esperadas
@st.composite
def valid_potability_dataframes(draw):
    """Gera DataFrames válidos com 10 colunas: 9 numéricas (com NaN opcional) e Potability binária."""
    n_rows = draw(st.integers(min_value=1, max_value=200))

    # Gerar coluna Potability com valores em {0, 1}
    potability = draw(
        st.lists(st.integers(min_value=0, max_value=1), min_size=n_rows, max_size=n_rows)
    )

    # Nomes das 9 colunas numéricas
    numeric_columns = [
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

    data = {}
    for col in numeric_columns:
        values = draw(
            st.lists(
                st.one_of(
                    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                    st.none(),  # Representa NaN
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        data[col] = [np.nan if v is None else v for v in values]

    data["Potability"] = potability

    df = pd.DataFrame(data)
    return df


@settings(max_examples=20, deadline=None)
@given(df=valid_potability_dataframes())
def test_summary_total_registros_equals_len_df(df: pd.DataFrame):
    """Verifica que total_registros é igual ao número de linhas do DataFrame."""
    summary = log_dataset_summary(df)
    assert summary["total_registros"] == len(df)


@settings(max_examples=20, deadline=None)
@given(df=valid_potability_dataframes())
def test_summary_potavel_count_equals_sum_potability_1(df: pd.DataFrame):
    """Verifica que potavel_count é igual à soma de registros com Potability == 1."""
    summary = log_dataset_summary(df)
    expected = int((df["Potability"] == 1).sum())
    assert summary["potavel_count"] == expected


@settings(max_examples=20, deadline=None)
@given(df=valid_potability_dataframes())
def test_summary_nao_potavel_count_equals_sum_potability_0(df: pd.DataFrame):
    """Verifica que nao_potavel_count é igual à soma de registros com Potability == 0."""
    summary = log_dataset_summary(df)
    expected = int((df["Potability"] == 0).sum())
    assert summary["nao_potavel_count"] == expected


@settings(max_examples=20, deadline=None)
@given(df=valid_potability_dataframes())
def test_summary_percentages_sum_to_100(df: pd.DataFrame):
    """Verifica que potavel_pct + nao_potavel_pct é aproximadamente 100.0."""
    summary = log_dataset_summary(df)
    total_pct = summary["potavel_pct"] + summary["nao_potavel_pct"]
    assert abs(total_pct - 100.0) < 1e-9, (
        f"Soma dos percentuais deveria ser 100.0, obteve {total_pct}"
    )
