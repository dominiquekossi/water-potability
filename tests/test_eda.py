"""Testes de propriedade para o módulo de Análise Exploratória de Dados (EDA).

Este módulo contém testes baseados em propriedades utilizando Hypothesis
para verificar a corretude das funções de EDA do pipeline de classificação
de potabilidade da água.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.eda import EXPECTED_NUMERIC_COLUMNS, report_missing_values

# Feature: water-potability-classification, Property 3: Missing Values Report Accuracy


@st.composite
def dataframes_with_nan_patterns(draw):
    """Estratégia que gera DataFrames com padrões NaN aleatórios nas 9 colunas numéricas.

    Gera DataFrames contendo as 9 colunas numéricas esperadas mais a coluna Potability,
    com valores NaN inseridos aleatoriamente nas colunas numéricas.
    """
    n_rows = draw(st.integers(min_value=1, max_value=100))

    data = {}
    for col in EXPECTED_NUMERIC_COLUMNS:
        # Gerar valores numéricos base
        values = draw(
            st.lists(
                st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        values = list(values)

        # Decidir quais posições serão NaN
        nan_mask = draw(
            st.lists(st.booleans(), min_size=n_rows, max_size=n_rows)
        )
        for i in range(n_rows):
            if nan_mask[i]:
                values[i] = np.nan

        data[col] = values

    # Coluna Potability (sem NaN)
    potability = draw(
        st.lists(st.integers(min_value=0, max_value=1), min_size=n_rows, max_size=n_rows)
    )
    data["Potability"] = potability

    return pd.DataFrame(data)


@given(df=dataframes_with_nan_patterns())
@settings(max_examples=20, deadline=None)
def test_missing_values_report_accuracy(df: pd.DataFrame):
    """Verifica que o relatório de valores ausentes reporta contagens e percentuais corretos.

    Para qualquer DataFrame com padrões NaN conhecidos, a função report_missing_values
    deve retornar contagens por coluna que correspondem exatamente ao número real de
    valores NaN, e percentuais iguais a (contagem / total_linhas) * 100.

    **Validates: Requirements 2.3**
    """
    report = report_missing_values(df)

    total_rows = len(df)

    for col in EXPECTED_NUMERIC_COLUMNS:
        expected_count = df[col].isna().sum()
        expected_pct = (expected_count / total_rows) * 100

        # Verificar que a coluna está presente no relatório
        assert col in report.index, (
            f"Coluna '{col}' não encontrada no índice do relatório"
        )

        # Verificar contagem exata
        assert report.loc[col, "contagem"] == expected_count, (
            f"Contagem incorreta para '{col}': "
            f"esperado {expected_count}, obtido {report.loc[col, 'contagem']}"
        )

        # Verificar percentual (arredondado para 2 casas decimais conforme implementação)
        assert report.loc[col, "percentual"] == round(expected_pct, 2), (
            f"Percentual incorreto para '{col}': "
            f"esperado {round(expected_pct, 2)}, obtido {report.loc[col, 'percentual']}"
        )
