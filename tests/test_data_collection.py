"""Testes de propriedade e unitários para o módulo de coleta e validação de dados.

Este módulo contém property-based tests utilizando Hypothesis para verificar
que a função validate_dataset aceita apenas DataFrames com a estrutura
correta do dataset Water Potability, além de testes unitários para cenários
específicos de erro e sucesso.

Requirements: 1.3, 1.4, 1.5
"""

import math
import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
import hypothesis.extra.pandas as pdst

from src.data_collection import (
    load_dataset,
    log_dataset_summary,
    validate_dataset,
    EXPECTED_COLUMNS,
)


# Caminho do dataset real na raiz do workspace
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "water_potability.csv",
)


# ============================================================================
# Feature: water-potability-classification, Property 1: Dataset Validation Correctness
# Validates: Requirements 1.1, 1.2, 1.3
# ============================================================================


NUMERIC_COLUMNS = EXPECTED_COLUMNS[:-1]


@st.composite
def valid_dataframes(draw):
    """Gera DataFrames válidos com a estrutura correta do dataset Water Potability.

    Gera DataFrames com exatamente 10 colunas esperadas, onde as 9 primeiras
    são numéricas (podendo conter NaN) e Potability contém apenas 0 ou 1
    sem valores ausentes.
    """
    n_rows = draw(st.integers(min_value=1, max_value=50))

    # Gerar dados numéricos para as 9 colunas (permitindo NaN)
    data = {}
    for col in NUMERIC_COLUMNS:
        values = draw(
            st.lists(
                st.one_of(
                    st.floats(
                        min_value=-1e6,
                        max_value=1e6,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    st.none(),
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        # Substituir None por NaN para simular dados ausentes
        data[col] = [np.nan if v is None else v for v in values]

    # Gerar Potability com apenas 0 e 1 (sem NaN)
    potability_values = draw(
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    data["Potability"] = potability_values

    df = pd.DataFrame(data)
    return df


@st.composite
def dataframes_with_wrong_columns(draw):
    """Gera DataFrames com nomes de colunas incorretos.

    Pode ter número diferente de colunas ou nomes diferentes dos esperados.
    """
    strategy_choice = draw(st.integers(min_value=0, max_value=2))
    n_rows = draw(st.integers(min_value=1, max_value=20))

    if strategy_choice == 0:
        # Menos colunas que o esperado
        n_cols = draw(st.integers(min_value=1, max_value=9))
        col_names = [f"col_{i}" for i in range(n_cols)]
    elif strategy_choice == 1:
        # Mais colunas que o esperado
        n_cols = draw(st.integers(min_value=11, max_value=15))
        col_names = [f"col_{i}" for i in range(n_cols)]
    else:
        # 10 colunas mas com nomes errados
        col_names = [
            draw(st.text(min_size=1, max_size=10, alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
            )))
            for _ in range(10)
        ]
        # Garantir que não coincide com as colunas esperadas
        if col_names == EXPECTED_COLUMNS:
            col_names[0] = "wrong_name"

    data = {}
    for col in col_names:
        data[col] = draw(
            st.lists(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            )
        )

    df = pd.DataFrame(data)
    return df


@st.composite
def dataframes_with_non_numeric_columns(draw):
    """Gera DataFrames com colunas corretas mas pelo menos uma coluna não-numérica.

    Mantém a estrutura de 10 colunas com nomes esperados, mas substitui
    pelo menos uma das 9 colunas numéricas por dados não-numéricos (strings).
    """
    n_rows = draw(st.integers(min_value=1, max_value=20))

    # Escolher qual(is) coluna(s) terá tipo não-numérico
    non_numeric_col = draw(st.sampled_from(NUMERIC_COLUMNS))

    data = {}
    for col in NUMERIC_COLUMNS:
        if col == non_numeric_col:
            # Gerar strings para esta coluna
            values = draw(
                st.lists(
                    st.text(min_size=1, max_size=5, alphabet="abcdefghij"),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
        else:
            values = draw(
                st.lists(
                    st.floats(
                        min_value=-1000, max_value=1000,
                        allow_nan=False, allow_infinity=False,
                    ),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
        data[col] = values

    # Potability válida
    data["Potability"] = draw(
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    df = pd.DataFrame(data)
    return df


@st.composite
def dataframes_with_invalid_potability(draw):
    """Gera DataFrames com valores inválidos na coluna Potability.

    Mantém a estrutura correta de 10 colunas numéricas, mas a coluna
    Potability contém valores fora de {0, 1} ou valores NaN.
    """
    n_rows = draw(st.integers(min_value=1, max_value=20))
    invalid_type = draw(st.integers(min_value=0, max_value=1))

    data = {}
    for col in NUMERIC_COLUMNS:
        data[col] = draw(
            st.lists(
                st.floats(
                    min_value=-1000, max_value=1000,
                    allow_nan=False, allow_infinity=False,
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )

    if invalid_type == 0:
        # Potability com valores fora de {0, 1}
        invalid_values = draw(
            st.lists(
                st.one_of(
                    st.integers(min_value=2, max_value=100),
                    st.integers(min_value=-100, max_value=-1),
                ),
                min_size=1,
                max_size=n_rows,
            )
        )
        # Preencher o restante com valores válidos se necessário
        valid_fill = draw(
            st.lists(
                st.integers(min_value=0, max_value=1),
                min_size=max(0, n_rows - len(invalid_values)),
                max_size=max(0, n_rows - len(invalid_values)),
            )
        )
        potability = (invalid_values + valid_fill)[:n_rows]
        data["Potability"] = potability
    else:
        # Potability com NaN
        values = draw(
            st.lists(
                st.integers(min_value=0, max_value=1),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        data["Potability"] = [float(v) for v in values]
        # Inserir pelo menos um NaN
        nan_indices = draw(
            st.lists(
                st.integers(min_value=0, max_value=n_rows - 1),
                min_size=1,
                max_size=max(1, n_rows // 2),
                unique=True,
            )
        )
        for idx in nan_indices:
            data["Potability"][idx] = float("nan")

    df = pd.DataFrame(data)
    return df


@given(df=valid_dataframes())
@settings(max_examples=20, deadline=None)
def test_validate_dataset_accepts_valid_dataframes(df):
    """Verifica que validate_dataset aceita DataFrames com estrutura correta.

    Para qualquer DataFrame com exatamente 10 colunas esperadas, tipos
    numéricos nas 9 colunas de atributos, e Potability contendo apenas
    valores 0 e 1 sem NaN, a função deve retornar o DataFrame sem erro.
    """
    result = validate_dataset(df)
    assert result is df, "validate_dataset deve retornar o mesmo DataFrame de entrada"


@given(df=dataframes_with_wrong_columns())
@settings(max_examples=20, deadline=None)
def test_validate_dataset_rejects_wrong_columns(df):
    """Verifica que validate_dataset rejeita DataFrames com colunas incorretas.

    Para qualquer DataFrame que não possua exatamente as 10 colunas
    esperadas (nomes diferentes ou quantidade diferente), a função deve
    lançar ValueError.
    """
    with pytest.raises(ValueError):
        validate_dataset(df)


@given(df=dataframes_with_non_numeric_columns())
@settings(max_examples=20, deadline=None)
def test_validate_dataset_rejects_non_numeric_columns(df):
    """Verifica que validate_dataset rejeita DataFrames com colunas não-numéricas.

    Para qualquer DataFrame com os nomes corretos mas pelo menos uma das
    9 colunas de atributos com tipo não-numérico, a função deve lançar
    ValueError.
    """
    with pytest.raises(ValueError):
        validate_dataset(df)


@given(df=dataframes_with_invalid_potability())
@settings(max_examples=20, deadline=None)
def test_validate_dataset_rejects_invalid_potability(df):
    """Verifica que validate_dataset rejeita DataFrames com Potability inválida.

    Para qualquer DataFrame com estrutura correta mas coluna Potability
    contendo valores fora de {0, 1} ou valores NaN, a função deve lançar
    ValueError.
    """
    with pytest.raises(ValueError):
        validate_dataset(df)


# ============================================================================
# Testes Unitários (Example-Based) para o módulo de coleta de dados
# Validates: Requirements 1.3, 1.4, 1.5
# ============================================================================


class TestLoadDatasetErrors:
    """Testes de erro para a função load_dataset."""

    def test_load_dataset_file_not_found(self):
        """Verifica que FileNotFoundError é levantado com caminho inválido.

        A mensagem de erro deve conter o caminho informado e o link do Kaggle
        para download do dataset.
        """
        caminho_invalido = "/caminho/inexistente/dados.csv"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_dataset(caminho_invalido)

        mensagem = str(exc_info.value)
        assert caminho_invalido in mensagem
        assert "kaggle" in mensagem.lower()

    def test_load_dataset_corrupt_csv(self, tmp_path):
        """Verifica que ValueError é levantado para CSV com conteúdo corrompido.

        Cria um arquivo temporário com conteúdo binário inválido e verifica que
        a exceção contém informação sobre a causa do erro.
        """
        arquivo_corrompido = tmp_path / "corrompido.csv"
        # Escreve conteúdo binário que não pode ser parseado como CSV válido
        arquivo_corrompido.write_bytes(b"\x80\x81\x82\x83\xff\xfe\x00\x01" * 100)

        with pytest.raises(ValueError) as exc_info:
            load_dataset(str(arquivo_corrompido))

        mensagem = str(exc_info.value)
        assert len(mensagem) > 0


class TestValidateDatasetErrors:
    """Testes de erro para a função validate_dataset."""

    def test_validate_dataset_invalid_potability(self):
        """Verifica que ValueError é levantado quando Potability tem valores fora de {0, 1}.

        A mensagem de erro deve conter os valores inválidos encontrados e a
        contagem de registros afetados.
        """
        dados = {
            "ph": [7.0, 6.5, 8.0],
            "Hardness": [200.0, 180.0, 220.0],
            "Solids": [10000.0, 12000.0, 11000.0],
            "Chloramines": [7.0, 6.0, 8.0],
            "Sulfate": [300.0, 320.0, 310.0],
            "Conductivity": [400.0, 420.0, 410.0],
            "Organic_carbon": [14.0, 15.0, 13.0],
            "Trihalomethanes": [60.0, 70.0, 65.0],
            "Turbidity": [4.0, 3.5, 4.5],
            "Potability": [0, 1, 2],  # Valor 2 é inválido
        }
        df = pd.DataFrame(dados)

        with pytest.raises(ValueError) as exc_info:
            validate_dataset(df)

        mensagem = str(exc_info.value)
        assert "2" in mensagem  # valor inválido presente na mensagem
        assert "1" in mensagem  # contagem de registros afetados

    def test_validate_dataset_potability_with_nan(self):
        """Verifica que ValueError é levantado quando Potability contém NaN.

        A coluna Potability não deve aceitar valores ausentes.
        """
        dados = {
            "ph": [7.0, 6.5, 8.0],
            "Hardness": [200.0, 180.0, 220.0],
            "Solids": [10000.0, 12000.0, 11000.0],
            "Chloramines": [7.0, 6.0, 8.0],
            "Sulfate": [300.0, 320.0, 310.0],
            "Conductivity": [400.0, 420.0, 410.0],
            "Organic_carbon": [14.0, 15.0, 13.0],
            "Trihalomethanes": [60.0, 70.0, 65.0],
            "Turbidity": [4.0, 3.5, 4.5],
            "Potability": [0, 1, np.nan],  # NaN é inválido
        }
        df = pd.DataFrame(dados)

        with pytest.raises(ValueError) as exc_info:
            validate_dataset(df)

        mensagem = str(exc_info.value)
        assert "ausente" in mensagem.lower() or "nan" in mensagem.lower()


class TestLoadDatasetSuccess:
    """Testes de sucesso para a função load_dataset."""

    def test_load_dataset_success(self):
        """Verifica carregamento bem-sucedido do dataset real.

        O DataFrame retornado deve ter 10 colunas e ao menos 1 registro.
        """
        df = load_dataset(DATASET_PATH)

        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 10
        assert len(df) > 0


class TestValidateDatasetSuccess:
    """Testes de sucesso para a função validate_dataset."""

    def test_validate_dataset_success(self):
        """Verifica que o dataset real passa na validação sem exceções."""
        df = load_dataset(DATASET_PATH)

        # Não deve levantar exceção
        resultado = validate_dataset(df)

        assert isinstance(resultado, pd.DataFrame)
        assert len(resultado) == len(df)


class TestLogDatasetSummary:
    """Testes para a função log_dataset_summary."""

    def test_log_dataset_summary(self):
        """Verifica que o resumo retornado contém as chaves corretas e percentuais somam ~100%.

        O dicionário retornado deve ter total_registros, potavel_count,
        potavel_pct, nao_potavel_count, nao_potavel_pct e valores_ausentes_total.
        Os percentuais de potável e não potável devem somar aproximadamente 100%.
        """
        df = load_dataset(DATASET_PATH)
        summary = log_dataset_summary(df)

        # Verificar chaves esperadas
        chaves_esperadas = {
            "total_registros",
            "potavel_count",
            "potavel_pct",
            "nao_potavel_count",
            "nao_potavel_pct",
            "valores_ausentes_total",
        }
        assert set(summary.keys()) == chaves_esperadas

        # Verificar que total é coerente
        assert summary["total_registros"] == len(df)
        assert summary["potavel_count"] + summary["nao_potavel_count"] == summary["total_registros"]

        # Verificar que percentuais somam ~100%
        soma_pct = summary["potavel_pct"] + summary["nao_potavel_pct"]
        assert math.isclose(soma_pct, 100.0, rel_tol=1e-9)

        # Verificar que valores são não-negativos
        assert summary["potavel_count"] >= 0
        assert summary["nao_potavel_count"] >= 0
        assert summary["valores_ausentes_total"] >= 0
