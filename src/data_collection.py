"""Módulo de coleta e validação de dados do pipeline de classificação de potabilidade da água.

Este módulo é responsável por carregar o dataset Water Potability a partir de um
arquivo CSV, validar sua estrutura e integridade, e registrar um resumo
estatístico do dataset carregado. Implementa tratamento robusto de erros para
arquivos não encontrados, CSV corrompidos e dados inválidos.
"""

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
    "Potability",
]


def load_dataset(file_path: str) -> pd.DataFrame:
    """Carrega o dataset Water Potability de um arquivo CSV.

    Realiza a leitura do arquivo CSV especificado e retorna um DataFrame pandas.
    Aceita valores ausentes (NaN) nos atributos numéricos sem interromper o
    carregamento. Valida que o arquivo existe e pode ser lido corretamente.

    Args:
        file_path: Caminho para o arquivo CSV.

    Returns:
        DataFrame com os dados carregados.

    Raises:
        FileNotFoundError: Se o arquivo não for encontrado no caminho especificado.
        ValueError: Se a estrutura do CSV for inválida (corrompido, encoding
            inválido ou estrutura de colunas diferente da esperada).
    """
    logger.info(f"Iniciando carregamento do dataset: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        msg = (
            f"Arquivo não encontrado: '{file_path}'. "
            f"Verifique se o dataset Water Potability está no caminho correto. "
            f"O dataset pode ser obtido em: "
            f"https://www.kaggle.com/datasets/adityakadiwal/water-potability"
        )
        logger.critical(msg)
        raise FileNotFoundError(msg)
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        msg = (
            f"Erro ao ler o arquivo CSV '{file_path}': {e}. "
            f"O arquivo pode estar corrompido ou com encoding inválido."
        )
        logger.critical(msg)
        raise ValueError(msg)
    except Exception as e:
        msg = (
            f"Erro inesperado ao ler o arquivo CSV '{file_path}': {e}."
        )
        logger.critical(msg)
        raise ValueError(msg)

    # Validar estrutura de colunas
    if list(df.columns) != EXPECTED_COLUMNS:
        colunas_encontradas = list(df.columns)
        msg = (
            f"Estrutura de colunas inválida no arquivo '{file_path}'. "
            f"Esperado: {EXPECTED_COLUMNS}. "
            f"Encontrado: {colunas_encontradas}."
        )
        logger.critical(msg)
        raise ValueError(msg)

    # Validar que há pelo menos 1 registro
    if len(df) == 0:
        msg = f"O arquivo '{file_path}' não contém registros."
        logger.critical(msg)
        raise ValueError(msg)

    logger.info(
        f"Dataset carregado com sucesso: {len(df)} registros, "
        f"{len(df.columns)} colunas."
    )
    return df


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Valida a estrutura e integridade do DataFrame carregado.

    Verifica que o DataFrame possui exatamente 10 colunas com os nomes
    esperados, que a coluna Potability contém apenas valores inteiros 0 e 1
    (sem NaN), e que os demais 9 atributos possuem tipo numérico.

    Args:
        df: DataFrame a ser validado.

    Returns:
        DataFrame validado (mesmo objeto de entrada se a validação passar).

    Raises:
        ValueError: Se o DataFrame não possuir as 10 colunas esperadas, se a
            coluna Potability contiver valores inválidos ou ausentes, ou se os
            atributos numéricos não forem do tipo correto.
    """
    logger.info("Iniciando validação do dataset.")

    # Verificar número e nomes das colunas
    if list(df.columns) != EXPECTED_COLUMNS:
        msg = (
            f"DataFrame não possui as colunas esperadas. "
            f"Esperado: {EXPECTED_COLUMNS}. "
            f"Encontrado: {list(df.columns)}."
        )
        logger.critical(msg)
        raise ValueError(msg)

    # Verificar tipos numéricos dos 9 atributos (exceto Potability)
    numeric_columns = EXPECTED_COLUMNS[:-1]  # Todas exceto Potability
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            msg = (
                f"A coluna '{col}' não é do tipo numérico. "
                f"Tipo encontrado: {df[col].dtype}."
            )
            logger.critical(msg)
            raise ValueError(msg)

    # Verificar valores ausentes na coluna Potability
    potability_nans = df["Potability"].isna().sum()
    if potability_nans > 0:
        msg = (
            f"A coluna 'Potability' contém {potability_nans} valor(es) ausente(s). "
            f"Esta coluna não pode conter valores NaN."
        )
        logger.critical(msg)
        raise ValueError(msg)

    # Verificar que Potability contém apenas 0 e 1
    valores_unicos = df["Potability"].unique()
    valores_invalidos = [v for v in valores_unicos if v not in (0, 1)]

    if valores_invalidos:
        registros_afetados = df[~df["Potability"].isin([0, 1])].shape[0]
        msg = (
            f"A coluna 'Potability' contém valores inválidos: {valores_invalidos}. "
            f"Apenas os valores 0 e 1 são permitidos. "
            f"Quantidade de registros afetados: {registros_afetados}."
        )
        logger.critical(msg)
        raise ValueError(msg)

    logger.info("Validação do dataset concluída com sucesso.")
    return df


def log_dataset_summary(df: pd.DataFrame) -> dict:
    """Gera e registra um resumo estatístico do dataset.

    Calcula e registra no log a quantidade total de registros, a distribuição
    das classes (potável e não potável) com contagem e percentual, e a
    quantidade total de valores ausentes no dataset.

    Args:
        df: DataFrame validado.

    Returns:
        Dicionário com estatísticas resumidas contendo:
            - total_registros: Número total de registros no dataset.
            - potavel_count: Quantidade de amostras potáveis (Potability=1).
            - potavel_pct: Percentual de amostras potáveis.
            - nao_potavel_count: Quantidade de amostras não potáveis (Potability=0).
            - nao_potavel_pct: Percentual de amostras não potáveis.
            - valores_ausentes_total: Quantidade total de valores ausentes.
    """
    total_registros = len(df)

    potavel_count = int((df["Potability"] == 1).sum())
    nao_potavel_count = int((df["Potability"] == 0).sum())

    potavel_pct = (potavel_count / total_registros) * 100
    nao_potavel_pct = (nao_potavel_count / total_registros) * 100

    valores_ausentes_total = int(df.isna().sum().sum())

    summary = {
        "total_registros": total_registros,
        "potavel_count": potavel_count,
        "potavel_pct": potavel_pct,
        "nao_potavel_count": nao_potavel_count,
        "nao_potavel_pct": nao_potavel_pct,
        "valores_ausentes_total": valores_ausentes_total,
    }

    logger.info(
        f"Resumo do dataset: "
        f"{total_registros} registros totais | "
        f"Potável: {potavel_count} ({potavel_pct:.2f}%) | "
        f"Não potável: {nao_potavel_count} ({nao_potavel_pct:.2f}%) | "
        f"Valores ausentes: {valores_ausentes_total}"
    )

    return summary
