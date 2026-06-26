"""Módulo de Análise Exploratória de Dados (EDA) do pipeline de potabilidade da água.

Este módulo implementa a análise exploratória completa do dataset Water Potability,
incluindo estatísticas descritivas, visualizações de distribuição, matriz de correlação,
análise da variável alvo e relatório de valores ausentes. Todos os artefatos gerados
são registrados no MLflow para rastreabilidade.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.logger import get_logger
from src.mlflow_utils import create_run, safe_log_artifact, setup_experiment

logger = get_logger(__name__)

# Colunas numéricas esperadas no dataset (9 atributos)
EXPECTED_NUMERIC_COLUMNS = [
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

# Coluna alvo
TARGET_COLUMN = "Potability"

# DPI mínimo para visualizações
MIN_DPI = 150


def _validate_dataframe(df: pd.DataFrame) -> bool:
    """Valida se o DataFrame contém as colunas numéricas esperadas e não está vazio.

    Args:
        df: DataFrame a ser validado.

    Returns:
        True se o DataFrame é válido, False caso contrário.
    """
    if df is None or df.empty:
        logger.error("DataFrame de entrada está vazio. Interrompendo EDA.")
        return False

    missing_cols = set(EXPECTED_NUMERIC_COLUMNS) - set(df.columns)
    if missing_cols:
        logger.error(
            f"DataFrame não contém as colunas numéricas esperadas. "
            f"Colunas ausentes: {missing_cols}. Interrompendo EDA."
        )
        return False

    return True


def compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatísticas descritivas para os 9 atributos numéricos.

    Gera média, mediana, desvio padrão, mínimo, máximo e quartis (25%, 50%, 75%)
    para cada um dos 9 atributos numéricos do dataset.

    Args:
        df: DataFrame contendo os dados do dataset Water Potability.

    Returns:
        DataFrame com as estatísticas descritivas (colunas como atributos,
        linhas como métricas estatísticas).
    """
    numeric_df = df[EXPECTED_NUMERIC_COLUMNS]

    stats = numeric_df.describe().T
    stats["median"] = numeric_df.median()

    # Reorganizar colunas para clareza
    stats = stats[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]

    logger.info("Estatísticas descritivas calculadas para os 9 atributos numéricos:")
    logger.info(f"\n{stats.to_string()}")

    return stats


def generate_distribution_plots(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Gera histogramas de distribuição para cada atributo e salva como PNG.

    Cria um histograma com curva de densidade (KDE) para cada um dos 9 atributos
    numéricos do dataset, salvando cada figura como arquivo PNG com resolução
    mínima de 150 DPI.

    Args:
        df: DataFrame contendo os dados do dataset Water Potability.
        output_dir: Diretório onde os arquivos PNG serão salvos.

    Returns:
        Lista de caminhos dos arquivos PNG gerados.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for col in EXPECTED_NUMERIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribuição de {col}", fontsize=14)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Frequência", fontsize=12)
        plt.tight_layout()

        filepath = os.path.join(output_dir, f"distribuicao_{col.lower()}.png")
        fig.savefig(filepath, dpi=MIN_DPI, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(filepath)
        logger.info(f"Histograma de distribuição salvo: {filepath}")

    return saved_paths


def generate_correlation_matrix(df: pd.DataFrame, output_dir: str) -> str:
    """Gera e salva a matriz de correlação como heatmap PNG.

    Calcula a correlação de Pearson entre os 9 atributos numéricos e gera
    um heatmap anotado com os valores de correlação, salvando como arquivo
    PNG com resolução mínima de 150 DPI.

    Args:
        df: DataFrame contendo os dados do dataset Water Potability.
        output_dir: Diretório onde o arquivo PNG será salvo.

    Returns:
        Caminho do arquivo PNG gerado.
    """
    os.makedirs(output_dir, exist_ok=True)

    numeric_df = df[EXPECTED_NUMERIC_COLUMNS]
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Matriz de Correlação - Atributos Numéricos", fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "matriz_correlacao.png")
    fig.savefig(filepath, dpi=MIN_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Matriz de correlação salva: {filepath}")
    return filepath


def generate_target_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """Gera gráfico de distribuição da variável alvo.

    Cria um gráfico de barras mostrando a contagem absoluta e o percentual
    de cada classe da variável alvo (Potability: 0=não potável, 1=potável),
    salvando como arquivo PNG com resolução mínima de 150 DPI.

    Args:
        df: DataFrame contendo os dados do dataset Water Potability.
        output_dir: Diretório onde o arquivo PNG será salvo.

    Returns:
        Caminho do arquivo PNG gerado.
    """
    os.makedirs(output_dir, exist_ok=True)

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = len(df)
    percentages = (counts / total) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [f"Não Potável (0)\n{counts[0]} ({percentages[0]:.1f}%)",
         f"Potável (1)\n{counts[1]} ({percentages[1]:.1f}%)"],
        counts.values,
        color=["#e74c3c", "#2ecc71"],
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Distribuição da Variável Alvo (Potability)", fontsize=14)
    ax.set_xlabel("Classe", fontsize=12)
    ax.set_ylabel("Contagem", fontsize=12)

    # Adicionar valores nas barras
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + total * 0.01,
            f"{count} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()

    filepath = os.path.join(output_dir, "distribuicao_variavel_alvo.png")
    fig.savefig(filepath, dpi=MIN_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        f"Distribuição da variável alvo salva: {filepath} | "
        f"Não potável: {counts[0]} ({percentages[0]:.1f}%), "
        f"Potável: {counts[1]} ({percentages[1]:.1f}%)"
    )
    return filepath


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica e reporta valores ausentes por atributo.

    Calcula a contagem e o percentual de valores ausentes (NaN) para cada
    atributo do dataset, reportando os resultados no log de execução.

    Args:
        df: DataFrame contendo os dados do dataset Water Potability.

    Returns:
        DataFrame com colunas 'contagem' e 'percentual' indexado pelos nomes
        dos atributos, ordenado por contagem decrescente.
    """
    missing_count = df[EXPECTED_NUMERIC_COLUMNS].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    missing_report = pd.DataFrame({
        "contagem": missing_count,
        "percentual": missing_pct.round(2),
    }).sort_values("contagem", ascending=False)

    logger.info("Relatório de valores ausentes por atributo:")
    logger.info(f"\n{missing_report.to_string()}")

    # Indicar atributos com valores ausentes
    attrs_with_missing = missing_report[missing_report["contagem"] > 0]
    if not attrs_with_missing.empty:
        logger.info(
            f"Atributos com valores ausentes: "
            f"{', '.join(attrs_with_missing.index.tolist())}"
        )
    else:
        logger.info("Nenhum atributo possui valores ausentes.")

    return missing_report


def run_eda(df: pd.DataFrame, output_dir: str = "reports/figures/") -> dict:
    """Executa a análise exploratória completa e registra artefatos no MLflow.

    Orquestra todas as funções de EDA em sequência: validação do DataFrame,
    estatísticas descritivas, histogramas de distribuição, matriz de correlação,
    distribuição da variável alvo e relatório de valores ausentes. Todos os
    artefatos gerados são registrados no MLflow.

    Args:
        df: DataFrame contendo os dados do dataset Water Potability.
        output_dir: Diretório onde as visualizações serão salvas.
            Padrão: "reports/figures/".

    Returns:
        Dicionário com os resultados da EDA contendo chaves:
        - 'statistics': DataFrame com estatísticas descritivas
        - 'distribution_plots': lista de caminhos dos histogramas
        - 'correlation_matrix': caminho do heatmap de correlação
        - 'target_distribution': caminho do gráfico da variável alvo
        - 'missing_values': DataFrame com relatório de valores ausentes
    """
    # Validação do DataFrame de entrada (Req 2.6)
    if not _validate_dataframe(df):
        return {}

    logger.info("Iniciando Análise Exploratória de Dados (EDA)...")

    # Configurar MLflow
    setup_experiment()
    run = create_run(stage="eda")

    try:
        # 1. Estatísticas descritivas (Req 2.1)
        logger.info("--- Etapa 1: Estatísticas Descritivas ---")
        stats = compute_descriptive_statistics(df)

        # Salvar estatísticas como CSV para registro como artefato
        os.makedirs(output_dir, exist_ok=True)
        stats_path = os.path.join(output_dir, "estatisticas_descritivas.csv")
        stats.to_csv(stats_path)
        safe_log_artifact(stats_path)

        # 2. Histogramas de distribuição (Req 2.2, 2.4)
        logger.info("--- Etapa 2: Histogramas de Distribuição ---")
        distribution_plots = generate_distribution_plots(df, output_dir)
        for plot_path in distribution_plots:
            safe_log_artifact(plot_path)

        # 3. Matriz de correlação (Req 2.2, 2.4)
        logger.info("--- Etapa 3: Matriz de Correlação ---")
        correlation_path = generate_correlation_matrix(df, output_dir)
        safe_log_artifact(correlation_path)

        # 4. Distribuição da variável alvo (Req 2.2, 2.4)
        logger.info("--- Etapa 4: Distribuição da Variável Alvo ---")
        target_path = generate_target_distribution(df, output_dir)
        safe_log_artifact(target_path)

        # 5. Relatório de valores ausentes (Req 2.3, 2.5)
        logger.info("--- Etapa 5: Relatório de Valores Ausentes ---")
        missing_report = report_missing_values(df)

        # Salvar relatório como CSV para registro como artefato
        missing_path = os.path.join(output_dir, "relatorio_valores_ausentes.csv")
        missing_report.to_csv(missing_path)
        safe_log_artifact(missing_path)

        logger.info("Análise Exploratória de Dados (EDA) concluída com sucesso.")

        return {
            "statistics": stats,
            "distribution_plots": distribution_plots,
            "correlation_matrix": correlation_path,
            "target_distribution": target_path,
            "missing_values": missing_report,
        }

    except Exception as e:
        logger.error(f"Erro durante a execução da EDA: {e}")
        raise

    finally:
        # Encerrar run do MLflow
        import mlflow
        try:
            mlflow.end_run()
        except Exception:
            pass
