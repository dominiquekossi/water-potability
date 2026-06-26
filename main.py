"""Orquestrador principal do pipeline de classificação de potabilidade da água.

Este módulo é o ponto de entrada único do pipeline. Carrega as configurações
do arquivo YAML, configura o experimento MLflow e executa sequencialmente
todas as etapas: coleta de dados, análise exploratória (EDA), preprocessamento,
treinamento de modelos e avaliação. Ao final, exibe um resumo indicando a
conclusão de todas as etapas e onde acessar os resultados.

Uso:
    python main.py
"""

import sys
import time
from pathlib import Path

import yaml

from src.logger import setup_logging, get_logger

# Configurar logging antes de qualquer operação
setup_logging()
logger = get_logger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Carrega as configurações do pipeline a partir de um arquivo YAML.

    Args:
        config_path: Caminho para o arquivo de configuração YAML.

    Returns:
        Dicionário com todas as configurações do pipeline.

    Raises:
        FileNotFoundError: Se o arquivo de configuração não for encontrado.
        ValueError: Se o arquivo YAML não puder ser interpretado.
    """
    logger.info(f"Carregando configurações de: {config_path}")

    config_file = Path(config_path)
    if not config_file.exists():
        msg = (
            f"Arquivo de configuração não encontrado: '{config_path}'. "
            f"Verifique se o arquivo existe no diretório do projeto."
        )
        logger.critical(msg)
        raise FileNotFoundError(msg)

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Erro ao interpretar o arquivo YAML '{config_path}': {e}"
        logger.critical(msg)
        raise ValueError(msg)

    logger.info("Configurações carregadas com sucesso.")
    return config


def main(config_path: str = "configs/config.yaml") -> None:
    """Executa o pipeline completo de classificação de potabilidade da água.

    Orquestra a execução sequencial de todas as etapas do pipeline:
    1. Coleta e validação de dados
    2. Análise Exploratória de Dados (EDA)
    3. Preprocessamento (imputação, split, escalonamento, balanceamento)
    4. Treinamento de modelos de classificação
    5. Avaliação e comparação dos modelos

    Cada etapa é executada dentro de um bloco de tratamento de erros individual,
    de modo que falhas em uma etapa geram mensagens claras indicando a etapa
    que falhou e o motivo do erro.

    Args:
        config_path: Caminho para o arquivo de configuração YAML.
            Padrão: "configs/config.yaml".
    """
    logger.info("=" * 70)
    logger.info("PIPELINE DE CLASSIFICAÇÃO DE POTABILIDADE DA ÁGUA")
    logger.info("=" * 70)

    inicio_pipeline = time.time()

    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Falha ao carregar configurações: {e}")
        sys.exit(1)

    try:
        from src.mlflow_utils import setup_experiment

        experiment_name = config.get("mlflow", {}).get(
            "experiment_name", "water-potability-classification"
        )
        experiment_id = setup_experiment(experiment_name)
        logger.info(f"Experimento MLflow configurado: ID={experiment_id}")
    except Exception as e:
        logger.warning(
            f"Não foi possível configurar o experimento MLflow: {e}. "
            "O pipeline continuará sem rastreamento MLflow."
        )

    logger.info("-" * 70)
    logger.info("ETAPA 1/5: Coleta e Validação de Dados")
    logger.info("-" * 70)

    try:
        from src.data_collection import load_dataset, validate_dataset, log_dataset_summary

        dataset_path = config.get("dataset_path", "water_potability.csv")
        df = load_dataset(dataset_path)
        df = validate_dataset(df)
        summary = log_dataset_summary(df)
        logger.info("Etapa de coleta de dados concluída com sucesso.")
    except Exception as e:
        logger.critical(f"FALHA NA ETAPA 'Coleta de Dados': {e}")
        sys.exit(1)

    logger.info("-" * 70)
    logger.info("ETAPA 2/5: Análise Exploratória de Dados (EDA)")
    logger.info("-" * 70)

    try:
        from src.eda import run_eda

        figures_dir = config.get("output", {}).get("figures_dir", "reports/figures/")
        eda_results = run_eda(df, output_dir=figures_dir)
        logger.info("Etapa de EDA concluída com sucesso.")
    except Exception as e:
        logger.critical(f"FALHA NA ETAPA 'Análise Exploratória (EDA)': {e}")
        sys.exit(1)

    logger.info("-" * 70)
    logger.info("ETAPA 3/5: Preprocessamento de Dados")
    logger.info("-" * 70)

    try:
        from src.preprocessing import run_preprocessing

        X_train, X_test, y_train, y_test, preprocess_params = run_preprocessing(
            df, config
        )
        logger.info("Etapa de preprocessamento concluída com sucesso.")
    except Exception as e:
        logger.critical(f"FALHA NA ETAPA 'Preprocessamento': {e}")
        sys.exit(1)

    logger.info("-" * 70)
    logger.info("ETAPA 4/5: Treinamento de Modelos")
    logger.info("-" * 70)

    try:
        from src.training import run_training

        trained_models = run_training(X_train, y_train, config)

        modelos_sucesso = [m for m in trained_models if m.get("status") == "success"]
        modelos_falha = [m for m in trained_models if m.get("status") != "success"]

        if modelos_falha:
            for m in modelos_falha:
                logger.warning(
                    f"Modelo '{m.get('model_name', 'desconhecido')}' falhou: "
                    f"{m.get('error_message', 'erro desconhecido')}"
                )

        if not modelos_sucesso:
            raise RuntimeError(
                "Nenhum modelo foi treinado com sucesso. "
                "Verifique os logs para detalhes das falhas individuais."
            )

        logger.info(
            f"Etapa de treinamento concluída: "
            f"{len(modelos_sucesso)} modelo(s) treinado(s) com sucesso, "
            f"{len(modelos_falha)} falha(s)."
        )
    except Exception as e:
        logger.critical(f"FALHA NA ETAPA 'Treinamento': {e}")
        sys.exit(1)

    logger.info("-" * 70)
    logger.info("ETAPA 5/5: Avaliação e Comparação de Modelos")
    logger.info("-" * 70)

    try:
        from src.evaluation import run_evaluation

        evaluation_results = run_evaluation(
            trained_models, X_test, y_test, output_dir=figures_dir
        )
        logger.info("Etapa de avaliação concluída com sucesso.")
    except Exception as e:
        logger.critical(f"FALHA NA ETAPA 'Avaliação': {e}")
        sys.exit(1)

    tempo_total = time.time() - inicio_pipeline

    logger.info("=" * 70)
    logger.info("PIPELINE CONCLUÍDO COM SUCESSO")
    logger.info("=" * 70)
    logger.info(f"Tempo total de execução: {tempo_total:.2f} segundos")
    logger.info("")
    logger.info("Resumo das etapas executadas:")
    logger.info("  ✓ 1. Coleta e validação de dados")
    logger.info("  ✓ 2. Análise Exploratória de Dados (EDA)")
    logger.info("  ✓ 3. Preprocessamento de dados")
    logger.info("  ✓ 4. Treinamento de modelos")
    logger.info("  ✓ 5. Avaliação e comparação de modelos")
    logger.info("")
    logger.info("Acesso aos resultados:")
    logger.info(f"  • Visualizações: {figures_dir}")
    logger.info(f"  • Modelos salvos: {config.get('output', {}).get('models_dir', 'models/')}")
    logger.info(
        f"  • MLflow UI: execute 'mlflow ui --port "
        f"{config.get('mlflow', {}).get('port', 5000)}' "
        f"e acesse http://localhost:{config.get('mlflow', {}).get('port', 5000)}"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
