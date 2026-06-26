"""Módulo de integração com MLflow para rastreamento de experimentos.

Este módulo gerencia toda a interação com o MLflow, incluindo configuração
de experimentos, criação de runs, registro de métricas e artefatos. Implementa
um padrão de resiliência com fallback para armazenamento local caso o
MLflow tracking server esteja indisponível.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from src.logger import get_logger

logger = get_logger(__name__)

# Diretório de fallback para armazenamento local quando MLflow falha
FALLBACK_DIR = Path("mlflow_fallback")


def setup_experiment(experiment_name: str = "water-potability-classification") -> str:
    """Configura experimento MLflow, criando se necessário.

    Configura o tracking URI e cria ou recupera o experimento com o nome
    especificado. Se o MLflow estiver indisponível, registra um aviso e
    retorna um ID de fallback.

    Args:
        experiment_name: Nome do experimento MLflow.

    Returns:
        ID do experimento configurado, ou "fallback" se MLflow indisponível.
    """
    try:
        mlflow.set_tracking_uri("mlruns")
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(
                f"Experimento '{experiment_name}' criado com ID: {experiment_id}"
            )
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Experimento '{experiment_name}' encontrado com ID: {experiment_id}"
            )

        mlflow.set_experiment(experiment_name)
        return experiment_id

    except Exception as e:
        logger.warning(
            f"MLflow indisponível ao configurar experimento: {e}. "
            "Usando armazenamento local como fallback."
        )
        return "fallback"


def create_run(stage: str, tags: dict = None) -> object:
    """Cria um run MLflow com tags de identificação.

    Cria um novo run dentro do experimento ativo com tags que identificam
    a etapa do pipeline e o timestamp de início em formato ISO 8601.

    Args:
        stage: Etapa do pipeline (coleta, eda, preprocessamento,
            treinamento, avaliacao).
        tags: Dicionário opcional com tags adicionais.

    Returns:
        Objeto do run MLflow ativo, ou None se MLflow indisponível.
    """
    timestamp_iso = datetime.now(timezone.utc).isoformat()

    run_tags = {
        "stage": stage,
        "start_timestamp": timestamp_iso,
    }

    if tags:
        run_tags.update(tags)

    try:
        run = mlflow.start_run(tags=run_tags)
        logger.info(
            f"Run MLflow criado para etapa '{stage}' "
            f"(timestamp: {timestamp_iso})"
        )
        return run

    except Exception as e:
        logger.warning(
            f"MLflow indisponível ao criar run: {e}. "
            "Usando armazenamento local como fallback."
        )
        # Salvar informações do run localmente como fallback
        _save_fallback_run_info(stage, run_tags)
        return None


def log_environment_info() -> None:
    """Registra versão do código, timestamp e versões de bibliotecas.

    Registra automaticamente no MLflow:
    - Git commit hash (quando disponível)
    - Timestamp de execução em formato ISO 8601
    - Versões das bibliotecas principais (pandas, scikit-learn, xgboost, mlflow)
    """
    timestamp_iso = datetime.now(timezone.utc).isoformat()
    git_hash = _get_git_commit_hash()

    # Obter versões das bibliotecas
    import pandas as pd
    import sklearn
    import xgboost as xgb

    env_info = {
        "execution_timestamp": timestamp_iso,
        "git_commit_hash": git_hash or "não disponível",
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
        "mlflow_version": mlflow.__version__,
    }

    try:
        for key, value in env_info.items():
            mlflow.log_param(key, value)

        logger.info(
            f"Informações de ambiente registradas: "
            f"git={git_hash or 'N/A'}, "
            f"pandas={pd.__version__}, "
            f"sklearn={sklearn.__version__}, "
            f"xgboost={xgb.__version__}, "
            f"mlflow={mlflow.__version__}"
        )

    except Exception as e:
        logger.warning(
            f"MLflow indisponível ao registrar ambiente: {e}. "
            "Salvando localmente."
        )
        _save_fallback_data("environment_info.json", env_info)


def safe_log_metric(key: str, value: float) -> None:
    """Registra métrica com fallback para armazenamento local.

    Tenta registrar a métrica no MLflow. Em caso de falha, salva
    a métrica em um arquivo JSON local para recuperação posterior.

    Args:
        key: Nome da métrica.
        value: Valor numérico da métrica.
    """

    def _log_metric():
        mlflow.log_metric(key, value)

    result = safe_mlflow_operation(_log_metric)
    if result is None:
        # Fallback já tratado dentro de safe_mlflow_operation
        pass
    else:
        logger.info(f"Métrica registrada: {key}={value}")


def safe_log_artifact(local_path: str) -> None:
    """Registra artefato com fallback para armazenamento local.

    Tenta registrar o artefato no MLflow. Em caso de falha, registra
    o caminho do artefato em um arquivo de log local.

    Args:
        local_path: Caminho local do arquivo a ser registrado como artefato.
    """

    def _log_artifact():
        mlflow.log_artifact(local_path)

    result = safe_mlflow_operation(_log_artifact)
    if result is None:
        # Fallback já tratado dentro de safe_mlflow_operation
        pass
    else:
        logger.info(f"Artefato registrado: {local_path}")


def safe_mlflow_operation(operation_fn, *args, **kwargs):
    """Executa operação MLflow com fallback para armazenamento local.

    Wrapper de resiliência que tenta executar uma operação MLflow e,
    em caso de falha, salva os dados localmente sem interromper a
    execução do pipeline.

    Args:
        operation_fn: Função de operação MLflow a ser executada.
        *args: Argumentos posicionais para a operação.
        **kwargs: Argumentos nomeados para a operação.

    Returns:
        Resultado da operação se bem-sucedida, ou None em caso de fallback.
    """
    try:
        return operation_fn(*args, **kwargs)
    except Exception as e:
        logger.warning(
            f"MLflow indisponível: {e}. Usando armazenamento local."
        )
        _save_fallback_data(
            "failed_operations.json",
            {
                "operation": operation_fn.__name__ if hasattr(operation_fn, '__name__') else str(operation_fn),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "args": str(args),
                "kwargs": str(kwargs),
            },
        )
        return None



def _get_git_commit_hash() -> str | None:
    """Obtém o hash do commit git atual.

    Returns:
        Hash curto do commit atual ou None se não disponível.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _save_fallback_run_info(stage: str, tags: dict) -> None:
    """Salva informações de run localmente como fallback.

    Args:
        stage: Etapa do pipeline.
        tags: Tags do run.
    """
    fallback_data = {
        "stage": stage,
        "tags": tags,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _save_fallback_data(f"run_{stage}.json", fallback_data)


def _save_fallback_data(filename: str, data: dict) -> None:
    """Salva dados em arquivo JSON local como fallback.

    Args:
        filename: Nome do arquivo de fallback.
        data: Dados a serem salvos.
    """
    try:
        FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        filepath = FALLBACK_DIR / filename

        # Se arquivo já existe, carregar e adicionar ao histórico
        existing_data = []
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        existing_data = content
                    else:
                        existing_data = [content]
            except (json.JSONDecodeError, IOError):
                existing_data = []

        existing_data.append(data)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Dados salvos localmente em: {filepath}")

    except Exception as e:
        logger.error(f"Falha ao salvar fallback local: {e}")
