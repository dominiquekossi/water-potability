"""Módulo de treinamento de modelos de classificação de potabilidade da água.

Este módulo implementa o pipeline de treinamento completo, incluindo a
configuração de hiperparâmetros, busca otimizada via GridSearchCV com
validação cruzada, persistência dos modelos treinados em formato joblib,
e registro de experimentos no MLflow.
"""

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

import mlflow

from src.logger import get_logger
from src.mlflow_utils import safe_log_artifact, safe_log_metric
from src.types import TrainedModelResult

logger = get_logger(__name__)


def get_model_configs() -> dict:
    """Retorna configurações de modelos e grids de hiperparâmetros.

    Define os classificadores disponíveis para treinamento e seus respectivos
    grids de hiperparâmetros para busca otimizada. Os grids são dimensionados
    para execução em tempo razoável sem comprometer a qualidade da busca.

    Returns:
        Dicionário onde cada chave é o nome do modelo e o valor é um
        dicionário com as chaves:
            - model: Instância não treinada do classificador.
            - param_grid: Dicionário com grid de hiperparâmetros.
    """
    configs = {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
                "penalty": ["l2"],
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "xgboost": {
            "model": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
            ),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 1.0],
            },
        },
    }

    return configs


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[object, dict, float]:
    """Treina um modelo com busca de hiperparâmetros via Grid Search.

    Realiza busca exaustiva de hiperparâmetros utilizando GridSearchCV com
    validação cruzada estratificada, otimizando pela métrica F1-Score.

    Args:
        model_name: Nome identificador do modelo (usado para log).
        X_train: Array de features de treinamento.
        y_train: Array de labels de treinamento.
        param_grid: Dicionário com grid de hiperparâmetros para busca.
        cv_folds: Número de folds para validação cruzada (mínimo 5).
        random_state: Seed para reprodutibilidade.

    Returns:
        Tupla contendo:
            - Modelo treinado com os melhores hiperparâmetros.
            - Dicionário com os melhores hiperparâmetros encontrados.
            - Tempo de treinamento em segundos.
    """
    configs = get_model_configs()

    if model_name in configs:
        base_model = configs[model_name]["model"]
    else:
        raise ValueError(
            f"Modelo '{model_name}' não reconhecido. "
            f"Modelos disponíveis: {list(configs.keys())}"
        )

    logger.info(
        f"Iniciando treinamento do modelo '{model_name}' "
        f"com GridSearchCV ({cv_folds} folds, métrica: f1)"
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(
        f"Modelo '{model_name}' treinado em {training_time:.2f}s. "
        f"Melhor F1-Score (CV): {best_score:.4f}"
    )
    logger.info(f"Melhores hiperparâmetros: {best_params}")

    return best_model, best_params, training_time


def save_model(model: object, model_name: str, output_dir: str = "models/") -> str:
    """Salva modelo treinado em formato joblib.

    Persiste o modelo treinado em disco utilizando o formato joblib,
    criando o diretório de saída se necessário.

    Args:
        model: Instância do modelo treinado a ser salvo.
        model_name: Nome identificador do modelo (usado no nome do arquivo).
        output_dir: Diretório de destino para os modelos salvos.

    Returns:
        Caminho completo do arquivo do modelo salvo.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{model_name}.joblib"
    joblib.dump(model, file_path)

    logger.info(f"Modelo '{model_name}' salvo em: {file_path}")

    return str(file_path)


def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> list[dict]:
    """Treina todos os modelos configurados e registra no MLflow.

    Orquestra o treinamento de todos os classificadores definidos na
    configuração, realizando para cada modelo:
    1. Busca de hiperparâmetros com validação cruzada
    2. Registro de parâmetros e artefatos no MLflow
    3. Persistência do modelo treinado em formato joblib

    Em caso de falha no treinamento de um modelo individual, registra
    o erro e continua com os demais modelos.

    Args:
        X_train: Array de features de treinamento.
        y_train: Array de labels de treinamento.
        config: Dicionário de configuração com chaves:
            - training.cv_folds: Número de folds para CV.
            - training.models: Lista de nomes de modelos a treinar.
            - output.models_dir: Diretório para salvar modelos.
            - random_state: Seed para reprodutibilidade.

    Returns:
        Lista de dicionários TrainedModelResult com informações de cada
        modelo treinado, incluindo status de sucesso ou falha.
    """
    training_config = config.get("training", {})
    output_config = config.get("output", {})

    cv_folds = training_config.get("cv_folds", 5)
    model_names = training_config.get(
        "models", ["logistic_regression", "random_forest", "xgboost"]
    )
    models_dir = output_config.get("models_dir", "models/")
    random_state = config.get("random_state", 42)

    all_configs = get_model_configs()
    results: list[dict] = []

    logger.info("=" * 60)
    logger.info("INÍCIO DO PIPELINE DE TREINAMENTO")
    logger.info(f"Modelos a treinar: {model_names}")
    logger.info(f"Validação cruzada: {cv_folds} folds")
    logger.info(f"Métrica de otimização: F1-Score")
    logger.info("=" * 60)

    for model_name in model_names:
        logger.info("-" * 40)
        logger.info(f"Treinando modelo: {model_name}")
        logger.info("-" * 40)

        try:
            if model_name not in all_configs:
                raise ValueError(
                    f"Modelo '{model_name}' não encontrado nas configurações. "
                    f"Disponíveis: {list(all_configs.keys())}"
                )

            param_grid = all_configs[model_name]["param_grid"]

            # Treinar modelo com busca de hiperparâmetros
            best_model, best_params, training_time = train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                param_grid=param_grid,
                cv_folds=cv_folds,
                random_state=random_state,
            )

            # Salvar modelo em formato joblib
            model_path = save_model(
                model=best_model,
                model_name=model_name,
                output_dir=models_dir,
            )

            # Registrar no MLflow
            _log_model_to_mlflow(
                model_name=model_name,
                best_params=best_params,
                training_time=training_time,
                model_path=model_path,
            )

            # Montar resultado de sucesso
            result: TrainedModelResult = {
                "model_name": model_name,
                "model": best_model,
                "best_params": best_params,
                "training_time_seconds": training_time,
                "status": "success",
                "error_message": None,
            }
            results.append(result)

            logger.info(
                f"Modelo '{model_name}' treinado com sucesso "
                f"em {training_time:.2f}s."
            )

        except Exception as e:
            # Registrar erro e continuar com próximo modelo
            error_msg = str(e)
            logger.error(
                f"Falha no treinamento do modelo '{model_name}': {error_msg}"
            )

            result: TrainedModelResult = {
                "model_name": model_name,
                "model": None,
                "best_params": {},
                "training_time_seconds": 0.0,
                "status": "failed",
                "error_message": error_msg,
            }
            results.append(result)

    # Resumo final
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    logger.info("=" * 60)
    logger.info("PIPELINE DE TREINAMENTO CONCLUÍDO")
    logger.info(f"  Modelos treinados com sucesso: {len(successful)}")
    logger.info(f"  Modelos com falha: {len(failed)}")
    if failed:
        for f in failed:
            logger.info(f"    - {f['model_name']}: {f['error_message']}")
    logger.info("=" * 60)

    return results


def _log_model_to_mlflow(
    model_name: str,
    best_params: dict,
    training_time: float,
    model_path: str,
) -> None:
    """Registra informações do modelo treinado no MLflow.

    Registra hiperparâmetros, tempo de treinamento e o modelo salvo
    como artefato no run MLflow ativo.

    Args:
        model_name: Nome identificador do modelo.
        best_params: Dicionário com melhores hiperparâmetros encontrados.
        training_time: Tempo de treinamento em segundos.
        model_path: Caminho do arquivo do modelo salvo.
    """
    try:
        # Registrar hiperparâmetros
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"{model_name}_{param_name}", param_value)

        # Registrar tempo de treinamento como métrica
        safe_log_metric(f"{model_name}_training_time_seconds", training_time)

        # Registrar modelo como artefato
        safe_log_artifact(model_path)

        logger.info(
            f"Informações do modelo '{model_name}' registradas no MLflow."
        )

    except Exception as e:
        logger.warning(
            f"Falha ao registrar modelo '{model_name}' no MLflow: {e}. "
            "Modelo salvo localmente."
        )
