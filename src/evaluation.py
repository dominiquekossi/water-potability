"""Módulo de avaliação e comparação de modelos de classificação.

Este módulo implementa as funções de avaliação dos modelos treinados,
incluindo cálculo de métricas (Accuracy, Precision, Recall, F1-Score,
AUC-ROC), geração de visualizações (matriz de confusão e curva ROC),
relatório comparativo e seleção do melhor modelo. Todos os resultados
são registrados no MLflow para rastreamento de experimentos.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

from src.logger import get_logger
from src.mlflow_utils import create_run, safe_log_artifact, safe_log_metric
from src.types import EvaluationResult, TrainedModelResult

import mlflow

logger = get_logger(__name__)


def compute_metrics(model: object, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Calcula Accuracy, Precision, Recall, F1-Score e AUC-ROC.

    Realiza predições com o modelo fornecido no conjunto de teste e calcula
    todas as métricas de classificação binária com no mínimo 4 casas decimais.

    Args:
        model: Modelo sklearn/xgboost treinado com métodos predict e
            predict_proba.
        X_test: Array com features do conjunto de teste.
        y_test: Array com rótulos verdadeiros do conjunto de teste.

    Returns:
        Dicionário com as métricas calculadas: accuracy, precision, recall,
        f1_score e auc_roc, cada uma com 4 casas decimais.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred, zero_division=0), 4)
    recall = round(recall_score(y_test, y_pred, zero_division=0), 4)
    f1 = round(f1_score(y_test, y_pred, zero_division=0), 4)
    auc_roc = round(roc_auc_score(y_test, y_proba), 4)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
    }

    logger.info(
        f"Métricas calculadas - Accuracy: {accuracy}, Precision: {precision}, "
        f"Recall: {recall}, F1-Score: {f1}, AUC-ROC: {auc_roc}"
    )

    return metrics


def generate_confusion_matrix_plot(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    output_dir: str,
) -> str:
    """Gera e salva a matriz de confusão como PNG.

    Cria uma visualização da matriz de confusão do modelo no conjunto de
    teste e salva como arquivo de imagem PNG com resolução de 150 DPI.

    Args:
        model: Modelo sklearn/xgboost treinado com método predict.
        X_test: Array com features do conjunto de teste.
        y_test: Array com rótulos verdadeiros do conjunto de teste.
        model_name: Nome identificador do modelo para título e nome do arquivo.
        output_dir: Diretório de saída para salvar o arquivo PNG.

    Returns:
        Caminho completo do arquivo PNG gerado.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Não Potável", "Potável"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matriz de Confusão - {model_name}", fontsize=14)
    ax.set_xlabel("Classe Predita", fontsize=12)
    ax.set_ylabel("Classe Verdadeira", fontsize=12)

    filepath = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Matriz de confusão salva em: {filepath}")
    return filepath


def generate_roc_curve_plot(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    output_dir: str,
) -> str:
    """Gera e salva a curva ROC como PNG.

    Cria uma visualização da curva ROC (Receiver Operating Characteristic)
    do modelo no conjunto de teste e salva como arquivo de imagem PNG com
    resolução de 150 DPI.

    Args:
        model: Modelo sklearn/xgboost treinado com método predict_proba.
        X_test: Array com features do conjunto de teste.
        y_test: Array com rótulos verdadeiros do conjunto de teste.
        model_name: Nome identificador do modelo para título e nome do arquivo.
        output_dir: Diretório de saída para salvar o arquivo PNG.

    Returns:
        Caminho completo do arquivo PNG gerado.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Aleatório")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taxa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiros Positivos", fontsize=12)
    ax.set_title(f"Curva ROC - {model_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(output_dir, f"roc_curve_{model_name}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Curva ROC salva em: {filepath}")
    return filepath


def generate_comparative_report(results: list[dict]) -> pd.DataFrame:
    """Gera relatório comparativo ordenado por F1-Score decrescente.

    Cria um DataFrame tabular com todas as métricas de avaliação de todos
    os modelos, ordenado por F1-Score em ordem decrescente para facilitar
    a comparação de desempenho.

    Args:
        results: Lista de dicionários EvaluationResult contendo as métricas
            de cada modelo avaliado.

    Returns:
        DataFrame pandas com colunas (model_name, accuracy, precision,
        recall, f1_score, auc_roc) ordenado por f1_score decrescente.
    """
    report_data = []
    for result in results:
        report_data.append(
            {
                "model_name": result["model_name"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "auc_roc": result["auc_roc"],
            }
        )

    df_report = pd.DataFrame(report_data)
    df_report = df_report.sort_values(by="f1_score", ascending=False).reset_index(
        drop=True
    )

    logger.info("Relatório comparativo gerado:")
    logger.info(f"\n{df_report.to_string(index=False)}")

    return df_report


def select_best_model(results: list[dict]) -> dict:
    """Seleciona melhor modelo por F1-Score (desempate por AUC-ROC).

    Identifica o modelo com maior F1-Score entre todos os modelos avaliados.
    Em caso de empate no F1-Score, utiliza o AUC-ROC como critério de
    desempate, selecionando o modelo com maior AUC-ROC.

    Args:
        results: Lista de dicionários EvaluationResult contendo as métricas
            de cada modelo avaliado.

    Returns:
        Dicionário EvaluationResult do melhor modelo selecionado.

    Raises:
        ValueError: Se a lista de resultados estiver vazia.
    """
    if not results:
        raise ValueError("Lista de resultados vazia. Nenhum modelo para selecionar.")

    best = max(results, key=lambda x: (x["f1_score"], x["auc_roc"]))

    logger.info(
        f"Melhor modelo selecionado: {best['model_name']} "
        f"(F1-Score: {best['f1_score']}, AUC-ROC: {best['auc_roc']})"
    )

    return best


def run_evaluation(
    trained_models: list[dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "reports/figures/",
) -> dict:
    """Executa avaliação completa de todos os modelos.

    Orquestra o fluxo completo de avaliação: filtra modelos com falha no
    treinamento, calcula métricas, gera visualizações, cria relatório
    comparativo, seleciona o melhor modelo e registra tudo no MLflow.

    Args:
        trained_models: Lista de dicionários TrainedModelResult retornados
            pelo módulo de treinamento.
        X_test: Array com features do conjunto de teste.
        y_test: Array com rótulos verdadeiros do conjunto de teste.
        output_dir: Diretório de saída para artefatos visuais.
            Padrão: "reports/figures/".

    Returns:
        Dicionário com chaves:
            - results: Lista de EvaluationResult de cada modelo avaliado.
            - comparative_report: DataFrame com relatório comparativo.
            - best_model: EvaluationResult do melhor modelo.
    """
    logger.info("Iniciando avaliação de modelos...")

    # Criar run MLflow para a etapa de avaliação
    run = create_run(stage="avaliacao")

    evaluation_results: list[dict] = []

    for trained_model in trained_models:
        model_name = trained_model["model_name"]
        status = trained_model["status"]

        # Req 5.7: Se modelo falhou no treinamento, registrar ausência e pular
        if status != "success":
            error_msg = trained_model.get("error_message", "Erro desconhecido")
            logger.warning(
                f"Modelo '{model_name}' indisponível para avaliação "
                f"(status: {status}, erro: {error_msg}). "
                "Excluindo do relatório comparativo."
            )
            continue

        model = trained_model["model"]

        logger.info(f"Avaliando modelo: {model_name}")

        # Calcular métricas
        metrics = compute_metrics(model, X_test, y_test)

        # Gerar visualizações
        cm_path = generate_confusion_matrix_plot(
            model, X_test, y_test, model_name, output_dir
        )
        roc_path = generate_roc_curve_plot(
            model, X_test, y_test, model_name, output_dir
        )

        # Registrar métricas no MLflow
        for metric_name, metric_value in metrics.items():
            safe_log_metric(f"{model_name}_{metric_name}", metric_value)

        # Registrar artefatos visuais no MLflow
        safe_log_artifact(cm_path)
        safe_log_artifact(roc_path)

        # Montar resultado da avaliação
        eval_result: EvaluationResult = {
            "model_name": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "auc_roc": metrics["auc_roc"],
            "confusion_matrix_path": cm_path,
            "roc_curve_path": roc_path,
        }
        evaluation_results.append(eval_result)

    # Verificar se há modelos avaliados
    if not evaluation_results:
        logger.error(
            "Nenhum modelo disponível para avaliação. "
            "Todos os modelos falharam no treinamento."
        )
        if run:
            mlflow.end_run()
        return {
            "results": [],
            "comparative_report": pd.DataFrame(),
            "best_model": None,
        }

    # Gerar relatório comparativo
    comparative_report = generate_comparative_report(evaluation_results)

    # Salvar relatório como CSV
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_comparativo.csv")
    comparative_report.to_csv(report_path, index=False)
    safe_log_artifact(report_path)
    logger.info(f"Relatório comparativo salvo em: {report_path}")

    # Selecionar melhor modelo
    best_model = select_best_model(evaluation_results)

    # Registrar melhor modelo no MLflow
    safe_log_metric("best_model_f1_score", best_model["f1_score"])
    safe_log_metric("best_model_auc_roc", best_model["auc_roc"])

    logger.info(
        f"Avaliação concluída. Melhor modelo: {best_model['model_name']} "
        f"(F1-Score: {best_model['f1_score']}, AUC-ROC: {best_model['auc_roc']})"
    )

    # Encerrar run MLflow
    if run:
        mlflow.end_run()

    return {
        "results": evaluation_results,
        "comparative_report": comparative_report,
        "best_model": best_model,
    }
