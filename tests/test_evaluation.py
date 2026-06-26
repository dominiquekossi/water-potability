"""Testes unitários para o módulo de avaliação de modelos.

Este módulo contém testes unitários e de propriedade para verificar a
corretude das funções de avaliação do pipeline de classificação de
potabilidade da água.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation import (
    compute_metrics,
    generate_comparative_report,
    generate_confusion_matrix_plot,
    generate_roc_curve_plot,
    run_evaluation,
    select_best_model,
)


@pytest.fixture
def trained_classifier():
    """Fixture que retorna um classificador treinado simples."""
    np.random.seed(42)
    X_train = np.random.randn(200, 5)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def test_data():
    """Fixture que retorna dados de teste."""
    np.random.seed(123)
    X_test = np.random.randn(50, 5)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    return X_test, y_test


@pytest.fixture
def sample_results():
    """Fixture que retorna resultados de avaliação de exemplo."""
    return [
        {
            "model_name": "model_a",
            "accuracy": 0.8500,
            "precision": 0.8200,
            "recall": 0.7800,
            "f1_score": 0.7995,
            "auc_roc": 0.8900,
            "confusion_matrix_path": "path_a_cm.png",
            "roc_curve_path": "path_a_roc.png",
        },
        {
            "model_name": "model_b",
            "accuracy": 0.9000,
            "precision": 0.8800,
            "recall": 0.8500,
            "f1_score": 0.8647,
            "auc_roc": 0.9200,
            "confusion_matrix_path": "path_b_cm.png",
            "roc_curve_path": "path_b_roc.png",
        },
        {
            "model_name": "model_c",
            "accuracy": 0.7500,
            "precision": 0.7200,
            "recall": 0.6800,
            "f1_score": 0.6994,
            "auc_roc": 0.7800,
            "confusion_matrix_path": "path_c_cm.png",
            "roc_curve_path": "path_c_roc.png",
        },
    ]


class TestComputeMetrics:
    """Testes para a função compute_metrics."""

    def test_metrics_keys(self, trained_classifier, test_data):
        """Verifica que as métricas retornadas contêm todas as chaves esperadas."""
        X_test, y_test = test_data
        metrics = compute_metrics(trained_classifier, X_test, y_test)

        expected_keys = {"accuracy", "precision", "recall", "f1_score", "auc_roc"}
        assert set(metrics.keys()) == expected_keys

    def test_metrics_range(self, trained_classifier, test_data):
        """Verifica que todas as métricas estão no intervalo [0, 1]."""
        X_test, y_test = test_data
        metrics = compute_metrics(trained_classifier, X_test, y_test)

        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"Métrica {key} fora do intervalo: {value}"

    def test_metrics_decimal_places(self, trained_classifier, test_data):
        """Verifica que as métricas possuem no máximo 4 casas decimais."""
        X_test, y_test = test_data
        metrics = compute_metrics(trained_classifier, X_test, y_test)

        for key, value in metrics.items():
            # Verificar que não tem mais de 4 casas decimais
            assert round(value, 4) == value, (
                f"Métrica {key} com mais de 4 casas decimais: {value}"
            )


class TestGenerateConfusionMatrixPlot:
    """Testes para a função generate_confusion_matrix_plot."""

    def test_generates_png_file(self, trained_classifier, test_data):
        """Verifica que um arquivo PNG é gerado."""
        X_test, y_test = test_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_confusion_matrix_plot(
                trained_classifier, X_test, y_test, "test_model", tmpdir
            )
            assert os.path.exists(path)
            assert path.endswith(".png")

    def test_filename_contains_model_name(self, trained_classifier, test_data):
        """Verifica que o nome do arquivo contém o nome do modelo."""
        X_test, y_test = test_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_confusion_matrix_plot(
                trained_classifier, X_test, y_test, "logistic_regression", tmpdir
            )
            assert "logistic_regression" in os.path.basename(path)


class TestGenerateRocCurvePlot:
    """Testes para a função generate_roc_curve_plot."""

    def test_generates_png_file(self, trained_classifier, test_data):
        """Verifica que um arquivo PNG é gerado."""
        X_test, y_test = test_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_roc_curve_plot(
                trained_classifier, X_test, y_test, "test_model", tmpdir
            )
            assert os.path.exists(path)
            assert path.endswith(".png")

    def test_filename_contains_model_name(self, trained_classifier, test_data):
        """Verifica que o nome do arquivo contém o nome do modelo."""
        X_test, y_test = test_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_roc_curve_plot(
                trained_classifier, X_test, y_test, "random_forest", tmpdir
            )
            assert "random_forest" in os.path.basename(path)


class TestGenerateComparativeReport:
    """Testes para a função generate_comparative_report."""

    def test_report_sorted_by_f1_descending(self, sample_results):
        """Verifica que o relatório está ordenado por F1-Score decrescente."""
        report = generate_comparative_report(sample_results)

        f1_scores = report["f1_score"].tolist()
        assert f1_scores == sorted(f1_scores, reverse=True)

    def test_report_contains_all_models(self, sample_results):
        """Verifica que o relatório contém todos os modelos."""
        report = generate_comparative_report(sample_results)
        assert len(report) == len(sample_results)

    def test_report_columns(self, sample_results):
        """Verifica que o relatório contém todas as colunas esperadas."""
        report = generate_comparative_report(sample_results)
        expected_cols = {"model_name", "accuracy", "precision", "recall", "f1_score", "auc_roc"}
        assert set(report.columns) == expected_cols


class TestSelectBestModel:
    """Testes para a função select_best_model."""

    def test_selects_highest_f1(self, sample_results):
        """Verifica que seleciona o modelo com maior F1-Score."""
        best = select_best_model(sample_results)
        assert best["model_name"] == "model_b"
        assert best["f1_score"] == 0.8647

    def test_tiebreaker_by_auc_roc(self):
        """Verifica desempate por AUC-ROC quando F1-Score é igual."""
        tied_results = [
            {
                "model_name": "model_x",
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.78,
                "f1_score": 0.80,
                "auc_roc": 0.85,
                "confusion_matrix_path": "",
                "roc_curve_path": "",
            },
            {
                "model_name": "model_y",
                "accuracy": 0.86,
                "precision": 0.83,
                "recall": 0.77,
                "f1_score": 0.80,
                "auc_roc": 0.90,
                "confusion_matrix_path": "",
                "roc_curve_path": "",
            },
        ]
        best = select_best_model(tied_results)
        assert best["model_name"] == "model_y"

    def test_empty_results_raises_error(self):
        """Verifica que uma lista vazia gera ValueError."""
        with pytest.raises(ValueError):
            select_best_model([])


class TestRunEvaluation:
    """Testes para a função run_evaluation."""

    def test_skips_failed_models(self, test_data):
        """Verifica que modelos com falha são excluídos da avaliação."""
        X_test, y_test = test_data

        # Treinar um modelo válido
        np.random.seed(42)
        X_train = np.random.randn(200, 5)
        y_train = (X_train[:, 0] > 0).astype(int)
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)

        trained_models = [
            {
                "model_name": "good_model",
                "model": model,
                "best_params": {},
                "training_time_seconds": 1.0,
                "status": "success",
                "error_message": None,
            },
            {
                "model_name": "failed_model",
                "model": None,
                "best_params": {},
                "training_time_seconds": 0.0,
                "status": "failed",
                "error_message": "Training error",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_evaluation(trained_models, X_test, y_test, output_dir=tmpdir)

        assert len(result["results"]) == 1
        assert result["results"][0]["model_name"] == "good_model"
        assert result["best_model"]["model_name"] == "good_model"

    def test_all_failed_returns_empty(self, test_data):
        """Verifica comportamento quando todos os modelos falharam."""
        X_test, y_test = test_data

        trained_models = [
            {
                "model_name": "failed_1",
                "model": None,
                "best_params": {},
                "training_time_seconds": 0.0,
                "status": "failed",
                "error_message": "Error 1",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_evaluation(trained_models, X_test, y_test, output_dir=tmpdir)

        assert result["results"] == []
        assert result["best_model"] is None
