"""Configuração de logging padrão do pipeline de classificação de potabilidade da água.

Este módulo configura o sistema de logging utilizado por todos os módulos do
pipeline. Os níveis de log seguem a convenção:

- INFO: Progresso normal (início/fim de etapas, resumos)
- WARNING: Alertas não-críticos (>80% valores ausentes, fallback MLflow)
- ERROR: Falhas que interrompem uma etapa mas não o pipeline inteiro
- CRITICAL: Falhas que interrompem o pipeline (dataset inválido)
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configura o logging padrão do projeto.

    Configura o formato de saída, nível de log e handler para stdout,
    garantindo que todas as mensagens do pipeline sejam exibidas de forma
    consistente e informativa.

    Args:
        level: Nível mínimo de logging. Padrão é INFO.
    """
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Obtém um logger configurado para o módulo especificado.

    Args:
        name: Nome do módulo (tipicamente __name__).

    Returns:
        Instância de Logger configurada com o nome do módulo.
    """
    return logging.getLogger(name)
