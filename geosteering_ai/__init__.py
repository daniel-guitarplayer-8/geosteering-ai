"""Geosteering AI — Pipeline de Inversao Geofisica 1D com Deep Learning.

Reproduz, com fidelidade fisica, a inversao eletromagnetica em tempo real
atraves de arquiteturas de Deep Learning. Suporta componentes EM + geosinais
e/ou Feature Views como features para inversao de resistividade em cenarios
de inferencia offline (acausal) e realtime causal para geosteering em
ambientes ruidosos (on-the-fly).

Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
Autor: Daniel Leal.

Modulos:
    config          PipelineConfig dataclass (ponto unico de verdade)
    data            Loading, splitting, FV, GS, scaling, DataPipeline
    noise           Noise on-the-fly (gaussian, multiplicative, curriculum)
    models          44 arquiteturas (39 standard + 5 geosteering)
    losses          26 funcoes de perda (13 gen + 4 geo + 9 adv)
    training        TrainingLoop, callbacks, N-Stage, metricas
    inference       InferencePipeline, SlidingWindowInference, export
    evaluation      Metricas, comparacao, sensibilidade
    visualization   Holdout plots, training curves, Picasso DOD, EDA
    utils           Logger, timer, validation, formatting, system, io

Example:
    >>> from geosteering_ai import PipelineConfig
    >>> from geosteering_ai.data import DataPipeline
    >>> from geosteering_ai.models import ModelRegistry
    >>> from geosteering_ai.training import TrainingLoop
    >>>
    >>> config = PipelineConfig.from_yaml("configs/robusto.yaml")
    >>> pipeline = DataPipeline(config)
    >>> data = pipeline.prepare("/path/to/dataset")
    >>> model = ModelRegistry().build(config)
    >>> history = TrainingLoop(config, model, pipeline, data).run()
"""

__version__ = "2.0.0"
__author__ = "Daniel Leal"
__framework__ = "TensorFlow 2.x / Keras"

from geosteering_ai.config import PipelineConfig

__all__ = [
    "PipelineConfig",
    "__version__",
    "__author__",
]
