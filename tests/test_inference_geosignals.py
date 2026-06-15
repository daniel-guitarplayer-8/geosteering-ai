# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_inference_geosignals.py                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Inferência — caminho de geosinais (P4)                     ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-15                                                 ║
# ║  Status      : Produção (gate de regressão — bug compute_geosignals)      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Gate de regressão do bug de geosinais em ``InferencePipeline.predict``.

Bug corrigido: ``inference/pipeline.py`` chamava
``compute_geosignals(x_expanded, self.config)`` — passando um ``PipelineConfig``
onde se espera ``families: List[str]`` E um tensor 3D reindexado (``x_expanded``)
onde ``compute_geosignals`` exige o tensor 2D 22-col completo (indexa componentes
por COLUNA ABSOLUTA via ``EM_COMPONENTS``). O caminho crashava/produzia lixo.

Fix: espelha ``data/pipeline.py`` (caminho offline validado) — ``raw_data`` 22-col
completo (2D) + ``config.resolve_families()`` + ``config.n_columns``. Estes testes
garantem (1) que ``predict`` COMPLETA com GS ativos (antes crashava) e (2) que os
geosinais da inferência são BIT-equivalentes aos do caminho offline (paridade).
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from geosteering_ai.config import PipelineConfig  # noqa: E402
from geosteering_ai.data.geosignals import compute_geosignals  # noqa: E402
from geosteering_ai.inference.pipeline import InferencePipeline  # noqa: E402


class _CaptureModel(tf.keras.Model):
    """Modelo tiny que CAPTURA a entrada (p/ inspecionar os canais GS)."""

    def __init__(self, n_targets: int = 2) -> None:
        super().__init__()
        self.dense = tf.keras.layers.Dense(n_targets)
        self.captured: np.ndarray | None = None

    def call(self, inputs, training=False):  # noqa: ANN001, D102
        self.captured = np.asarray(inputs)
        return self.dense(inputs)


def _make(seed: int = 0):
    cfg = PipelineConfig.geosinais_p4()  # use_geosignal_features=True, set "usd_uhr"
    n_total = cfg.n_prefix + cfg.n_base_features + cfg.n_geosignal_channels
    rng = np.random.default_rng(seed)
    raw = rng.uniform(1.0, 5.0, (2, 30, 22)).astype(np.float64)
    return cfg, n_total, raw


def test_predict_with_geosignals_runs():
    """``predict`` COMPLETA com geosinais ativos (regressão: antes crashava)."""
    cfg, n_total, raw = _make()
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(None, n_total)), tf.keras.layers.Dense(2)]
    )
    pipe = InferencePipeline(cfg, model=model, scaler_params={})
    preds = pipe.predict(raw)  # antes: TypeError/garbage no passo GS
    arr = preds[0] if isinstance(preds, tuple) else preds
    assert np.asarray(arr).shape[0] == 2  # n_samples preservado


def test_inference_geosignals_match_offline():
    """Os geosinais da inferência == caminho offline (paridade bit-equivalente).

    Sem scaler (scaler_params={}), o input do modelo é [FV(features)... GS], então
    os últimos ``n_gs`` canais capturados são exatamente os GS computados pelo fix.
    """
    cfg, n_total, raw = _make()
    families = cfg.resolve_families()
    n_gs = cfg.n_geosignal_channels  # 2 famílias × 2 = 4

    # Referência: caminho offline (data/pipeline.py usa exatamente esta chamada).
    gs_offline = compute_geosignals(
        raw.reshape(-1, 22), families, cfg.n_columns
    ).reshape(2, 30, n_gs)

    model = _CaptureModel(n_targets=2)
    # run_eagerly=True força model.predict a executar call() em modo eager → a
    # captura via np.asarray funciona (em graph o tensor é simbólico).
    model.compile(run_eagerly=True)
    pipe = InferencePipeline(cfg, model=model, scaler_params={})
    pipe.predict(raw)

    assert model.captured is not None, "modelo não foi chamado"
    gs_inference = model.captured[..., -n_gs:]  # últimos n_gs canais = GS do fix
    # float32 (inferência) vs float64 (offline) → tolerância de precisão simples.
    assert np.allclose(gs_inference, gs_offline, rtol=1e-5, atol=1e-4), (
        f"GS da inferência divergiu do offline. "
        f"max|Δ|={np.max(np.abs(gs_inference - gs_offline)):.3e}"
    )
