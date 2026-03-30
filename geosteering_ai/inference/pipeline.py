# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: inference/pipeline.py                                             ║
# ║  Bloco: 7 — Inference                                                    ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • InferencePipeline: cadeia completa FV+GS+scalers+modelo               ║
# ║    • Serializa artefatos: model.keras + scalers.joblib + config.yaml      ║
# ║    • Predicoes no dominio original (Ohm.m), nao log10                     ║
# ║    • Suporte a estimativa de incerteza (MC dropout / ensemble)            ║
# ║                                                                            ║
# ║  Dependencias: config.py, data/feature_views.py, data/geosignals.py,     ║
# ║                data/scaling.py                                             ║
# ║  Exports: ~1 classe (InferencePipeline)                                   ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6.1                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C25/C46)         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""InferencePipeline — Cadeia completa de inferencia para inversao 1D.

Encapsula o fluxo: raw 22-col → FV → GS → scale → model.predict → inverse_scale.
Serializavel em disco (model.keras + scalers.joblib + config.yaml) para
implantacao em producao via joblib ou SavedModel.

Cadeia de preprocessamento identica ao treinamento (P6):

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │  CADEIA DE INFERENCIA (P6)                                          │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  raw(22-col) → FV_transform → GS_transform → scale → model.predict │
    │       │                                                   │          │
    │       │                              inverse_target_scaling ← ───┘  │
    │       │                                        │                     │
    │       │                              predicoes em Ohm.m             │
    │                                                                      │
    │  Scalers: identicos ao treinamento (fitados em dados LIMPOS, P3)    │
    │  FV/GS: identicos ao treinamento (mesma feature_view, mesmas GS)    │
    └──────────────────────────────────────────────────────────────────────┘

Referencia: docs/ARCHITECTURE_v2.md secao 6.1.

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Referenciado em:
        - inference/__init__.py: re-exportado como API publica
        - inference/realtime.py: RealtimeInference compoe InferencePipeline
        - data/pipeline.py: PreparedData documenta scaler_em, scaler_gs
        - tests/test_inference.py: TestInferencePipeline (shapes, roundtrip)
    Ref: docs/ARCHITECTURE_v2.md secao 6.1 (InferencePipeline).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos deste modulo
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    "InferencePipeline",
]


# ════════════════════════════════════════════════════════════════════════
# INFERENCE PIPELINE — Cadeia completa de producao
#
# Encapsula modelo Keras + scalers + config para reproducibilidade
# total da inferencia. Serializa em 3 artefatos:
#   1. model.keras    — pesos e arquitetura do modelo treinado
#   2. scalers.joblib — estado do scaler (mean, std, etc.)
#   3. config.yaml    — snapshot da configuracao usada no treinamento
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  ARTEFATOS DE PERSISTENCIA                                          │
# ├──────────────────────────────────────────────────────────────────────┤
# │  pipeline_dir/                                                       │
# │  ├── model.keras        ← tf.keras.Model completo                   │
# │  ├── scalers.joblib     ← {scaler_em, scaler_gs, n_em_features}    │
# │  └── config.yaml        ← PipelineConfig snapshot                   │
# └──────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════


class InferencePipeline:
    """Cadeia completa de inferencia: FV + GS + scalers + model.predict.

    Encapsula o fluxo preprocessamento → modelo → pos-processamento
    para inferencia em producao. Serializavel em disco via save()/load().

    Attributes:
        config: PipelineConfig snapshot do treinamento.
        model: Modelo Keras treinado (tf.keras.Model).
        scaler_params: Dict com estado dos scalers (mean, std, scale, etc.)
            e metadados como n_em_features e expanded_features.

    Example:
        >>> from geosteering_ai import PipelineConfig
        >>> from geosteering_ai.inference import InferencePipeline
        >>>
        >>> config = PipelineConfig.robusto()
        >>> pipeline = InferencePipeline(config, model=trained_model,
        ...     scaler_params={"scaler_em": scaler_em, "scaler_gs": scaler_gs})
        >>> predictions = pipeline.predict(raw_data)
        >>> pipeline.save("/path/to/pipeline_dir")
        >>>
        >>> loaded = InferencePipeline.load("/path/to/pipeline_dir")
        >>> preds2 = loaded.predict(raw_data)

    Note:
        Referenciado em:
            - inference/__init__.py: re-exportado como API publica
            - inference/realtime.py: RealtimeInference.pipeline (composicao)
            - inference/export.py: export_* funcoes extraem self.model
            - data/pipeline.py: PreparedData fornece scaler_em/scaler_gs
        Ref: docs/ARCHITECTURE_v2.md secao 6.1 (InferencePipeline).
        Scalers DEVEM ser os mesmos fitados em dados LIMPOS (P3).
        FV/GS DEVEM ser identicos ao treinamento (P6).
    """

    # ────────────────────────────────────────────────────────────────
    # D2: Inicializacao — armazena config, modelo e scalers
    # ────────────────────────────────────────────────────────────────

    def __init__(
        self,
        config: PipelineConfig,
        model: Any = None,
        scaler_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Inicializa InferencePipeline com config, modelo e scalers.

        Args:
            config: PipelineConfig snapshot do treinamento. Contem todas
                as FLAGS necessarias para reproduzir a cadeia FV+GS+scale.
            model: Modelo Keras treinado (tf.keras.Model). Pode ser None
                se o pipeline sera carregado via load().
            scaler_params: Dict com estado dos scalers e metadados.
                Chaves esperadas:
                    - "scaler_em": Scaler EM fitado em dados limpos
                    - "scaler_gs": Scaler GS fitado em dados limpos (ou None)
                    - "n_em_features": int, numero de features EM base
                    - "expanded_features": list[int], colunas expandidas (GS)

        Raises:
            ValueError: Se config for None.

        Note:
            Referenciado em:
                - InferencePipeline.predict(): usa self.model e self.scaler_params
                - InferencePipeline.save(): serializa todos os atributos
                - InferencePipeline.load(): reconstroi via __init__
            Ref: docs/ARCHITECTURE_v2.md secao 6.1.
        """
        if config is None:
            raise ValueError("config nao pode ser None")

        self.config = config
        self.model = model
        self.scaler_params = scaler_params or {}

        logger.info(
            "InferencePipeline inicializado — model_type=%s, "
            "feature_view=%s, use_geosignal_features=%s",
            config.model_type,
            config.feature_view,
            config.use_geosignal_features,
        )

    # ────────────────────────────────────────────────────────────────
    # D2: Predicao — cadeia completa raw → FV → GS → scale → predict
    # ────────────────────────────────────────────────────────────────

    def predict(
        self,
        raw_data: np.ndarray,
        *,
        theta: Optional[float] = None,
        freq: Optional[float] = None,
        return_uncertainty: bool = False,
        mc_samples: int = 30,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Executa inferencia sobre dados brutos 22-colunas.

        Cadeia: raw → [inject theta/freq] → FV_transform → GS_transform
        → scale → model.predict → inverse_target_scaling.
        Retorna predicoes em Ohm.m (dominio original, nao log10).

        Args:
            raw_data: np.ndarray de shape (n_samples, sequence_length, 22)
                contendo dados brutos no formato 22-colunas padrao.
            theta: Angulo de inclinacao em graus (0-90). Obrigatorio quando
                config.use_theta_as_feature=True. Normalizado internamente
                como theta/90.0. Ref: Perspectiva P2.
            freq: Frequencia EM em Hz (ex: 20000.0). Obrigatorio quando
                config.use_freq_as_feature=True. Normalizado conforme
                config.freq_normalization ("log10", "khz", "raw").
                Ref: Perspectiva P3.
            return_uncertainty: Se True, estima incerteza via MC Dropout
                (multiplas forward passes com dropout ativo) e retorna
                tuple (mean_predictions, std_predictions).
            mc_samples: Numero de amostras MC Dropout para estimativa de
                incerteza. Ignorado se return_uncertainty=False. Default: 30.

        Returns:
            Se return_uncertainty=False:
                np.ndarray de shape (n_samples, sequence_length, n_targets)
                com predicoes em Ohm.m.
            Se return_uncertainty=True:
                Tuple (predictions, uncertainty) onde ambos sao np.ndarray
                de shape (n_samples, sequence_length, n_targets).

        Raises:
            RuntimeError: Se self.model for None (modelo nao carregado).
            ValueError: Se raw_data nao tiver shape esperado (3D com 22 colunas).

        Note:
            Referenciado em:
                - inference/realtime.py: RealtimeInference.update() chama predict()
                - tests/test_inference.py: TestInferencePipeline.test_predict_shape
            Ref: docs/ARCHITECTURE_v2.md secao 6.1.
            inverse_target_scaling converte de log10 para Ohm.m: y = 10^y'.
            MC Dropout: ativa dropout em inference via training=True,
            coleta mc_samples forward passes, retorna media e std.
        """
        import tensorflow as tf

        if self.model is None:
            raise RuntimeError(
                "Modelo nao carregado. Use InferencePipeline.load() ou "
                "passe model= no construtor."
            )

        if raw_data.ndim != 3 or raw_data.shape[-1] != self.config.n_columns:
            raise ValueError(
                f"raw_data deve ter shape (n_samples, seq_len, {self.config.n_columns}), "
                f"recebido: {raw_data.shape}"
            )

        # ── Passo 1: Extrair features de entrada (INPUT_FEATURES) ──
        # Seleciona colunas base do formato 22-col.
        # Default: [1, 4, 5, 20, 21] (z + Hxx + Hzz).
        # Expandido: [1, 4, 5, 6, 7, ..., 20, 21] (z + N componentes EM).
        _feats = self.config.input_features
        x = raw_data[:, :, _feats].astype(np.float32)

        # ── Passo 1B: Injetar theta/freq como prefixo (P2/P3) ─────
        # theta e freq NAO existem no .dat 22-col — sao parametros
        # conhecidos do cabecalho (.out) injetados como colunas constantes.
        # Layout: [theta_norm?, f_norm?, z_obs, EM...]
        # Ordem: freq prepended primeiro, theta prepended segundo
        # (ultimo prepend fica na posicao 0 = theta).
        if self.config.use_freq_as_feature:
            if self.config.freq_normalization == "log10":
                f_val = np.log10(freq) if freq is not None else 4.301
            elif self.config.freq_normalization == "khz":
                f_val = (freq / 1000.0) if freq is not None else 20.0
            else:
                f_val = freq if freq is not None else 20000.0
            n_seq, seq_len, _ = x.shape
            freq_col = np.full((n_seq, seq_len, 1), f_val, dtype=np.float32)
            x = np.concatenate([freq_col, x], axis=-1)

        if self.config.use_theta_as_feature:
            theta_norm = (theta / 90.0) if theta is not None else 0.0
            n_seq, seq_len, _ = x.shape
            theta_col = np.full((n_seq, seq_len, 1), theta_norm, dtype=np.float32)
            x = np.concatenate([theta_col, x], axis=-1)

        # ── Passo 2: Feature View — transforma componentes EM ──
        # h1_cols/h2_cols com offset n_prefix para pular theta/freq.
        # Bug fix v2.0.1: Legado passava config (objeto) em vez de view (str).
        from geosteering_ai.data.feature_views import apply_feature_view

        _offset = self.config.n_prefix
        try:
            _h1_cols = (_feats.index(4) + _offset, _feats.index(5) + _offset)
            _h2_cols = (_feats.index(20) + _offset, _feats.index(21) + _offset)
        except ValueError as exc:
            raise ValueError(
                f"input_features {_feats} deve conter colunas 4,5 (Re/Im Hxx) "
                f"e 20,21 (Re/Im Hzz). Errata v5.0.15: baseline obrigatorio."
            ) from exc
        x = apply_feature_view(
            x,
            self.config.feature_view,
            h1_cols=_h1_cols,
            h2_cols=_h2_cols,
        )

        # ── Passo 3: Geosinais — features derivadas do tensor EM ──
        # Se GS ativos, computa geosinais e concatena com features EM.
        # NOTA: expanded_features e usado SOMENTE para GS (nao para FV).
        # FV opera sobre h1/h2 cols (Hxx, Hzz) que ja estao em x via
        # input_features. GS precisa de componentes off-diagonal (Hxy,
        # Hxz, Hyz, etc.) que NAO estao em input_features baseline.
        if self.config.use_geosignal_features:
            from geosteering_ai.data.geosignals import compute_geosignals

            expanded_features = self.scaler_params.get("expanded_features")
            if expanded_features is not None:
                # Extrair colunas expandidas para computacao de GS
                x_expanded = raw_data[:, :, expanded_features].astype(np.float32)
            else:
                x_expanded = x

            gs_channels = compute_geosignals(x_expanded, self.config)
            x = np.concatenate([x, gs_channels], axis=-1)

        # ── Passo 4: Scaling — normaliza features com scaler treinado ──
        # Usa o mesmo scaler fitado em dados LIMPOS durante treinamento (P3).
        scaler_em = self.scaler_params.get("scaler_em")
        scaler_gs = self.scaler_params.get("scaler_gs")
        n_em = self.scaler_params.get("n_em_features", self.config.n_base_features)

        if scaler_em is not None:
            original_shape = x.shape
            # Scaler opera em 2D: (n_samples * seq_len, n_features)
            x_2d = x.reshape(-1, x.shape[-1])

            if scaler_gs is not None and x.shape[-1] > n_em:
                # Scaling separado EM vs GS (per-group, P3)
                x_em = scaler_em.transform(x_2d[:, :n_em])
                x_gs = scaler_gs.transform(x_2d[:, n_em:])
                x_2d = np.concatenate([x_em, x_gs], axis=-1)
            else:
                # Scaling unico para todas as features
                x_2d = scaler_em.transform(x_2d)

            x = x_2d.reshape(original_shape)

        # ── Passo 5: Predicao do modelo Keras ──
        if return_uncertainty:
            # MC Dropout: multiplas forward passes com dropout ativo
            predictions_list = []
            for _ in range(mc_samples):
                # training=True ativa dropout layers durante inferencia
                pred = self.model(tf.constant(x, dtype=tf.float32), training=True)
                predictions_list.append(pred.numpy())

            predictions_stack = np.stack(predictions_list, axis=0)
            y_mean = np.mean(predictions_stack, axis=0)
            y_std = np.std(predictions_stack, axis=0)

            # ── Passo 6: Inverse target scaling → Ohm.m ──
            from geosteering_ai.data.scaling import inverse_target_scaling

            y_mean_ohm = inverse_target_scaling(y_mean, method=self.config.target_scaling)
            # std em log10 decades — NAO aplicar inverse scaling
            # (10^std e fisicamente errado para desvio-padrao).
            # y_std ja esta em unidades interpretaveis: 0.1 = ~0.1 decada log10.
            y_std_ohm = y_std

            logger.info(
                "Inferencia MC Dropout concluida — %d amostras, "
                "%d forward passes, shape=%s",
                x.shape[0],
                mc_samples,
                y_mean_ohm.shape,
            )
            return y_mean_ohm, y_std_ohm

        # Forward pass unico (sem incerteza)
        y_pred = self.model.predict(x, verbose=0)

        # ── Passo 6: Inverse target scaling → Ohm.m ──
        # Converte de log10 para Ohm.m: y = 10^y' (para target_scaling="log10")
        from geosteering_ai.data.scaling import inverse_target_scaling

        y_pred_ohm = inverse_target_scaling(y_pred, method=self.config.target_scaling)

        logger.info(
            "Inferencia concluida — %d amostras, shape=%s",
            x.shape[0],
            y_pred_ohm.shape,
        )
        return y_pred_ohm

    # ────────────────────────────────────────────────────────────────
    # D2: Persistencia — save/load do pipeline completo
    # ────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Salva pipeline em diretorio (model.keras + scalers.joblib + config.yaml).

        Cria o diretorio se nao existir. Serializa 3 artefatos:
            1. model.keras    — modelo Keras completo (pesos + arquitetura)
            2. scalers.joblib — estado dos scalers EM e GS
            3. config.yaml    — snapshot da configuracao para reprodutibilidade

        Args:
            path: Caminho do diretorio de saida. Sera criado se necessario.

        Raises:
            RuntimeError: Se self.model for None.

        Note:
            Referenciado em:
                - inference/pipeline.py: InferencePipeline.load() (par save/load)
                - tests/test_inference.py: TestInferencePipeline.test_save_load
            Ref: docs/ARCHITECTURE_v2.md secao 6.1.
            Requer joblib (lazy import — pip install joblib).
            config.yaml usa PipelineConfig.to_yaml() para serializar.
        """
        import joblib

        if self.model is None:
            raise RuntimeError("Modelo nao definido — nada para salvar.")

        if not self.scaler_params:
            logger.warning(
                "scaler_params vazio — scalers.joblib sera salvo vazio. "
                "predict() pode falhar ao carregar este pipeline."
            )

        os.makedirs(path, exist_ok=True)

        # Salvar modelo Keras no formato nativo .keras
        model_path = os.path.join(path, "model.keras")
        self.model.save(model_path)
        logger.info("Modelo salvo em %s", model_path)

        # Salvar scalers e metadados via joblib
        scalers_path = os.path.join(path, "scalers.joblib")
        joblib.dump(self.scaler_params, scalers_path)
        logger.info("Scalers salvos em %s", scalers_path)

        # Salvar config como YAML para reprodutibilidade
        config_path = os.path.join(path, "config.yaml")
        self.config.to_yaml(config_path)
        logger.info("Config salvo em %s", config_path)

        logger.info(
            "InferencePipeline salvo em %s — 3 artefatos "
            "(model.keras, scalers.joblib, config.yaml)",
            path,
        )

    @classmethod
    def load(cls, path: str) -> "InferencePipeline":
        """Carrega pipeline de diretorio previamente salvo.

        Reconstroi InferencePipeline a partir dos 3 artefatos serializados
        (model.keras + scalers.joblib + config.yaml).

        Args:
            path: Caminho do diretorio contendo os artefatos.

        Returns:
            InferencePipeline reconstruido com modelo, scalers e config.

        Raises:
            FileNotFoundError: Se algum artefato estiver ausente.
            ImportError: Se joblib ou pyyaml nao estiverem instalados.

        Note:
            Referenciado em:
                - inference/pipeline.py: InferencePipeline.save() (par save/load)
                - tests/test_inference.py: TestInferencePipeline.test_save_load
            Ref: docs/ARCHITECTURE_v2.md secao 6.1.
            Modelo carregado via tf.keras.models.load_model().
            Config reconstruido via PipelineConfig.from_yaml().
            Validacao fail-fast no PipelineConfig.__post_init__().
        """
        import joblib
        import tensorflow as tf

        from geosteering_ai.config import PipelineConfig

        # Carregar config YAML → PipelineConfig (validacao fail-fast)
        config_path = os.path.join(path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.yaml nao encontrado em {path}")
        config = PipelineConfig.from_yaml(config_path)

        # Carregar modelo Keras
        model_path = os.path.join(path, "model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.keras nao encontrado em {path}")
        model = tf.keras.models.load_model(model_path)

        # Carregar scalers via joblib
        scalers_path = os.path.join(path, "scalers.joblib")
        if not os.path.exists(scalers_path):
            raise FileNotFoundError(f"scalers.joblib nao encontrado em {path}")
        scaler_params = joblib.load(scalers_path)

        # ── Validar compatibilidade shape config ↔ modelo (warning) ──
        # Dual-input models tem tuple de tuples (2 inputs) — so validar
        # single-input models onde input_shape e uma tupla simples.
        _inp = getattr(model, "input_shape", None)
        if isinstance(_inp, tuple) and not isinstance(_inp[0], tuple):
            actual_feat = _inp[-1]
            if actual_feat is not None and actual_feat != config.n_features:
                logger.warning(
                    "Model input_shape[-1]=%d != config.n_features=%d. "
                    "Verifique compatibilidade config <-> modelo.",
                    actual_feat,
                    config.n_features,
                )

        logger.info(
            "InferencePipeline carregado de %s — model_type=%s",
            path,
            config.model_type,
        )
        return cls(config=config, model=model, scaler_params=scaler_params)

    # ────────────────────────────────────────────────────────────────
    # D2: Representacao — info concisa para logging/debug
    # ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Representacao concisa do pipeline para logging.

        Note:
            Exibe model_type, feature_view e status do modelo.
        """
        model_status = "loaded" if self.model is not None else "not loaded"
        return (
            f"InferencePipeline("
            f"model_type={self.config.model_type!r}, "
            f"feature_view={self.config.feature_view!r}, "
            f"gs={self.config.use_geosignal_features}, "
            f"model={model_status})"
        )
