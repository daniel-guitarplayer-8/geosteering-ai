# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: inference/export.py                                               ║
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
# ║    • Exporta modelos Keras em 3 formatos de implantacao:                  ║
# ║      1. TF SavedModel — formato padrao TensorFlow Serving                 ║
# ║      2. TFLite — otimizado para edge/mobile (quantizacao opt-in)          ║
# ║      3. ONNX — interoperabilidade multi-framework (requer tf2onnx)       ║
# ║    • Todas as funcoes recebem config: PipelineConfig                       ║
# ║    • Input signature com shape explicita para reproducibilidade           ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), TensorFlow 2.x                ║
# ║  Exports: ~3 funcoes (export_saved_model, export_tflite, export_onnx)    ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6.3                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (3 formatos)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""Export — Exportacao de modelos Keras para formatos de implantacao.

Suporta 3 formatos de exportacao para diferentes cenarios de deploy:

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │  FORMATOS DE EXPORTACAO                                             │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  SavedModel (.pb)                                                    │
    │    • Formato padrao TF Serving / TF Hub                             │
    │    • Preserva grafo completo + pesos + assinaturas                  │
    │    • Ideal: server-side, Docker, cloud endpoints                    │
    │                                                                      │
    │  TFLite (.tflite)                                                    │
    │    • Formato otimizado para edge/mobile/IoT                         │
    │    • Quantizacao opcional (float16 ou int8)                         │
    │    • Ideal: dispositivos embarcados, inferencia on-rig              │
    │                                                                      │
    │  ONNX (.onnx)                                                        │
    │    • Formato aberto multi-framework                                  │
    │    • Requer tf2onnx (pip install tf2onnx)                           │
    │    • Ideal: interop com PyTorch, ONNX Runtime, Azure ML            │
    └──────────────────────────────────────────────────────────────────────┘

Referencia: docs/ARCHITECTURE_v2.md secao 6.3.

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Referenciado em:
        - inference/__init__.py: re-exportado como API publica
        - tests/test_inference.py: TestExport (shapes, formatos)
    Ref: docs/ARCHITECTURE_v2.md secao 6.3 (Export).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos deste modulo
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    "export_saved_model",
    "export_tflite",
    "export_onnx",
]


# ════════════════════════════════════════════════════════════════════════
# SAVED MODEL — Formato padrao TF Serving
#
# Salva o modelo completo (grafo + pesos + assinaturas de servico)
# no formato SavedModel. Input signature explicita garante que o
# modelo aceita tensores com shape correta na implantacao.
#
# Input shape: (batch, SEQUENCE_LENGTH, n_features)
#   - SEQUENCE_LENGTH = 600 (Errata v4.4.5 — NUNCA 601)
#   - n_features = 5 (baseline) ou 9 (P4 usd_uhr: 5 + 4 GS)
# ════════════════════════════════════════════════════════════════════════

def export_saved_model(
    model: Any,
    path: str,
    config: PipelineConfig,
) -> str:
    """Exporta modelo Keras como TF SavedModel.

    Formato padrao para TensorFlow Serving, TF Hub e cloud endpoints.
    Preserva grafo computacional completo, pesos treinados e assinaturas
    de servico com shapes explicitas.

    Args:
        model: tf.keras.Model treinado para exportar.
        path: Caminho do diretorio de saida. Sera criado se necessario.
            Ex: "/path/to/export/saved_model"
        config: PipelineConfig com parametros de shape (sequence_length,
            n_features) para definir a input signature.

    Returns:
        Caminho absoluto do diretorio SavedModel criado.

    Raises:
        ValueError: Se model for None.
        OSError: Se nao for possivel criar o diretorio.

    Note:
        Referenciado em:
            - inference/__init__.py: re-exportado como API publica
            - tests/test_inference.py: TestExport.test_export_saved_model
        Ref: docs/ARCHITECTURE_v2.md secao 6.3.
        Input signature: (None, sequence_length, n_features) onde
        sequence_length=600 e n_features depende de feature_view + GS.
    """
    import tensorflow as tf

    if model is None:
        raise ValueError("model nao pode ser None")

    os.makedirs(path, exist_ok=True)

    # Definir input signature com shapes explicitas do config
    # batch_size=None permite batches de tamanho variavel
    input_signature = [
        tf.TensorSpec(
            shape=(None, config.sequence_length, config.n_features),
            dtype=tf.float32,
            name="input_features",
        )
    ]

    # Salvar com assinatura de servico explicita
    @tf.function(input_signature=input_signature)
    def serve(x: tf.Tensor) -> tf.Tensor:
        """Funcao de servico com shape fixa para TF Serving."""
        return model(x, training=False)

    tf.saved_model.save(
        model,
        path,
        signatures={"serving_default": serve},
    )

    logger.info(
        "SavedModel exportado em %s — input_shape=(None, %d, %d)",
        path, config.sequence_length, config.n_features,
    )
    return os.path.abspath(path)


# ════════════════════════════════════════════════════════════════════════
# TFLITE — Formato otimizado para edge/mobile
#
# Converte modelo Keras para TFLite via TFLiteConverter.
# Quantizacao opcional:
#   - float16: reduz tamanho ~50%, preserva precisao
#   - dynamic range (default quantize=True): int8 pesos, float ativacoes
#
# Util para inferencia em dispositivos embarcados na plataforma de
# perfuracao (on-rig), onde recursos computacionais sao limitados.
# ════════════════════════════════════════════════════════════════════════

def export_tflite(
    model: Any,
    path: str,
    config: PipelineConfig,
    *,
    quantize: bool = False,
) -> str:
    """Exporta modelo Keras como TFLite para edge deployment.

    Formato otimizado para dispositivos embarcados, mobile e IoT.
    Quantizacao opcional reduz tamanho do modelo e acelera inferencia
    em hardware sem GPU dedicada.

    Args:
        model: tf.keras.Model treinado para exportar.
        path: Caminho do arquivo .tflite de saida.
            Ex: "/path/to/export/model.tflite"
        config: PipelineConfig com parametros de shape.
        quantize: Se True, aplica quantizacao dynamic range (int8 pesos,
            float32 ativacoes). Reduz tamanho ~4x com perda minima de
            precisao. Default: False (float32 puro).

    Returns:
        Caminho absoluto do arquivo .tflite criado.

    Raises:
        ValueError: Se model for None.
        OSError: Se nao for possivel escrever o arquivo.

    Note:
        Referenciado em:
            - inference/__init__.py: re-exportado como API publica
            - tests/test_inference.py: TestExport.test_export_tflite
        Ref: docs/ARCHITECTURE_v2.md secao 6.3.
        Quantizacao dynamic range: pesos int8, ativacoes float32.
        Para quantizacao full int8, seria necessario dataset representativo
        (nao implementado nesta versao — futuro via representative_dataset).
    """
    import tensorflow as tf

    if model is None:
        raise ValueError("model nao pode ser None")

    # Converter Keras model → TFLite via converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        # Dynamic range quantization: pesos int8, ativacoes float32
        # Reduz tamanho ~4x com perda minima de precisao
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        logger.info(
            "Quantizacao dynamic range ativada — pesos int8, ativacoes float32"
        )

    tflite_model = converter.convert()

    # Garantir diretorio de saida existe
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "wb") as f:
        f.write(tflite_model)

    # Tamanho do arquivo para logging
    size_mb = os.path.getsize(path) / (1024 * 1024)

    logger.info(
        "TFLite exportado em %s — %.2f MB, quantize=%s",
        path, size_mb, quantize,
    )
    return os.path.abspath(path)


# ════════════════════════════════════════════════════════════════════════
# ONNX — Formato aberto multi-framework
#
# Converte modelo Keras para ONNX via tf2onnx (lazy import).
# Requer: pip install tf2onnx
#
# ONNX permite:
#   - Inferencia via ONNX Runtime (CPU/GPU otimizado)
#   - Interoperabilidade com outras ferramentas (Azure ML, etc.)
#   - Visualizacao do grafo via Netron
#
# Nota: ONNX e usado apenas para EXPORTACAO. O treinamento e a
# inferencia principal permanecem exclusivamente em TensorFlow/Keras.
# ════════════════════════════════════════════════════════════════════════

def export_onnx(
    model: Any,
    path: str,
    config: PipelineConfig,
) -> str:
    """Exporta modelo Keras como ONNX para interoperabilidade.

    Formato aberto que permite inferencia via ONNX Runtime e
    interoperabilidade com outras ferramentas de ML. Requer tf2onnx.

    Args:
        model: tf.keras.Model treinado para exportar.
        path: Caminho do arquivo .onnx de saida.
            Ex: "/path/to/export/model.onnx"
        config: PipelineConfig com parametros de shape.

    Returns:
        Caminho absoluto do arquivo .onnx criado.

    Raises:
        ValueError: Se model for None.
        ImportError: Se tf2onnx nao estiver instalado.
        OSError: Se nao for possivel escrever o arquivo.

    Note:
        Referenciado em:
            - inference/__init__.py: re-exportado como API publica
            - tests/test_inference.py: TestExport.test_export_onnx
        Ref: docs/ARCHITECTURE_v2.md secao 6.3.
        Requer: pip install tf2onnx (lazy import).
        ONNX opset_version=15 para compatibilidade ampla.
        ONNX e usado APENAS para exportacao — treinamento e inferencia
        principal permanecem exclusivamente em TensorFlow/Keras.
    """
    import tensorflow as tf

    if model is None:
        raise ValueError("model nao pode ser None")

    try:
        import tf2onnx
    except ImportError:
        raise ImportError(
            "tf2onnx necessario para exportacao ONNX: pip install tf2onnx"
        )

    # Garantir diretorio de saida existe
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Definir input signature para conversao
    input_signature = [
        tf.TensorSpec(
            shape=(None, config.sequence_length, config.n_features),
            dtype=tf.float32,
            name="input_features",
        )
    ]

    # Converter Keras model → ONNX via tf2onnx
    # opset_version=15 garante compatibilidade com ONNX Runtime >= 1.10
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=15,
        output_path=path,
    )

    logger.info(
        "ONNX exportado em %s — opset=15, input_shape=(None, %d, %d)",
        path, config.sequence_length, config.n_features,
    )
    return os.path.abspath(path)
