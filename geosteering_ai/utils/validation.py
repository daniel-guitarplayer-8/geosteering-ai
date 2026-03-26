# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: utils/validation.py                                               ║
# ║  Bloco: 5 — Utilitarios                                                   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • ValidationTracker para verificacoes formais com contagem [VN]        ║
# ║    • validate_shape() para checagem de shapes de tensors                  ║
# ║    • Formatacao consistente de resultados de validacao                     ║
# ║                                                                            ║
# ║  Dependencias: logging (stdlib), typing (stdlib)                           ║
# ║  Exports: ~2 simbolos — ver __all__                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9 (utils/)                           ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: VALIDATION TRACKER
# ════════════════════════════════════════════════════════════════════════════
# Tracker formalizado para verificacoes de integridade do pipeline.
# Cada check recebe indice sequencial [VN], status (pass/fail),
# e descricao. Ao final, finalize() reporta totais e opcionalmente
# levanta excecao se houver falhas.
# Adaptado do legado C0 ValidationTracker para v2.0.
# ──────────────────────────────────────────────────────────────────────────


class ValidationTracker:
    """Tracker de validacoes com contagem [VN] e relatorio final.

    Registra verificacoes sequenciais com status pass/fail. Cada check
    eh logado com indice ``[V{N}]`` e simbolo visual. Ao chamar
    finalize(), reporta totais e opcionalmente levanta RuntimeError
    se houver falhas.

    Usado extensivamente em todos os modulos para garantir integridade
    do pipeline: shapes de dados, valores fisicos, compatibilidade
    de configuracoes.

    Attributes:
        module_name (str): Nome do modulo sendo validado (para log).
        passed (int): Contagem de checks aprovados.
        failed (int): Contagem de checks reprovados.
        checks (list[dict]): Historico de todos os checks realizados.

    Example:
        >>> from geosteering_ai.utils.validation import ValidationTracker
        >>> vt = ValidationTracker("data/loading")
        >>> vt.check(data.shape[1] == 22, "Dataset tem 22 colunas")
        >>> vt.check(len(models) > 0, "Ao menos 1 modelo geologico")
        >>> vt.finalize(raise_on_error=True)

    Note:
        Referenciado em: TODOS os modulos de geosteering_ai/.
        Ref: Padrao legado C0-C47 (ValidationTracker), adaptado para
        v2.0 com logging ao inves de print.
    """

    def __init__(
        self,
        module_name: str,
        log: Optional[logging.Logger] = None,
    ) -> None:
        """Inicializa tracker para um modulo.

        Args:
            module_name: Nome descritivo do modulo (e.g., "data/loading").
            log: Logger opcional. Se None, usa logger deste modulo.
        """
        self.module_name = module_name
        self.passed = 0
        self.failed = 0
        self.checks: list[dict[str, Any]] = []
        self._log = log or logger
        self._counter = 0

    def check(self, condition: bool, description: str) -> bool:
        """Registra uma verificacao e loga resultado.

        Incrementa contador sequencial, registra no historico, e loga
        com simbolo visual e indice [VN].

        Args:
            condition: Resultado da verificacao (True = pass).
            description: Descricao curta da verificacao.

        Returns:
            bool: O proprio condition (para encadeamento).
        """
        self._counter += 1
        idx = self._counter

        if condition:
            self.passed += 1
            self._log.info("[V%d] %s", idx, description)
        else:
            self.failed += 1
            self._log.warning("[V%d] %s — FALHOU", idx, description)

        self.checks.append({
            "index": idx,
            "description": description,
            "passed": condition,
        })

        return condition

    def finalize(self, raise_on_error: bool = False) -> bool:
        """Reporta totais e opcionalmente levanta excecao.

        Loga resumo com contagens de pass/fail e nome do modulo.
        Se raise_on_error=True e houver falhas, levanta RuntimeError
        com detalhes das falhas.

        Args:
            raise_on_error: Se True, levanta RuntimeError em caso de
                falhas. Default: False (apenas loga).

        Returns:
            bool: True se todas as verificacoes passaram.

        Raises:
            RuntimeError: Se raise_on_error=True e failed > 0.
        """
        total = self.passed + self.failed
        all_passed = self.failed == 0

        if all_passed:
            self._log.info(
                "[%s] Validacao concluida: %d/%d checks aprovados",
                self.module_name, self.passed, total,
            )
        else:
            failed_checks = [
                c for c in self.checks if not c["passed"]
            ]
            failed_desc = "; ".join(
                f"V{c['index']}: {c['description']}"
                for c in failed_checks
            )
            self._log.error(
                "[%s] Validacao FALHOU: %d/%d aprovados, %d falhas: %s",
                self.module_name, self.passed, total,
                self.failed, failed_desc,
            )

            if raise_on_error:
                raise RuntimeError(
                    f"[{self.module_name}] {self.failed} validacao(oes) "
                    f"falharam: {failed_desc}"
                )

        return all_passed


# ════════════════════════════════════════════════════════════════════════════
# SECAO: SHAPE VALIDATION
# ════════════════════════════════════════════════════════════════════════════
# Validacao de shapes de tensors/arrays com suporte a wildcards.
# None em expected_shape = wildcard (aceita qualquer valor).
# Suporta numpy arrays, TensorFlow tensors, e tuplas/listas.
# ──────────────────────────────────────────────────────────────────────────


def validate_shape(
    tensor: Any,
    expected: Tuple[Optional[int], ...],
    name: str = "tensor",
) -> bool:
    """Valida shape de tensor/array contra formato esperado.

    Compara cada dimensao do tensor com o valor esperado. None no
    expected funciona como wildcard (aceita qualquer valor). Suporta
    numpy arrays, TensorFlow tensors, e objetos com atributo .shape.

    Args:
        tensor: Tensor ou array a ser validado. Deve ter atributo .shape.
        expected: Tupla com shape esperada. None = wildcard.
            Exemplo: (None, 600, 5) aceita qualquer batch size.
        name: Nome descritivo para mensagem de erro.

    Returns:
        bool: True se shape eh compativel.

    Raises:
        ValueError: Se ndim nao coincide ou dimensao fixa nao bate.

    Example:
        >>> import numpy as np
        >>> from geosteering_ai.utils.validation import validate_shape
        >>> data = np.zeros((32, 600, 5))
        >>> validate_shape(data, (None, 600, 5), "x_train")
        True

    Note:
        Referenciado em:
            - data/pipeline.py (validacao de shapes pos-prepare)
            - models/registry.py (validacao de output shape)
            - training/loop.py (sanity check pre-fit)
        Wildcard None: permite batch size variavel.
    """
    shape = tuple(tensor.shape)

    # ── Verificar numero de dimensoes ──────────────────────────────────
    if len(shape) != len(expected):
        raise ValueError(
            f"{name}: ndim={len(shape)} (shape={shape}), "
            f"esperado ndim={len(expected)} (shape={expected})"
        )

    # ── Verificar cada dimensao (None = wildcard) ─────────────────────
    # None em expected = wildcard (aceita qualquer valor).
    # None em actual = dimensao dinamica TF (aceitar silenciosamente).
    for i, (actual, exp) in enumerate(zip(shape, expected)):
        if exp is not None and actual is not None and actual != exp:
            raise ValueError(
                f"{name}: dim[{i}]={actual}, esperado {exp} "
                f"(shape={shape}, esperado={expected})"
            )

    return True


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente para facilitar navegacao.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Tracker ───────────────────────────────────────────────────────
    "ValidationTracker",
    # ── Shape ─────────────────────────────────────────────────────────
    "validate_shape",
]
