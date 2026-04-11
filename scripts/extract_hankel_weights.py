#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  scripts/extract_hankel_weights.py                                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python (Sprint 1.1 — Extração de Pesos Hankel)   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Data        : 2026-04-11                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: numpy (extração), pytest (validação)                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Extrai os pesos e abscissas dos filtros Hankel digitais utilizados     ║
# ║    pelo simulador Fortran (Fortran_Gerador/filtersv2.f08) e salva os      ║
# ║    dados em arquivos NumPy (.npz) para consumo pelo simulador Python     ║
# ║    otimizado (backends JAX e Numba).                                      ║
# ║                                                                           ║
# ║  FILTROS EXTRAÍDOS                                                        ║
# ║    ┌──────────────────────┬──────┬────────────────────────────────────┐  ║
# ║    │  Filtro              │ npt  │ Uso (filter_type Fortran)          │  ║
# ║    ├──────────────────────┼──────┼────────────────────────────────────┤  ║
# ║    │  Werthmüller (★)     │ 201  │ filter_type = 0 (default)          │  ║
# ║    │  Kong                │  61  │ filter_type = 1 (rápido, 3.3×)    │  ║
# ║    │  Anderson            │ 801  │ filter_type = 2 (máxima precisão) │  ║
# ║    └──────────────────────┴──────┴────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  FORMATO DE SAÍDA (.npz)                                                  ║
# ║    • abscissas  : array(npt,) float64 — pontos de quadratura kr          ║
# ║    • weights_j0 : array(npt,) float64 — pesos para integral com J₀      ║
# ║    • weights_j1 : array(npt,) float64 — pesos para integral com J₁      ║
# ║    • metadata   : dict com 'filter_name', 'npt', 'source_hash',         ║
# ║                   'source_file', 'extracted_at', 'fortran_subroutine'   ║
# ║                                                                           ║
# ║  CONTRATO MATEMÁTICO                                                      ║
# ║    Aproximação de integral de Hankel de ordem ν:                         ║
# ║                                                                           ║
# ║        ∫₀^∞ f(kr) · Jν(kr·r) · dkr ≈ (1/r) · Σᵢ f(aᵢ/r) · wᵢ(ν)        ║
# ║                                                                           ║
# ║    onde:                                                                  ║
# ║      aᵢ = abscissas[i]         (pontos logaritmicamente espaçados)       ║
# ║      wᵢ = weights_jν[i]        (pesos pré-computados do filtro digital) ║
# ║      r  = afastamento radial   (distância transmissor-receptor)          ║
# ║                                                                           ║
# ║  USO                                                                      ║
# ║    $ python scripts/extract_hankel_weights.py                            ║
# ║    $ python scripts/extract_hankel_weights.py --verify                   ║
# ║    $ python scripts/extract_hankel_weights.py --source PATH/filtersv2.f08║
# ║                                                                           ║
# ║  REFERÊNCIAS BIBLIOGRÁFICAS                                              ║
# ║    • Kong, F. N. (2007). "Hankel transform filters for dipole antenna    ║
# ║      radiation in a conductive medium." Geophysical Prospecting, 55(1).  ║
# ║    • Werthmüller, D. (2017). "An open-source full 3D electromagnetic    ║
# ║      modeler for 1D VTI media in Python: empymod." Geophysics, 82(6).   ║
# ║    • Anderson, W. L. (1989). "A hybrid fast Hankel transform algorithm   ║
# ║      for electromagnetic modeling." Geophysics, 54(2), 263-266.          ║
# ║                                                                           ║
# ║  NOTAS DE IMPLEMENTAÇÃO                                                  ║
# ║    1. O parser lê diretamente filtersv2.f08 via regex determinístico.   ║
# ║    2. A notação Fortran "D+XX" é convertida para "E+XX" (Python).       ║
# ║    3. Valida len(array) == npt esperado antes de salvar.                ║
# ║    4. Registra SHA-256 do arquivo fonte no metadata para auditoria.      ║
# ║    5. Idempotente: rodar N vezes produz artefatos bit-idênticos.        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Extrai pesos Hankel do simulador Fortran para arquivos NumPy .npz.

Este módulo é um utilitário de build executado uma única vez (ou quando
`filtersv2.f08` for atualizado). Os artefatos .npz gerados são versionados
no repositório em `geosteering_ai/simulation/filters/` e carregados em
tempo de execução pelo `FilterLoader` do simulador Python.

Example:
    Extração completa dos 3 filtros::

        $ python scripts/extract_hankel_weights.py
        [INFO] Parsing Fortran_Gerador/filtersv2.f08 (5559 linhas, SHA=a1b2c3...)
        [INFO] Extraindo Werthmüller 201pt... OK (npt=201, norma J0=2.94e+00)
        [INFO] Extraindo Kong 61pt... OK (npt=61, norma J0=1.45e+02)
        [INFO] Extraindo Anderson 801pt... OK (npt=801)
        [INFO] Gravando werthmuller_201pt.npz (9.6 KB)
        [INFO] Gravando kong_61pt.npz (3.0 KB)
        [INFO] Gravando anderson_801pt.npz (38.5 KB)
        [OK] 3/3 filtros extraídos com sucesso.

Note:
    Este script NÃO depende de Fortran — faz parsing puro de texto.
    Mudanças em `filtersv2.f08` DEVEM disparar re-extração manual:
        $ python scripts/extract_hankel_weights.py --verify
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Final

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES DO MÓDULO
# ──────────────────────────────────────────────────────────────────────────────
# Paths relativos ao root do repositório (quando o script é invocado de lá).
# `ROOT_DIR` é computado a partir da localização deste arquivo para suportar
# invocação de qualquer diretório (__file__ → scripts/ → parent → root).
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE: Final[Path] = ROOT_DIR / "Fortran_Gerador" / "filtersv2.f08"
OUTPUT_DIR: Final[Path] = ROOT_DIR / "geosteering_ai" / "simulation" / "filters"

# Identificação semântica dos filtros alvo. A tupla carrega:
#   (nome_canônico, arquivo_saida, nome_subrotina_fortran, npt_esperado)
#
# O mapeamento é ordenado intencionalmente para produzir logs determinísticos
# e facilitar comparações com a documentação do simulador Fortran (F1).
FILTROS_ALVO: Final[tuple[tuple[str, str, str, int], ...]] = (
    ("werthmuller_201pt", "werthmuller_201pt.npz", "J0J1Wer", 201),
    ("kong_61pt",         "kong_61pt.npz",         "J0J1Kong", 61),
    ("anderson_801pt",    "anderson_801pt.npz",    "J0J1And",  801),
)

# Expressões regulares para parsing. Todas usam re.DOTALL para permitir
# que o corpo do array atravesse múltiplas linhas (arrays de centenas de
# valores sempre quebram linhas com continuação "&").
#
# Notação Fortran para reais: 1.234567D+02 (D em vez de E como expoente).
# Notação Python/NumPy:         1.234567E+02
#
# A mantissa aceita quatro formas (em ordem de prevalência em filtersv2.f08):
#   1. `1.234D+02`  (ponto no meio)      — Kong, Werthmüller
#   2. `0.234D-28`  (leading zero)       — Anderson
#   3. `.234D-28`   (dot-prefix)         — não observada, mas Fortran permite
#   4. `1D0`        (inteiro + expoente) — não observada, mas legacy permite
#
# A alternativa `(?:\d+\.?\d*|\.\d+)` cobre todas as formas acima. Observe
# que o expoente `[DdEe][+-]?\d+` é obrigatório: isso evita capturar
# inteiros de índice (ex.: `absc(1:3)` → o `1`, `3` não seriam convertidos).
_RE_FORTRAN_REAL: Final[re.Pattern[str]] = re.compile(
    r"([+-]?(?:\d+\.?\d*|\.\d+))[DdEe]([+-]?\d+)"
)

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING (respeita a regra D9 do padrão de documentação: nunca print)
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("extract_hankel_weights")


def _setup_logging(verbose: bool) -> None:
    """Configura o logger com formato estruturado.

    Args:
        verbose: Se True, usa nível DEBUG; caso contrário, INFO.

    Note:
        Formato padronizado `[NIVEL] mensagem` para facilitar grep em
        logs de CI. Não é necessário timestamp porque o script é rápido
        (< 1 segundo em hardware moderno).
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ROTINAS DE PARSING
# ──────────────────────────────────────────────────────────────────────────────
def _fortran_real_to_python(s: str) -> str:
    """Converte notação Fortran D-exponent para notação Python E-exponent.

    Fortran usa o caractere "D" (ou "d") como separador de mantissa e
    expoente para precisão dupla (real(dp)), enquanto Python/NumPy usa "E"
    (ou "e"). Esta função aplica a substituição preservando sinais.

    Args:
        s: String contendo um ou mais números em notação Fortran.

    Returns:
        String com todos os números convertidos para notação Python.

    Example:
        >>> _fortran_real_to_python("1.234D+02, -5.6D-01")
        '1.234E+02, -5.6E-01'
    """
    return _RE_FORTRAN_REAL.sub(r"\1E\2", s)


def _parse_array_block(
    source: str,
    var_name: str,
    subroutine: str,
    expected_npt: int,
) -> np.ndarray:
    """Extrai um array `absc`, `wJ0` ou `wJ1` de uma subrotina Fortran.

    Suporta dois dialetos de atribuição presentes em `filtersv2.f08`:

      Dialeto A (Kong 61pt):
        absc( 1:3 ) = (/ val1, val2, val3 /)
        absc( 4:6 ) = (/ val4, val5, val6 /)
        ...  (20+ blocos de 3 valores cada)

      Dialeto B (Werthmüller, Anderson):
        absc = (/val1, val2, val3, &
                 val4, val5, val6, &
                 ...  /)

    O parser detecta automaticamente qual dialeto está em uso e normaliza
    a saída em um único vetor NumPy float64 de comprimento `expected_npt`.

    Args:
        source: Conteúdo completo do arquivo Fortran fonte.
        var_name: Nome da variável a extrair ('absc', 'wJ0', 'wJ1',
            ou identificadores Anderson 'an_abs_801', 'an_pes_J0801',
            'an_pes_J1801').
        subroutine: Nome da subrotina Fortran que contém o array
            (para mensagens de erro e localização).
        expected_npt: Comprimento esperado do array (usado para validação
            final; falha se len(array) != expected_npt).

    Returns:
        Array NumPy 1D de float64 com comprimento `expected_npt`.

    Raises:
        ValueError: Se o array não for encontrado, se a conversão de notação
            falhar, ou se o comprimento extraído não bater com `expected_npt`.

    Note:
        A regex para Dialeto A captura APENAS blocos explicitamente
        atribuídos (absc(1:3) = ...), ignorando atribuições que aparecem
        em contextos fora da subrotina alvo. A localização da subrotina
        é resolvida por fatiamento do `source` antes da busca.
    """
    # ── Passo 1: Isolar o bloco da subrotina alvo ────────────────────────
    # Localiza a subrotina pelo nome e fatia o texto até a próxima subrotina
    # ou `end subroutine` imediatamente anterior ao próximo `subroutine`.
    # Isso garante que a regex não cruze fronteiras de subrotinas.
    sub_pattern = re.compile(
        rf"^\s*subroutine\s+{re.escape(subroutine)}\b.*?^\s*end\s+subroutine",
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    sub_match = sub_pattern.search(source)
    if sub_match is None:
        raise ValueError(
            f"Subrotina '{subroutine}' não encontrada em filtersv2.f08. "
            f"Verifique se a assinatura da subrotina não foi renomeada."
        )
    sub_body = sub_match.group(0)

    # Anderson (e outras subrotinas com npt único) não usa elseif/if — o
    # corpo é todo relevante. Para Kong/Werthmüller com múltiplos npt,
    # precisamos restringir ao bloco `if (npt == <expected>)` ou
    # `elseif (npt == <expected>)` correspondente.
    if subroutine in ("J0J1Kong", "J0J1Wer"):
        # ── Passo 1b: isolar o bloco do npt esperado ─────────────────────
        # Padrão: a partir de `(else)if (npt == N) then` até o próximo
        # `(else)if (npt ==` ou `end subroutine`. Evita que blocos de
        # valores para outros npt (ex.: Kong 241) sejam capturados.
        branch_pattern = re.compile(
            rf"(?:if|elseif)\s*\(\s*npt\s*==\s*{expected_npt}\s*\)\s*then"
            rf"(.*?)(?=(?:elseif\s*\(\s*npt\s*==|end\s+subroutine))",
            re.DOTALL | re.IGNORECASE,
        )
        branch_match = branch_pattern.search(sub_body)
        if branch_match is None:
            raise ValueError(
                f"Bloco 'if (npt == {expected_npt})' não encontrado em "
                f"subrotina {subroutine}."
            )
        sub_body = branch_match.group(1)
    # Para Anderson, sub_body é o corpo inteiro (não há elseif).

    # ── Passo 2: extrair os valores numéricos ────────────────────────────
    # Estratégia unificada: capturar QUALQUER atribuição a `var_name`
    # seguida de `(/` ... `/)`. Isso cobre Dialeto A (múltiplos blocos
    # acumulados) e Dialeto B (bloco único multilinha).
    #
    # Regex explicada:
    #   var_name          → literal (absc, wJ0, wJ1, an_abs_801, ...)
    #   (?:\s*\(\s*\d[\d:\s]*\))?  → índice opcional "( 1:3 )" (Dialeto A)
    #   \s*=\s*\(/         → atribuição até "(/"
    #   (.*?)              → corpo do array (lazy)
    #   /\)                → fecha com "/)"
    assign_pattern = re.compile(
        rf"{re.escape(var_name)}\s*(?:\(\s*\d[\d:\s]*\))?\s*=\s*\(/(.*?)/\)",
        re.DOTALL,
    )

    values: list[float] = []
    # Regex de token numérico espelha a mantissa de `_RE_FORTRAN_REAL`
    # (aceita `1.23`, `0.21`, `.21`, `1.`, `1`) mas exige o expoente
    # `[EeDd][+-]?\d+` para NÃO capturar inteiros puros (ex.: `201` em
    # `absc(201)` de um comentário) nem casar números de índice residuais.
    # Observe que após `_fortran_real_to_python` o expoente já está em
    # `E/e`, mas o regex ainda aceita `D/d` por defesa — nunca há custo
    # em ser tolerante a ambas as notações.
    _token_re = re.compile(
        r"[+-]?(?:\d+\.?\d*|\.\d+)[EeDd][+-]?\d+"
    )
    for match in assign_pattern.finditer(sub_body):
        body = match.group(1)
        # Converter D-notation para E-notation
        body_py = _fortran_real_to_python(body)
        # Extrair números usando regex tolerante a vírgulas, espaços, newlines
        # e caracteres de continuação `&`. Cada token é um float válido.
        for tok in _token_re.finditer(body_py):
            values.append(float(tok.group(0)))

    # ── Passo 3: validação de comprimento ────────────────────────────────
    # Se len != expected_npt, há um bug no parser ou no arquivo fonte.
    # Falha explícita — nunca "silenciosa" com valores parciais.
    if len(values) != expected_npt:
        raise ValueError(
            f"Extração de '{var_name}' em {subroutine} produziu "
            f"{len(values)} valores, esperava {expected_npt}. "
            f"Possível regressão em filtersv2.f08 ou no parser."
        )

    return np.array(values, dtype=np.float64)


def _compute_source_hash(source_path: Path) -> str:
    """Computa SHA-256 do arquivo fonte para registro de auditoria.

    Args:
        source_path: Caminho para filtersv2.f08.

    Returns:
        Hash hexadecimal (64 caracteres) do arquivo em modo binário.

    Note:
        O hash é usado como chave de auditoria nos arquivos .npz:
        se o Fortran for atualizado sem re-rodar este script, o hash
        presente no .npz ficará desatualizado e o validador
        `tests/test_simulation_filters.py` levantará um alerta.
    """
    h = hashlib.sha256()
    with source_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# ROTINAS DE EXTRAÇÃO POR FILTRO
# ──────────────────────────────────────────────────────────────────────────────
def _extract_filter(
    name: str,
    subroutine: str,
    expected_npt: int,
    source: str,
) -> dict[str, np.ndarray]:
    """Extrai abscissas e pesos (J₀ e J₁) de um filtro específico.

    Dispatcher que escolhe os nomes de variáveis corretos de acordo
    com a convenção da subrotina Fortran:

      J0J1Kong / J0J1Wer → variáveis 'absc', 'wJ0', 'wJ1'
      J0J1And            → variáveis 'an_abs_801', 'an_pes_J0801', 'an_pes_J1801'

    Args:
        name: Nome canônico do filtro (para logging).
        subroutine: Nome da subrotina Fortran.
        expected_npt: Comprimento esperado dos arrays.
        source: Conteúdo do arquivo filtersv2.f08.

    Returns:
        Dict com chaves 'abscissas', 'weights_j0', 'weights_j1'.

    Raises:
        ValueError: Propagado de `_parse_array_block` se qualquer array
            falhar na extração ou validação.
    """
    # ── Seleção dos nomes de variáveis ──────────────────────────────────
    # Anderson 801pt usa nomes distintos (convenção histórica da subrotina
    # J0J1And, que segue o paper original Anderson 1989). As outras duas
    # subrotinas compartilham a nomenclatura curta absc/wJ0/wJ1.
    if subroutine == "J0J1And":
        var_absc, var_j0, var_j1 = "an_abs_801", "an_pes_J0801", "an_pes_J1801"
    else:
        var_absc, var_j0, var_j1 = "absc", "wJ0", "wJ1"

    logger.debug("Extraindo %s da subrotina %s...", var_absc, subroutine)
    abscissas = _parse_array_block(source, var_absc, subroutine, expected_npt)

    logger.debug("Extraindo %s da subrotina %s...", var_j0, subroutine)
    weights_j0 = _parse_array_block(source, var_j0, subroutine, expected_npt)

    logger.debug("Extraindo %s da subrotina %s...", var_j1, subroutine)
    weights_j1 = _parse_array_block(source, var_j1, subroutine, expected_npt)

    # ── Sanidade adicional: abscissas devem ser estritamente crescentes ─
    # Todos os filtros digitais Hankel usam abscissas logaritmicamente
    # espaçadas (monotônicas). Se não forem, há corrupção no parser.
    if not np.all(np.diff(abscissas) > 0):
        raise ValueError(
            f"Abscissas de {name} não são estritamente crescentes. "
            f"Corrupção no parser ou no arquivo fonte."
        )

    # ── Sanidade adicional: abscissas devem ser estritamente positivas ─
    # Pontos de quadratura Hankel representam kr > 0 (número de onda
    # radial). Valores não positivos quebrariam a transformação.
    if not np.all(abscissas > 0):
        raise ValueError(
            f"Abscissas de {name} contêm valores ≤ 0. "
            f"Filtros Hankel digitais exigem kr > 0."
        )

    logger.info(
        "Extraindo %s... OK (npt=%d, kr∈[%.2e, %.2e], Σ|wJ0|=%.3e, Σ|wJ1|=%.3e)",
        name,
        expected_npt,
        float(abscissas[0]),
        float(abscissas[-1]),
        float(np.sum(np.abs(weights_j0))),
        float(np.sum(np.abs(weights_j1))),
    )

    return {
        "abscissas": abscissas,
        "weights_j0": weights_j0,
        "weights_j1": weights_j1,
    }


def _save_npz(
    data: dict[str, np.ndarray],
    output_path: Path,
    filter_name: str,
    subroutine: str,
    source_path: Path,
    source_hash: str,
) -> None:
    """Grava o artefato .npz com dados + metadata.

    Args:
        data: Dict com 'abscissas', 'weights_j0', 'weights_j1'.
        output_path: Caminho de destino (.npz).
        filter_name: Nome canônico do filtro.
        subroutine: Nome da subrotina Fortran origem.
        source_path: Path absoluto de filtersv2.f08.
        source_hash: SHA-256 hexadecimal do arquivo fonte.

    Note:
        Usa `np.savez_compressed` para reduzir tamanho em disco. Em tempo
        de carregamento, a descompressão é transparente (uma única linha
        no `FilterLoader`).
    """
    # Metadata serializada como array 0-d de string JSON para garantir
    # compatibilidade com o carregamento via `np.load(allow_pickle=False)`.
    # Usar dicts Python requer pickle, o que complica CI e torna o arquivo
    # dependente de versão do Python. `json` já está importado no topo do
    # módulo (PEP 8).
    metadata = {
        "filter_name": filter_name,
        "npt": int(len(data["abscissas"])),
        "source_file": str(source_path.relative_to(ROOT_DIR)),
        "source_sha256": source_hash,
        "fortran_subroutine": subroutine,
        "extracted_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "extractor_script": "scripts/extract_hankel_weights.py",
        "extractor_version": "1.0.0",
    }
    metadata_json = np.array(json.dumps(metadata, ensure_ascii=False))

    np.savez_compressed(
        output_path,
        abscissas=data["abscissas"],
        weights_j0=data["weights_j0"],
        weights_j1=data["weights_j1"],
        metadata=metadata_json,
    )

    size_kb = output_path.stat().st_size / 1024.0
    logger.info("Gravando %s (%.1f KB)", output_path.name, size_kb)


# ──────────────────────────────────────────────────────────────────────────────
# VERIFICAÇÃO (modo --verify)
# ──────────────────────────────────────────────────────────────────────────────
def _verify_outputs(source_hash: str) -> bool:
    """Verifica se os artefatos .npz existem e se o hash do Fortran bate.

    Args:
        source_hash: Hash atual do arquivo filtersv2.f08.

    Returns:
        True se todos os .npz existirem e estiverem sincronizados;
        False caso contrário (indicando que re-extração é necessária).
    """
    all_ok = True
    for name, output_name, subroutine, expected_npt in FILTROS_ALVO:
        path = OUTPUT_DIR / output_name
        if not path.exists():
            logger.warning("[MISSING] %s não existe", path.name)
            all_ok = False
            continue
        with np.load(path, allow_pickle=False) as npz:
            if "metadata" not in npz.files:
                logger.warning(
                    "[STALE] %s sem metadata (versão antiga)", path.name
                )
                all_ok = False
                continue
            meta = json.loads(str(npz["metadata"]))
            if meta.get("source_sha256") != source_hash:
                logger.warning(
                    "[STALE] %s fora de sincronia com filtersv2.f08 "
                    "(hash do npz=%s..., hash atual=%s...)",
                    path.name,
                    meta.get("source_sha256", "?")[:12],
                    source_hash[:12],
                )
                all_ok = False
                continue
            if meta.get("npt") != expected_npt:
                logger.warning(
                    "[CORRUPT] %s com npt=%d, esperado %d",
                    path.name,
                    meta.get("npt"),
                    expected_npt,
                )
                all_ok = False
                continue
            logger.info("[OK] %s (npt=%d, hash OK)", path.name, expected_npt)
    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# ORQUESTRADOR (main)
# ──────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """Orquestrador principal.

    Args:
        argv: Lista de argumentos (None → sys.argv[1:]).

    Returns:
        Código de saída (0=sucesso, 1=falha).
    """
    parser = argparse.ArgumentParser(
        prog="extract_hankel_weights",
        description=(
            "Extrai pesos e abscissas dos filtros Hankel digitais do "
            "simulador Fortran (filtersv2.f08) para arquivos .npz "
            "usados pelo simulador Python."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Caminho para filtersv2.f08 (default: Fortran_Gerador/filtersv2.f08)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Diretório de saída dos .npz (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Apenas verifica se os .npz existem e estão sincronizados.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Habilita logs de nível DEBUG.",
    )
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    # ── Passo 0: verificar existência do arquivo fonte ─────────────────
    if not args.source.exists():
        logger.error("Arquivo fonte não encontrado: %s", args.source)
        return 1

    source_hash = _compute_source_hash(args.source)
    num_lines = sum(1 for _ in args.source.open("r", encoding="utf-8"))
    logger.info(
        "Parsing %s (%d linhas, SHA=%s...)",
        args.source.relative_to(ROOT_DIR) if args.source.is_absolute()
        else args.source,
        num_lines,
        source_hash[:12],
    )

    # ── Modo verify: apenas checa estado dos .npz existentes ────────────
    if args.verify:
        ok = _verify_outputs(source_hash)
        return 0 if ok else 1

    # ── Modo extração: lê source e gera .npz ────────────────────────────
    source = args.source.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_ok = 0
    for name, output_name, subroutine, expected_npt in FILTROS_ALVO:
        try:
            data = _extract_filter(name, subroutine, expected_npt, source)
            _save_npz(
                data=data,
                output_path=args.output_dir / output_name,
                filter_name=name,
                subroutine=subroutine,
                source_path=args.source,
                source_hash=source_hash,
            )
            num_ok += 1
        except ValueError as exc:
            logger.error("Falha ao extrair %s: %s", name, exc)

    total = len(FILTROS_ALVO)
    if num_ok == total:
        logger.info("[OK] %d/%d filtros extraídos com sucesso.", num_ok, total)
        return 0
    logger.error("[FALHA] %d/%d filtros extraídos.", num_ok, total)
    return 1


if __name__ == "__main__":
    sys.exit(main())
