# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/loading.py                                                   ║
# ║  Bloco: 2 — Preparacao de Dados                                           ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Parsing de metadados .out (angulos, modelos, frequencias)            ║
# ║    • Carregamento binario .dat (22-col Fortran, 172 bytes/registro)       ║
# ║    • Decoupling EM (remocao acoplamento direto Tx-Rx: ACp, ACx)           ║
# ║    • Segregacao por angulo em AngleGroup (layout MODEL-MAJOR)             ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig)                                 ║
# ║  Exports: ~8 (OutMetadata, AngleGroup, COL_MAP_22, EM_COMPONENTS,        ║
# ║           parse_out_metadata, load_binary_dat, apply_decoupling,          ║
# ║           segregate_by_angle, load_dataset)                               ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 4.1-4.2                              ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C19-C21)         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""Carregamento de dados brutos do simulador Fortran PerfilaAnisoOmp.

Funcoes para parsing de metadados (.out) e dados binarios (.dat) no
formato 22-colunas. Segregacao MODEL-MAJOR para multi-angulo/multi-freq.

Pipeline de carregamento:
    parse_out_metadata(.out)  →  metadados (angulos, freqs, n_models)
    load_binary_dat(.dat)     →  array (total_rows, 22) raw
    apply_decoupling(raw)     →  array (total_rows, 22) decoupled
    segregate_by_angle(data)  →  Dict[theta, AngleGroup] MODEL-MAJOR

Referencia: docs/ARCHITECTURE_v2.md secao 5.1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════
# D8: __all__ — Exports publicos deste modulo
# ════════════════════════════════════════════════════════════════════════
# Controla o que e importado com `from geosteering_ai.data.loading import *`.
# Tambem serve como inventario rapido dos nomes publicos.

__all__ = [
    # Dataclasses
    "OutMetadata",
    "AngleGroup",
    # Constantes
    "COL_MAP_22",
    "EM_COMPONENTS",
    # Funcoes
    "parse_out_metadata",
    "load_binary_dat",
    "apply_decoupling",
    "segregate_by_angle",
    "load_dataset",
]


# ════════════════════════════════════════════════════════════════════════
# CONSTANTES — Formato 22-colunas do PerfilaAnisoOmp
# ────────────────────────────────────────────────────────────────────────
# O simulador Fortran PerfilaAnisoOmp gera arquivos .dat em formato
# binario com registros de tamanho fixo. Cada registro contem 1 int32
# (indice de medicao) seguido de 21 float64 (dados geofisicos).
# Total: 4 + 21*8 = 172 bytes por registro.
# ════════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  LAYOUT 22 COLUNAS (.dat Fortran binary, 172 bytes/registro)           │
# ├────┬──────────┬──────────────────────────────────────────────────────────┤
# │ Col│ Nome     │ Descricao                                               │
# ├────┼──────────┼──────────────────────────────────────────────────────────┤
# │  0 │ meds     │ Indice de medicao (metadata, NUNCA feature)             │
# │  1 │ zobs     │ Profundidade observada [m] → INPUT_FEATURE              │
# │  2 │ res_h    │ Resistividade horizontal [Ohm.m] → OUTPUT_TARGET        │
# │  3 │ res_v    │ Resistividade vertical [Ohm.m] → OUTPUT_TARGET          │
# │ 4-5│ Re/Im Hxx│ Componente planar (acoplamento ACp)                     │
# │ 6-7│ Re/Im Hxy│ Cross-component                                         │
# │ 8-9│ Re/Im Hxz│ Cross-component                                         │
# │10-11│Re/Im Hyx│ Cross-component                                         │
# │12-13│Re/Im Hyy│ Componente planar (acoplamento ACp)                     │
# │14-15│Re/Im Hyz│ Cross-component                                         │
# │16-17│Re/Im Hzx│ Cross-component                                         │
# │18-19│Re/Im Hzy│ Cross-component                                         │
# │20-21│Re/Im Hzz│ Componente axial (acoplamento ACx)                      │
# └────┴──────────┴──────────────────────────────────────────────────────────┘
#
# Indices criticos (errata imutavel — PipelineConfig valida):
#   INPUT_FEATURES  = [1, 4, 5, 20, 21]  (NUNCA [0, 3, 4, 7, 8])
#   OUTPUT_TARGETS  = [2, 3]              (NUNCA [1, 2])

# D10: Tamanho do registro binario no formato 22-colunas.
# Composicao: 1×int32 (4 bytes, coluna meds) + 21×float64 (168 bytes) = 172 bytes.
# Este valor e usado para validar que o tamanho do .dat e divisivel por 172.
_BYTES_PER_RECORD_22 = 172

# D10: Mapeamento semantico de cada coluna do .dat para nome legivel.
# Coluna 0 (meds) e int32 no binario, convertida para float64 na leitura.
# Colunas 1-21 sao float64 nativas.
# As 9 componentes complexas do tensor H formam pares (Re, Im) nas colunas 4-21.
COL_MAP_22 = {
    0: "meds",       # int32 — indice sequencial de medicao (metadata, NUNCA feature)
    1: "zobs",       # float64 — profundidade de observacao [m] (INPUT_FEATURE idx=1)
    2: "res_h",      # float64 — resistividade horizontal [Ohm.m] (OUTPUT_TARGET idx=2)
    3: "res_v",      # float64 — resistividade vertical [Ohm.m] (OUTPUT_TARGET idx=3)
    4: "re_hxx",  5: "im_hxx",   # Hxx — planar, recebe decoupling ACp
    6: "re_hxy",  7: "im_hxy",   # Hxy — cross (off-diagonal)
    8: "re_hxz",  9: "im_hxz",   # Hxz — cross
    10: "re_hyx", 11: "im_hyx",  # Hyx — cross (off-diagonal)
    12: "re_hyy", 13: "im_hyy",  # Hyy — planar, recebe decoupling ACp
    14: "re_hyz", 15: "im_hyz",  # Hyz — cross
    16: "re_hzx", 17: "im_hzx",  # Hzx — cross
    18: "re_hzy", 19: "im_hzy",  # Hzy — cross
    20: "re_hzz", 21: "im_hzz",  # Hzz — axial, recebe decoupling ACx (INPUT_FEATURE idx=20,21)
}

# D10: Mapeamento nome-da-componente → (indice_Re, indice_Im) no array 22-col.
# Usado para selecao programatica de componentes do tensor EM.
# As 9 componentes formam o tensor 3x3 do campo magnetico H:
#   ┌              ┐
#   │ Hxx  Hxy  Hxz│
#   │ Hyx  Hyy  Hyz│  →  cada entrada e complexa (Re + j*Im)
#   │ Hzx  Hzy  Hzz│
#   └              ┘
EM_COMPONENTS = {
    "XX": (4, 5),   "XY": (6, 7),   "XZ": (8, 9),
    "YX": (10, 11), "YY": (12, 13), "YZ": (14, 15),
    "ZX": (16, 17), "ZY": (18, 19), "ZZ": (20, 21),
}


# ════════════════════════════════════════════════════════════════════════
# DATACLASSES — Metadados e resultados
# ────────────────────────────────────────────────────────────────────────
# OutMetadata encapsula informacoes do .out (header do simulador).
# AngleGroup encapsula dados segregados para um angulo especifico.
# Ambos sao imutaveis apos criacao (exceto campos calculados no
# __post_init__).
# ════════════════════════════════════════════════════════════════════════

@dataclass
class OutMetadata:
    """Metadados do arquivo .out do simulador Fortran.

    O arquivo .out e gerado pelo PerfilaAnisoOmp e contem exatamente
    4 linhas de texto com a configuracao da simulacao. Este dataclass
    armazena os campos parseados e calcula derivados (total_rows,
    rows_per_model) no __post_init__.

    Attributes:
        n_angles: Numero de angulos de inclinacao.
        n_freqs: Numero de frequencias.
        n_models: Numero total de modelos geologicos.
        theta_list: Lista de angulos (graus).
        freq_list: Lista de frequencias (Hz).
        nmeds_list: Medicoes por angulo.
        total_rows: Linhas totais no .dat (calculado).
        rows_per_model: Linhas por modelo geologico (calculado).

    Note:
        Referenciado em:
            - data/loading.py: load_dataset() (retorno de parse_out_metadata)
            - data/loading.py: segregate_by_angle() (parametro metadata)
            - data/pipeline.py: DataPipeline.prepare() (Step 1, metadados)
            - data/pipeline.py: PreparedData.metadata (campo armazenado)
            - tests/test_data_pipeline.py: TestParseOut (3 test cases)
        Ref: docs/ARCHITECTURE_v2.md secao 4.1.
        Formato .out: 4 linhas fixas geradas pelo PerfilaAnisoOmp.
        total_rows e rows_per_model sao auto-calculados no __post_init__.
    """
    n_angles: int
    n_freqs: int
    n_models: int
    theta_list: List[float]
    freq_list: List[float]
    nmeds_list: List[int]
    total_rows: int = 0
    rows_per_model: int = 0

    def __post_init__(self):
        if self.total_rows == 0:
            self.rows_per_model = sum(
                nm * self.n_freqs for nm in self.nmeds_list
            )
            self.total_rows = self.n_models * self.rows_per_model


@dataclass
class AngleGroup:
    """Dados segregados para um angulo especifico.

    Resultado da segregacao MODEL-MAJOR: cada AngleGroup contem todas
    as sequencias (modelos × frequencias) para um dado theta, ja com
    features/targets selecionados conforme config.input_features e
    config.output_targets.

    Attributes:
        theta: Angulo de inclinacao (graus).
        x: Features, shape (n_seq, seq_len, n_features).
        y: Targets, shape (n_seq, seq_len, n_targets).
        z_meters: Profundidade em metros, shape (n_seq, seq_len).
        model_ids: ID do modelo geologico por sequencia, shape (n_seq,).
        nmeds: Numero de medicoes para este angulo.

    Note:
        Referenciado em:
            - data/loading.py: segregate_by_angle() (retorno)
            - data/splitting.py: apply_split() (parametro angle_group)
            - data/splitting.py: split_angle_group() (parametro angle_group)
            - data/pipeline.py: DataPipeline.prepare() (Step 2, selecao angulo)
            - tests/test_data_pipeline.py: _make_synthetic_angle_group (fixture)
        Ref: docs/ARCHITECTURE_v2.md secao 4.2.
        model_ids: usado para split por modelo geologico [P1].
        z_meters: preservado separadamente (NUNCA escalado pelo scaler).
    """
    theta: float
    x: np.ndarray
    y: np.ndarray
    z_meters: np.ndarray
    model_ids: np.ndarray
    nmeds: int


# ════════════════════════════════════════════════════════════════════════
# PARSING .out — Extracao de metadados do simulador Fortran
# ────────────────────────────────────────────────────────────────────────
# O arquivo .out e um header de texto gerado pelo PerfilaAnisoOmp
# que descreve a configuracao da simulacao. Formato fixo de 4 linhas:
#   L1: nt nf nm        (contadores)
#   L2: theta_0 ... theta_{nt-1}  (angulos em graus)
#   L3: f_0 ... f_{nf-1}          (frequencias em Hz)
#   L4: nmeds_0 ... nmeds_{nt-1}  (medicoes por angulo)
# ════════════════════════════════════════════════════════════════════════

def parse_out_metadata(filepath: str) -> OutMetadata:
    """Extrai metadados do arquivo .out do simulador Fortran.

    O .out tem exatamente 4 linhas:
        1: nt nf nm  (n_angles, n_freqs, n_models)
        2: theta_0 theta_1 ... theta_{nt-1}
        3: f_0 f_1 ... f_{nf-1}
        4: nmeds_0 nmeds_1 ... nmeds_{nt-1}

    Args:
        filepath: Caminho para o arquivo .out.

    Returns:
        OutMetadata com todos os campos preenchidos.

    Raises:
        FileNotFoundError: Se o arquivo nao existir.
        ValueError: Se o formato for invalido.

    Note:
        Referenciado em:
            - data/loading.py: load_dataset() (Step 1 da cadeia)
            - data/pipeline.py: DataPipeline.prepare() (Step 1, duplicado
              para capturar metadata separadamente)
            - tests/test_data_pipeline.py: TestParseOut (3 test cases)
        Ref: docs/ARCHITECTURE_v2.md secao 4.1.
        Formato do .out: 4 linhas de texto fixo (PerfilaAnisoOmp).
        Frequencia validada externamente: FREQUENCY_HZ = 20000.0 (Errata v4.4.5).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo .out nao encontrado: {filepath}")

    lines = path.read_text().strip().split("\n")
    if len(lines) < 4:
        raise ValueError(
            f"Arquivo .out deve ter >= 4 linhas, encontradas: {len(lines)}"
        )

    # Linha 1: nt nf nm
    tokens = lines[0].split()
    if len(tokens) < 3:
        raise ValueError(f"Linha 1 do .out invalida: '{lines[0]}'")
    n_angles = int(tokens[0])
    n_freqs = int(tokens[1])
    n_models = int(tokens[2])

    # Linha 2: angulos
    theta_list = [float(x) for x in lines[1].split()]
    if len(theta_list) != n_angles:
        raise ValueError(
            f"Esperados {n_angles} angulos, encontrados {len(theta_list)}"
        )

    # Linha 3: frequencias
    freq_list = [float(x) for x in lines[2].split()]
    if len(freq_list) != n_freqs:
        raise ValueError(
            f"Esperadas {n_freqs} frequencias, encontradas {len(freq_list)}"
        )

    # Linha 4: nmeds por angulo
    nmeds_list = [int(x) for x in lines[3].split()]
    if len(nmeds_list) != n_angles:
        raise ValueError(
            f"Esperados {n_angles} nmeds, encontrados {len(nmeds_list)}"
        )

    metadata = OutMetadata(
        n_angles=n_angles,
        n_freqs=n_freqs,
        n_models=n_models,
        theta_list=theta_list,
        freq_list=freq_list,
        nmeds_list=nmeds_list,
    )

    logger.info(
        "Metadados .out: %d angulos, %d freqs, %d modelos, %d linhas totais",
        n_angles, n_freqs, n_models, metadata.total_rows,
    )
    return metadata


# ════════════════════════════════════════════════════════════════════════
# CARREGAMENTO .dat (BINARIO 22-COL)
# ────────────────────────────────────────────────────────────────────────
# O .dat e um arquivo binario SEM header, com registros de tamanho fixo.
# Formato 22-col (ativo): 1×int32 (4B) + 21×float64 (168B) = 172B/reg.
# Formato 12-col (legado): 12×float64 (96B) = 96B/reg.
# A coluna 0 (meds, int32) e promovida para float64 no array de saida.
# ════════════════════════════════════════════════════════════════════════

def load_binary_dat(
    filepath: str,
    n_columns: int = 22,
    expected_rows: Optional[int] = None,
) -> np.ndarray:
    """Carrega arquivo binario .dat no formato 22-colunas.

    Formato por registro: 1×int32 + 21×float64 = 172 bytes.
    Coluna 0 (meds, int32) e convertida para float64.

    Args:
        filepath: Caminho para o arquivo .dat.
        n_columns: Numero de colunas (22 para formato ativo).
        expected_rows: Linhas esperadas (validacao opcional).

    Returns:
        Array float64 com shape (total_rows, n_columns).

    Raises:
        FileNotFoundError: Se o arquivo nao existir.
        ValueError: Se o tamanho nao bater com expected_rows.

    Note:
        Referenciado em:
            - data/loading.py: load_dataset() (Step 2 da cadeia)
            - tests/test_data_pipeline.py: TestLoadDat (3 test cases)
        Ref: docs/ARCHITECTURE_v2.md secao 4.1.
        Formato 22-col: 172 bytes/registro (1×int32 + 21×float64).
        Formato 12-col: 96 bytes/registro (legado, tudo float64).
        Coluna 0 (meds) e metadata — NUNCA usada como feature (Errata v5.0.15).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo .dat nao encontrado: {filepath}")

    file_size = path.stat().st_size

    if n_columns == 22:
        # ── 22-col ativo: 172 bytes/registro (1×int32 + 21×float64).
        #    Formato padrao do PerfilaAnisoOmp v5+.
        #    Coluna 0 (meds, int32) e promovida para float64 no array de saida.
        n_rows = file_size // _BYTES_PER_RECORD_22
        if file_size % _BYTES_PER_RECORD_22 != 0:
            raise ValueError(
                f"Tamanho do .dat ({file_size}B) nao e multiplo de "
                f"{_BYTES_PER_RECORD_22}B/registro"
            )

        # dtype estruturado: 1×int32 + 21×float64
        dtype_22 = np.dtype(
            [("meds", np.int32)]
            + [(f"col_{i}", np.float64) for i in range(1, n_columns)]
        )
        raw = np.fromfile(str(path), dtype=dtype_22)

        # Converter para array float64 homogeneo (meds int32 → float64)
        data = np.zeros((n_rows, n_columns), dtype=np.float64)
        data[:, 0] = raw["meds"].astype(np.float64)
        for i in range(1, n_columns):
            data[:, i] = raw[f"col_{i}"]
    else:
        # ── 12-col legado: tudo float64 (96 bytes/registro).
        #    Formato antigo sem coluna meds (pre-PerfilaAnisoOmp v5).
        #    Suportado para compatibilidade reversa.
        bytes_per_record = n_columns * 8
        n_rows = file_size // bytes_per_record
        data = np.fromfile(str(path), dtype=np.float64).reshape(n_rows, n_columns)

    if expected_rows is not None and n_rows != expected_rows:
        raise ValueError(
            f"Esperadas {expected_rows} linhas, encontradas {n_rows}"
        )

    logger.info(
        "Carregado %s: shape=%s, %.1f MB",
        path.name, data.shape, file_size / 1e6,
    )
    return data


# ════════════════════════════════════════════════════════════════════════
# DECOUPLING EM — Remocao do acoplamento direto Tx-Rx
# ────────────────────────────────────────────────────────────────────────
# O campo medido contem o campo primario (acoplamento direto entre
# transmissor e receptor) somado ao campo secundario (resposta da
# formacao). O decoupling remove o campo primario analitico para
# isolar a resposta geofisica.
#
# Para L = SPACING_METERS (1.0 m, errata imutavel):
#   ACp = -1 / (4*pi*L^3) ≈ -0.079577  (planar: Hxx, Hyy)
#   ACx = +1 / (2*pi*L^3) ≈ +0.159155  (axial: Hzz)
#
# Nota: O sinal de ACp e negativo porque as componentes planares
# medem o campo na direcao perpendicular ao eixo da ferramenta.
# ACx e positivo porque Hzz mede na direcao axial.
# ════════════════════════════════════════════════════════════════════════

def apply_decoupling(
    data: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """Remove acoplamento direto (primary field) das componentes EM.

    Para L = SPACING_METERS:
        ACp = -1/(4*pi*L^3)  (planar: Hxx, Hyy)
        ACx = +1/(2*pi*L^3)  (axial: Hzz)

    Args:
        data: Array (n_rows, 22) com dados brutos do .dat.
        config: PipelineConfig com spacing_meters e flags de decoupling.

    Returns:
        Copia do array com decoupling aplicado.

    Note:
        Referenciado em:
            - data/loading.py: load_dataset() (Step 3 da cadeia)
            - data/pipeline.py: DataPipeline.prepare() (via load_dataset, Step 1)
            - tests/test_data_pipeline.py: TestDecoupling (2 test cases)
        Ref: docs/physics/errata_valores.md (ACp/ACx constants).
        Constantes fisicas: ACp = -1/(4*pi*L^3) = -0.079577 (planar: Hxx, Hyy)
                           ACx = +1/(2*pi*L^3) = +0.159155 (axial: Hzz)
        L = SPACING_METERS = 1.0 m (NUNCA 1000.0 m — Errata v4.4.5).
        Decoupling subtrai campo primario (free-space) da parte real APENAS.
        Parte imaginaria NAO recebe decoupling (campo primario e puramente real).
    """
    import math

    result = data.copy()
    L = config.spacing_meters  # 1.0 m (errata imutavel — NUNCA 1000.0)
    L3 = L ** 3

    # D7: Constantes de acoplamento analitico
    # ACp (planar): campo primario para componentes no plano (Hxx, Hyy)
    # Derivado da solucao analitica para dipolo magnetico em meio homogeneo
    ACp = -1.0 / (4.0 * math.pi * L3)  # ≈ -0.079577 A/m (para L=1.0 m)

    # ACx (axial): campo primario para componente axial (Hzz)
    # Fator 2× maior que ACp e sinal oposto (geometria coaxial)
    ACx = +1.0 / (2.0 * math.pi * L3)  # ≈ +0.159155 A/m (para L=1.0 m)

    # Componentes planares: subtrair ACp da parte real (Re)
    # Parte imaginaria NAO recebe decoupling (campo primario e puramente real)
    if config.decoupling_hxx:
        # ── Hxx planar: Re{Hxx} -= ACp (campo primario no plano Hxx).
        #    col 4 = Re{Hxx}, componente diagonal do tensor EM.
        #    ACp < 0 → subtrai valor negativo → resultado AUMENTA.
        result[:, 4] -= ACp   # Re{Hxx} — col 4 (planar)
    if config.decoupling_hyy:
        # ── Hyy planar: Re{Hyy} -= ACp (simetria com Hxx).
        #    col 12 = Re{Hyy}, componente diagonal do tensor EM.
        #    Mesmo coeficiente ACp que Hxx (ambas planares).
        result[:, 12] -= ACp  # Re{Hyy} — col 12 (planar)

    # Componente axial: subtrair ACx da parte real
    if config.decoupling_hzz:
        # ── Hzz axial: Re{Hzz} -= ACx (campo primario coaxial).
        #    col 20 = Re{Hzz}, componente axial (INPUT_FEATURE).
        #    ACx > 0 → subtrai valor positivo → resultado DIMINUI.
        result[:, 20] -= ACx  # Re{Hzz} — col 20 (axial)

    if config.decoupling_full_tensor:
        # ── Full tensor: off-diagonais Hxy, Hyx recebem ACp.
        #    Componentes off-diagonal do plano tambem possuem
        #    acoplamento direto com coeficiente planar ACp.
        result[:, 6] -= ACp   # Re{Hxy} — col 6
        result[:, 10] -= ACp  # Re{Hyx} — col 10

    n_decoupled = sum([
        config.decoupling_hxx, config.decoupling_hyy,
        config.decoupling_hzz, config.decoupling_full_tensor,
    ])
    logger.info(
        "Decoupling aplicado: ACp=%.6f, ACx=%.6f (%d componentes)",
        ACp, ACx, n_decoupled,
    )
    return result


# ════════════════════════════════════════════════════════════════════════
# SEGREGACAO MODEL-MAJOR — Agrupamento por angulo de inclinacao
# ────────────────────────────────────────────────────────────────────────
# O .dat armazena dados em ordem MODEL-MAJOR (modelo geologico e o
# loop mais externo). A segregacao reorganiza os dados em grupos por
# angulo, onde cada grupo contem todas as sequencias daquele angulo.
#
# Ordem de iteracao no .dat:
#   for m in range(n_models):         ← loop externo (MODEL-MAJOR)
#       for k in range(n_angles):     ← angulo de inclinacao
#           for j in range(n_freqs):  ← frequencia
#               → nmeds[k] linhas    ← registros deste bloco
#
# Cada bloco (m, k, j) vira uma sequencia no AngleGroup de theta[k].
# ════════════════════════════════════════════════════════════════════════

def segregate_by_angle(
    data: np.ndarray,
    metadata: OutMetadata,
    config: PipelineConfig,
) -> Dict[float, AngleGroup]:
    """Segrega dados em grupos por angulo, layout MODEL-MAJOR.

    A ordem no .dat e:
        for m in range(n_models):
            for k in range(n_angles):
                for j in range(n_freqs):
                    → nmeds[k] linhas

    Args:
        data: Array (total_rows, 22) do .dat.
        metadata: OutMetadata com angulos, freqs, nmeds.
        config: PipelineConfig com input_features, output_targets.

    Returns:
        Dict[theta, AngleGroup] com x, y, z_meters, model_ids.

    Note:
        Referenciado em:
            - data/loading.py: load_dataset() (Step 4 da cadeia)
            - data/pipeline.py: DataPipeline.prepare() (via load_dataset, Step 1)
        Ref: docs/ARCHITECTURE_v2.md secao 4.2.
        Layout MODEL-MAJOR: modelo geologico e o loop mais externo.
        model_ids atribuidos sequencialmente [0, N-1] — requisito para
        split_model_ids (splitting.py) que assume contiguidade.
        INPUT_FEATURES = [1, 4, 5, 20, 21] (Errata v5.0.15 — NUNCA [0, 3, 4, 7, 8]).
        OUTPUT_TARGETS = [2, 3] (Errata v5.0.15 — NUNCA [1, 2]).
    """
    groups: Dict[float, dict] = {}
    for k, theta in enumerate(metadata.theta_list):
        groups[theta] = {
            "x_seqs": [],
            "y_seqs": [],
            "z_seqs": [],
            "model_ids": [],
        }

    row_idx = 0
    for m in range(metadata.n_models):
        for k in range(metadata.n_angles):
            theta = metadata.theta_list[k]
            nmeds = metadata.nmeds_list[k]
            for _j in range(metadata.n_freqs):
                block = data[row_idx : row_idx + nmeds]
                row_idx += nmeds

                # z_meters: coluna 1 (profundidade em metros — NUNCA escalada)
                z_seq = block[:, 1].copy()

                # Features: colunas definidas em config.input_features
                # Default: [1, 4, 5, 20, 21] = [zobs, Re/Im Hxx, Re/Im Hzz]
                x_seq = block[:, config.input_features].copy()

                # Targets: colunas definidas em config.output_targets
                # Default: [2, 3] = [res_h, res_v]
                y_seq = block[:, config.output_targets].copy()

                groups[theta]["x_seqs"].append(x_seq)
                groups[theta]["y_seqs"].append(y_seq)
                groups[theta]["z_seqs"].append(z_seq)
                groups[theta]["model_ids"].append(m)

    result = {}
    for theta, grp in groups.items():
        n_seq = len(grp["x_seqs"])
        nmeds = metadata.nmeds_list[metadata.theta_list.index(theta)]

        result[theta] = AngleGroup(
            theta=theta,
            x=np.array(grp["x_seqs"], dtype=np.float64),   # (n_seq, nmeds, n_feat)
            y=np.array(grp["y_seqs"], dtype=np.float64),   # (n_seq, nmeds, n_tgt)
            z_meters=np.array(grp["z_seqs"], dtype=np.float64),  # (n_seq, nmeds)
            model_ids=np.array(grp["model_ids"], dtype=np.int32),  # (n_seq,)
            nmeds=nmeds,
        )
        logger.info(
            "Angulo %.1f°: %d sequencias, nmeds=%d, x=%s, y=%s",
            theta, n_seq, nmeds,
            result[theta].x.shape, result[theta].y.shape,
        )

    if row_idx != data.shape[0]:
        raise ValueError(
            f"Consumidas {row_idx} linhas, esperadas {data.shape[0]}. "
            "Verifique .out vs .dat."
        )

    return result


# ════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO — Funcao de conveniencia
# ────────────────────────────────────────────────────────────────────────
# Encadeia as 4 etapas de carregamento em uma unica chamada:
#   .out → metadados → .dat → raw → decoupling → segregacao
# Usado por DataPipeline.prepare() como ponto de entrada principal.
# ════════════════════════════════════════════════════════════════════════

def load_dataset(
    dat_path: str,
    out_path: str,
    config: PipelineConfig,
) -> Dict[float, AngleGroup]:
    """Pipeline completo: parse .out → load .dat → decoupling → segregate.

    Funcao de conveniencia que executa as 3 etapas de carregamento.

    Args:
        dat_path: Caminho para o arquivo .dat.
        out_path: Caminho para o arquivo .out.
        config: PipelineConfig.

    Returns:
        Dict[theta, AngleGroup] com dados segregados por angulo.

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline.prepare() (Step 1, ponto de
              entrada principal para carregamento)
            - data/__init__.py: re-exportado como API publica
        Ref: docs/ARCHITECTURE_v2.md secao 4.1.
        Cadeia interna: parse_out_metadata → load_binary_dat → apply_decoupling
        → segregate_by_angle. Todas as etapas sao executadas em sequencia.
        Config necessario: n_columns, spacing_meters, decoupling_*, input_features,
        output_targets.
    """
    metadata = parse_out_metadata(out_path)
    raw_data = load_binary_dat(dat_path, config.n_columns, metadata.total_rows)
    decoupled = apply_decoupling(raw_data, config)
    return segregate_by_angle(decoupled, metadata, config)
