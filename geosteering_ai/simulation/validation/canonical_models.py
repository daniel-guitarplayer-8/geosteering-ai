# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/validation/canonical_models.py                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Modelos canônicos (Sprint 2.9+)        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy                                                      ║
# ║  Dependências: numpy, geosteering_ai.simulation                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Biblioteca de 7 modelos geológicos canônicos de referência usados     ║
# ║    para validação de inversão 1D de resistividade. Port direto de       ║
# ║    `buildValidamodels.py::modelo_valida()` + adição do Viking Graben.   ║
# ║                                                                           ║
# ║  CATÁLOGO DE MODELOS                                                      ║
# ║    ┌─────┬────────────────────────┬───────┬──────────────┬───────────┐  ║
# ║    │  Id │  Nome                  │ Cam.  │  Tipo        │  Ref.     │  ║
# ║    ├─────┼────────────────────────┼───────┼──────────────┼───────────┤  ║
# ║    │  0  │  Oklahoma 3            │   3   │  TIV simples │  TR 32    │  ║
# ║    │  1  │  Oklahoma 5            │   5   │  TIV gradual │  TR 32    │  ║
# ║    │  2  │  Devine 8              │   8   │  Isotrópico  │  TR 32    │  ║
# ║    │  3  │  Oklahoma 15           │  15   │  Isotrópico  │  TR 32    │  ║
# ║    │  4  │  Oklahoma 28           │  28   │  TIV forte   │  TR 32    │  ║
# ║    │  5  │  Hou et al. 7          │   7   │  TIV          │  Hou 2006 │  ║
# ║    │  6  │  Viking Graben 10      │  10   │  TIV (North  │  NEW      │  ║
# ║    │     │                        │       │  Sea salt)   │           │  ║
# ║    └─────┴────────────────────────┴───────┴──────────────┴───────────┘  ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Tech. Report 32_2011 — University of Texas at Austin                ║
# ║    • Hou, Mallan & Verdin (2006), Geophysics 71(5), F101–F114            ║
# ║    • Viking Graben: Eidesmo et al. (2002) — "Sea Bed Logging (SBL)...",║
# ║      First Break 20(3) 144–152                                           ║
# ║    • Fortran_Gerador/buildValidamodels.py — implementação original      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Biblioteca de 7 modelos geológicos canônicos para validação.

Cada modelo retorna um :class:`CanonicalModel` com resistividades ρₕ/ρᵥ,
espessuras e metadados (nome, referência, tipo). Úteis para testes de
regressão cruzada Fortran ↔ Python e plotagens reprodutíveis.

Example:
    Carregar modelo Oklahoma 3 e simular::

        >>> import numpy as np
        >>> from geosteering_ai.simulation.validation import get_canonical_model
        >>> from geosteering_ai.simulation import simulate, SimulationConfig
        >>> m = get_canonical_model("oklahoma_3")
        >>> cfg = SimulationConfig(frequency_hz=20000.0, tr_spacing_m=1.0)
        >>> positions_z = np.linspace(
        ...     m.min_depth - 2.0,
        ...     m.max_depth + 2.0,
        ...     200,
        ... )
        >>> result = simulate(
        ...     rho_h=m.rho_h, rho_v=m.rho_v, esp=m.esp,
        ...     positions_z=positions_z, cfg=cfg,
        ... )
        >>> result.H_tensor.shape
        (200, 1, 9)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
_FT_TO_M: float = 0.3048  # 1 pé = 0.3048 m (Oklahoma/Devine em pés originalmente)

ModelId = Literal[
    "oklahoma_3",
    "oklahoma_5",
    "devine_8",
    "oklahoma_15",
    "oklahoma_28",
    "hou_7",
    "viking_graben_10",
]


# ──────────────────────────────────────────────────────────────────────────────
# CanonicalModel — container de resultado
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CanonicalModel:
    """Descritor de um modelo geológico canônico (imutável).

    Attributes:
        name: Nome canônico (ex.: ``"oklahoma_3"``).
        title: Nome legível em PT-BR (ex.: ``"Oklahoma 3 camadas"``).
        n_layers: Número total de camadas (inclui 2 semi-espaços).
        rho_h: Array (n_layers,) com resistividades horizontais (Ω·m).
        rho_v: Array (n_layers,) com resistividades verticais (Ω·m).
        esp: Array (n_layers-2,) com espessuras internas (m).
        reference: Referência bibliográfica curta.
        anisotropy_type: Tipo da anisotropia (``"isotropic"``, ``"tiv"``,
            ``"tiv_strong"``).

    Properties:
        interfaces: Profundidades das n_layers-1 interfaces (m).
        min_depth, max_depth: Range coberto pelo modelo (m).
        rho_h_min, rho_h_max: Range de ρₕ em Ω·m.
    """

    name: str
    title: str
    n_layers: int
    rho_h: np.ndarray
    rho_v: np.ndarray
    esp: np.ndarray
    reference: str
    anisotropy_type: str

    @property
    def interfaces(self) -> np.ndarray:
        """Profundidades das interfaces em m (shape ``(n_layers-1,)``).

        Note:
            A primeira interface está em z=0 (topo da primeira camada
            interna); as demais são acumuladas de ``esp``.
        """
        return np.concatenate(([0.0], np.cumsum(self.esp)))

    @property
    def min_depth(self) -> float:
        """Menor profundidade de interface (m) — topo do modelo."""
        interfaces = self.interfaces
        return float(interfaces[0])

    @property
    def max_depth(self) -> float:
        """Maior profundidade de interface (m) — base do modelo."""
        interfaces = self.interfaces
        return float(interfaces[-1])

    @property
    def rho_h_min(self) -> float:
        return float(np.min(self.rho_h))

    @property
    def rho_h_max(self) -> float:
        return float(np.max(self.rho_h))


# ──────────────────────────────────────────────────────────────────────────────
# CATÁLOGO DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────

_pe = _FT_TO_M


def _make_oklahoma_3() -> CanonicalModel:
    """Oklahoma 3 camadas — TIV simples (ρᵥ = 2ρₕ na camada média)."""
    return CanonicalModel(
        name="oklahoma_3",
        title="Oklahoma 3 camadas (TIV simples)",
        n_layers=3,
        rho_h=np.array([1.0, 20.0, 1.0], dtype=np.float64),
        rho_v=np.array([1.0, 40.0, 1.0], dtype=np.float64),
        esp=np.array([8.0 * _pe], dtype=np.float64),
        reference="Tech. Report 32_2011",
        anisotropy_type="tiv",
    )


def _make_oklahoma_5() -> CanonicalModel:
    """Oklahoma 5 camadas — TIV gradual."""
    return CanonicalModel(
        name="oklahoma_5",
        title="Oklahoma 5 camadas (TIV gradual)",
        n_layers=5,
        rho_h=np.array([5.0, 50.0, 2.0, 15.0, 4.5], dtype=np.float64),
        rho_v=np.array([10.0, 100.0, 4.0, 30.0, 9.0], dtype=np.float64),
        esp=np.array([17.0 * _pe, 8.0 * _pe, 4.0 * _pe], dtype=np.float64),
        reference="Tech. Report 32_2011",
        anisotropy_type="tiv",
    )


def _make_devine_8() -> CanonicalModel:
    """Devine 8 camadas — isotrópico (ρᵥ = ρₕ)."""
    rho_h = np.array(
        [2.0, 3.0, 1.8, 4.71, 8.89, 5.0, 3.2, 8.0],
        dtype=np.float64,
    )
    return CanonicalModel(
        name="devine_8",
        title="Devine 8 camadas (isotrópico)",
        n_layers=8,
        rho_h=rho_h,
        rho_v=rho_h.copy(),
        esp=np.array(
            [20.0 * _pe, 15.0 * _pe, 19.0 * _pe, 16.0 * _pe, 15.0 * _pe, 45.0 * _pe],
            dtype=np.float64,
        ),
        reference="Tech. Report 32_2011",
        anisotropy_type="isotropic",
    )


def _make_oklahoma_15() -> CanonicalModel:
    """Oklahoma 15 camadas — isotrópico, ρₕ spans 2 ordens de magnitude."""
    rho_h = np.array(
        [
            5.0,
            50.0,
            2.0,
            15.0,
            4.5,
            100.0,
            3.5,
            450.0,
            30.12,
            602.41,
            20.0,
            746.27,
            200.0,
            7.5,
            500.0,
        ],
        dtype=np.float64,
    )
    esp_ft = [17, 8, 4, 3, 7, 4, 6, 3, 5, 7, 18, 8, 7]
    return CanonicalModel(
        name="oklahoma_15",
        title="Oklahoma 15 camadas (isotrópico)",
        n_layers=15,
        rho_h=rho_h,
        rho_v=rho_h.copy(),
        esp=np.array([v * _pe for v in esp_ft], dtype=np.float64),
        reference="Tech. Report 32_2011",
        anisotropy_type="isotropic",
    )


def _make_oklahoma_28() -> CanonicalModel:
    """Oklahoma 28 camadas — TIV forte (ρᵥ = 2·ρₕ), alta resistividade."""
    z_ft = [
        46,
        63,
        71,
        75,
        78,
        85,
        89,
        95,
        98,
        103,
        110,
        128,
        136,
        143,
        153,
        157,
        162,
        165,
        169,
        173,
        177,
        182,
        185,
        187,
        189,
        191,
        203,
    ]
    esp = np.diff(z_ft).astype(np.float64) * _pe
    rho_h = np.array(
        [
            10.0,
            100.0,
            4.0,
            30.0,
            9.0,
            200.0,
            7.0,
            909.0,
            60.0,
            1250.0,
            40.0,
            1416.0,
            400.0,
            15.0,
            1000.0,
            179.0,
            1000.0,
            15.0,
            75.0,
            9.0,
            20.0,
            100.0,
            18.0,
            200.0,
            75.0,
            149.0,
            7.0,
            11.0,
        ],
        dtype=np.float64,
    )
    return CanonicalModel(
        name="oklahoma_28",
        title="Oklahoma 28 camadas (TIV forte)",
        n_layers=28,
        rho_h=rho_h,
        rho_v=2.0 * rho_h,
        esp=esp,
        reference="Tech. Report 32_2011 p.58",
        anisotropy_type="tiv_strong",
    )


def _make_hou_7() -> CanonicalModel:
    """Hou, Mallan & Verdin (2006) — 7 camadas TIV.

    Modelo de referência para inversão de perfilagem LWD com anisotropia
    TIV. Contém camada condutiva fina (ρ=0.3 Ω·m) — desafia a resolução.
    """
    return CanonicalModel(
        name="hou_7",
        title="Hou et al. (2006) 7 camadas (TIV)",
        n_layers=7,
        rho_h=np.array([1.0, 80.0, 1.0, 10.0, 1.0, 0.3, 1.0], dtype=np.float64),
        rho_v=np.array([10.0, 80.0, 10.0, 10.0, 10.0, 0.3, 10.0], dtype=np.float64),
        esp=np.array([1.52, 2.35, 2.1, 1.88, 0.92], dtype=np.float64),
        reference="Hou, Mallan & Verdin (2006) Geophysics 71(5) F101",
        anisotropy_type="tiv",
    )


def _make_viking_graben_10() -> CanonicalModel:
    """Viking Graben (Mar do Norte) — 10 camadas TIV com reservatório salino.

    Modelo simplificado baseado em perfis LWD do Viking Graben (North Sea),
    representativo de estratigrafia siliciclástica com folhelhos, arenitos
    aquíferos e reservatório de óleo (alta resistividade) em ambiente salino.

    Estratigrafia (topo → base):
      - Água do mar (muito condutiva)
      - Sedimento marinho / lama salina (muito condutiva)
      - Folhelho superior (TIV leve)
      - Arenito aquífero (água salgada, baixa ρ)
      - Folhelho intermediário (TIV)
      - Arenito reservatório (saturado de óleo, alta ρ) ← target
      - Folhelho base
      - Cimento carbonático (alta ρ isotrópico)
      - Folhelho profundo
      - Basalto/embasamento (muito alta ρ)

    Note:
        Valores representativos mas simplificados. Para benchmarks
        quantitativos rigorosos, consultar Eidesmo et al. (2002).
    """
    return CanonicalModel(
        name="viking_graben_10",
        title="Viking Graben 10 camadas (reservatório N. Sea)",
        n_layers=10,
        rho_h=np.array(
            [0.3, 0.5, 2.5, 0.8, 2.0, 50.0, 1.8, 150.0, 2.2, 1000.0],
            dtype=np.float64,
        ),
        rho_v=np.array(
            [0.3, 0.5, 4.5, 0.8, 3.5, 80.0, 3.0, 150.0, 3.8, 1000.0],
            dtype=np.float64,
        ),
        esp=np.array(
            [50.0, 30.0, 25.0, 40.0, 20.0, 15.0, 18.0, 35.0],
            dtype=np.float64,
        ),
        reference="Eidesmo et al. (2002) First Break 20(3) (adapt.)",
        anisotropy_type="tiv",
    )


_REGISTRY = {
    "oklahoma_3": _make_oklahoma_3,
    "oklahoma_5": _make_oklahoma_5,
    "devine_8": _make_devine_8,
    "oklahoma_15": _make_oklahoma_15,
    "oklahoma_28": _make_oklahoma_28,
    "hou_7": _make_hou_7,
    "viking_graben_10": _make_viking_graben_10,
}


# ──────────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────────


def get_canonical_model(name: ModelId) -> CanonicalModel:
    """Retorna um modelo canônico do catálogo.

    Args:
        name: Identificador do modelo. Um de: ``oklahoma_3``,
            ``oklahoma_5``, ``devine_8``, ``oklahoma_15``,
            ``oklahoma_28``, ``hou_7``, ``viking_graben_10``.

    Returns:
        Instância :class:`CanonicalModel` com arrays NumPy prontos
        para :func:`geosteering_ai.simulation.simulate`.

    Raises:
        ValueError: Se ``name`` não estiver no catálogo.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Modelo canônico {name!r} desconhecido. " f"Opções disponíveis: {available}."
        )
    return _REGISTRY[name]()


def list_canonical_models() -> list[str]:
    """Lista os nomes dos modelos canônicos disponíveis.

    Returns:
        Lista ordenada de strings com os ids dos modelos.

    Example:
        >>> list_canonical_models()
        ['devine_8', 'hou_7', 'oklahoma_15', 'oklahoma_28', 'oklahoma_3', ...]
    """
    return sorted(_REGISTRY.keys())


def get_all_canonical_models() -> list[CanonicalModel]:
    """Retorna todos os 7 modelos canônicos em ordem de n_layers crescente.

    Útil para loops de validação em batch.
    """
    models = [factory() for factory in _REGISTRY.values()]
    models.sort(key=lambda m: m.n_layers)
    return models


__all__ = [
    "CanonicalModel",
    "ModelId",
    "get_canonical_model",
    "list_canonical_models",
    "get_all_canonical_models",
]
