# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_canonical_profiles.py                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Adaptador de Perfis Canônicos GUI     ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-19                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Adaptador fino entre a API canônica                                    ║
# ║    ``geosteering_ai.simulation.validation.canonical_models`` e a GUI      ║
# ║    Simulation Manager. Expõe uma lista ordenada de pares                  ║
# ║    (label_ptbr, canonical_key) para popular o QComboBox                   ║
# ║    "Perfil Pré-configurado" na aba Simulador e um helper que aplica       ║
# ║    o perfil selecionado à ParametersPage, preservando os campos que       ║
# ║    representam configuração da ferramenta (h1, tj, p_med, freqs, TR,      ║
# ║    dip, n_models) e sobrescrevendo apenas os campos que representam o     ║
# ║    modelo geológico (n_camadas, ρₕ, ρᵥ, espessuras).                      ║
# ║                                                                           ║
# ║  FONTE DE VERDADE                                                         ║
# ║    Os valores numéricos dos 6 perfis canônicos NÃO são duplicados aqui —  ║
# ║    vêm de ``canonical_models.get_canonical_model(key)``. Qualquer         ║
# ║    alteração nos valores canônicos deve ser feita em                      ║
# ║    ``geosteering_ai/simulation/validation/canonical_models.py``.          ║
# ║                                                                           ║
# ║  PERFIS EXPOSTOS                                                          ║
# ║    ┌──────────────────────────────────────────┬────────────────────────┐  ║
# ║    │  Padrão (configuração atual)             │  key=None (no-op)      │  ║
# ║    │  Oklahoma 3 camadas (TIV simples)        │  key="oklahoma_3"      │  ║
# ║    │  Oklahoma 5 camadas (TIV gradual)        │  key="oklahoma_5"      │  ║
# ║    │  Oklahoma 28 camadas (TIV forte λ=2)     │  key="oklahoma_28"     │  ║
# ║    │  Devine 8 camadas (isotrópico)           │  key="devine_8"        │  ║
# ║    │  Viking Graben 10 (reservatório N. Sea)  │  key="viking_graben_10"│  ║
# ║    └──────────────────────────────────────────┴────────────────────────┘  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Adaptador fino entre ``canonical_models`` e o Simulation Manager GUI."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

# Import preguiçoso da API canônica — evita custo na importação do módulo
# quando o usuário nunca abre o combo. O import real ocorre no helper.


# Ordem do combo (label PT-BR → key em canonical_models). ``None`` = preserva
# configuração atual do usuário (no-op).
CANONICAL_PROFILE_LABELS: List[Tuple[str, Optional[str]]] = [
    ("Padrão (configuração atual)", None),
    ("Oklahoma 3 camadas (TIV simples)", "oklahoma_3"),
    ("Oklahoma 5 camadas (TIV gradual)", "oklahoma_5"),
    ("Oklahoma 15 camadas (TIV intermediário)", "oklahoma_15"),
    ("Oklahoma 28 camadas (TIV forte λ²=2, ρᵥ=2·ρₕ)", "oklahoma_28"),
    ("Devine 8 camadas (isotrópico)", "devine_8"),
    ("Hou et al. 7 camadas (TIV)", "hou_7"),
    ("Viking Graben 10 (reservatório N. Sea)", "viking_graben_10"),
]

# Chaves usadas por ``buildValidamodels.py`` como conjunto de referência para
# calcular ``tj_ref = max(Σesp) + margem``. Exclui Viking Graben 10 (Σesp=233 m,
# escala muito maior que os modelos de validação clássicos). Para reproduzir
# fielmente os valores de ``buildValidamodels.py`` (tj=67.85 m, margem=20 m,
# max_Σesp=47.85 m em Oklahoma 28), a fórmula automática usa este subconjunto.
BUILDVALID_REFERENCE_KEYS: Tuple[str, ...] = (
    "oklahoma_3",
    "oklahoma_5",
    "devine_8",
    "oklahoma_15",
    "oklahoma_28",
    "hou_7",
)

# Margem (m) aplicada em cima de max(Σesp) para obter tj_ref. Valor herdado de
# ``buildValidamodels.py``:
#   tj_v = maior espessura (47.85 m em Oklahoma 28) + 20 m margem = 67.85 m.
BUILDVALID_MARGIN_M: float = 20.0


def get_profile_by_label(label: str) -> Optional[Any]:
    """Retorna o :class:`CanonicalModel` associado a um label, ou ``None``.

    ``None`` é retornado tanto quando o label é "Padrão (configuração atual)"
    quanto quando o label é desconhecido — em ambos os casos a UI deve
    preservar os parâmetros atuais sem tocar.

    Args:
        label: Texto exibido no QComboBox "Perfil Pré-configurado".

    Returns:
        ``CanonicalModel`` com campos ``name``, ``title``, ``n_layers``,
        ``rho_h``, ``rho_v``, ``esp``, ``reference``, ``anisotropy_type``,
        ou ``None`` se o label não corresponder a um perfil canônico.
    """
    for lbl, key in CANONICAL_PROFILE_LABELS:
        if lbl == label and key is not None:
            from geosteering_ai.simulation.validation.canonical_models import (
                get_canonical_model,
            )

            return get_canonical_model(key)
    return None


def format_profile_info(model: Any) -> str:
    """Formata um ``CanonicalModel`` como string curta para QLabel.

    Exemplo de saída:
        "Oklahoma 3 camadas — 3 camadas | ρₕ: 1.0–20.0 Ω·m |
         λ=2.0 | Ref: Moran-Gianzero (1979)"
    """
    if model is None:
        return ""
    try:
        rho_h = list(model.rho_h)
        rho_v = list(model.rho_v)
        rho_min = min(rho_h)
        rho_max = max(rho_h)
        # λ² médio = média(ρᵥ/ρₕ) — informativo apenas
        lam_sq = [rv / rh for rv, rh in zip(rho_v, rho_h) if rh > 0]
        lam_avg = (sum(lam_sq) / len(lam_sq)) ** 0.5 if lam_sq else 1.0
        ref = getattr(model, "reference", "") or ""
        aniso = getattr(model, "anisotropy_type", "") or ""
        aniso_txt = (
            f" | λ≈{lam_avg:.2f}"
            if aniso == "tiv" and abs(lam_avg - 1.0) > 1e-3
            else " | isotrópico" if aniso == "isotropic" else ""
        )
        return (
            f"{model.title} — {int(model.n_layers)} camadas | "
            f"ρₕ: {rho_min:.2f}–{rho_max:.2f} Ω·m{aniso_txt} | "
            f"Ref: {ref}"
        )
    except Exception:
        return getattr(model, "title", "") or ""


def build_single_model_dict(canonical_model: Any) -> dict:
    """Converte um :class:`CanonicalModel` em ``dict`` compatível com os
    workers do Simulation Manager (``run_numba_chunk`` / ``run_fortran_chunk``).

    Os workers esperam dicionários com as chaves ``rho_h``, ``rho_v``,
    ``thicknesses`` e ``n_layers`` (formato de ``sm_model_gen.generate_models``).

    Args:
        canonical_model: Instância retornada por ``get_canonical_model``.

    Returns:
        ``dict`` com listas Python nativas (picklable para IPC spawn).
    """
    return {
        "n_layers": int(canonical_model.n_layers),
        "rho_h": [float(x) for x in canonical_model.rho_h],
        "rho_v": [float(x) for x in canonical_model.rho_v],
        "thicknesses": [float(x) for x in canonical_model.esp],
    }


def compute_canonical_reference_tj(
    current_esp_sum: float = 0.0,
    margin_m: float = BUILDVALID_MARGIN_M,
) -> float:
    """Calcula ``tj_ref`` — janela de investigação canônica global.

    Reproduz a convenção de ``buildValidamodels.py``::

        tj_ref = max(Σesp de todos os modelos de validação) + margem

    O ``max`` é calculado sobre ``BUILDVALID_REFERENCE_KEYS`` (6 modelos de
    Anderson 2001 + Devine 2000 + Hou et al.), excluindo Viking Graben 10
    por ser de escala muito diferente (Σesp=233 m vs ~47 m dos clássicos).

    Se o modelo atualmente aplicado tem ``Σesp`` maior que o batch de
    referência (caso Viking Graben 10), a janela é expandida para acomodá-lo:
    ``tj = max(tj_ref_batch, current_esp + margem)``.

    Args:
        current_esp_sum: Σesp do modelo canônico atualmente aplicado, em
            metros. Usado para expandir a janela caso o modelo ultrapasse
            o batch de referência. Default 0.0 (usa só o batch).
        margin_m: Margem total (acima + abaixo) em metros. Default 20.0
            conforme ``buildValidamodels.py``.

    Returns:
        ``tj_ref`` em metros. Para o conjunto padrão + margem=20 m:
        ``tj_ref ≈ 67.85 m`` (max=47.85 m em Oklahoma 28).

    Example:
        >>> compute_canonical_reference_tj()
        67.85
        >>> compute_canonical_reference_tj(current_esp_sum=233.0)  # Viking
        253.0
    """
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    max_batch_esp = 0.0
    for key in BUILDVALID_REFERENCE_KEYS:
        try:
            cm = get_canonical_model(key)
            s = float(sum(float(x) for x in cm.esp))
            if s > max_batch_esp:
                max_batch_esp = s
        except Exception:
            continue

    # Expande a janela se o modelo atual ultrapassa o batch de referência
    # (ex.: Viking Graben 10 com Σesp=233 m).
    effective_max = max(max_batch_esp, float(current_esp_sum))
    return round(effective_max + float(margin_m), 3)


def compute_canonical_h1(tj: float, sum_esp: float) -> float:
    """Calcula h1 centralizado conforme ``buildValidamodels.py``::

        h1 = (tj − Σesp) / 2

    Args:
        tj: Janela de investigação em metros.
        sum_esp: Σ(espessuras internas) do modelo canônico, em metros.

    Returns:
        ``h1`` em metros, com piso de 0.1 m para respeitar o range do
        ``QDoubleSpinBox`` da GUI. Arredondado a 3 casas decimais.
    """
    return max(0.1, round((float(tj) - float(sum_esp)) * 0.5, 3))


__all__ = [
    "BUILDVALID_MARGIN_M",
    "BUILDVALID_REFERENCE_KEYS",
    "CANONICAL_PROFILE_LABELS",
    "build_single_model_dict",
    "compute_canonical_h1",
    "compute_canonical_reference_tj",
    "format_profile_info",
    "get_profile_by_label",
]
