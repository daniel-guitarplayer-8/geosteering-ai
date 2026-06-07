# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/services/manual_geology.py                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ManualLayersModel — geologia manual N-camadas (PURO)       ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — services (spec 0015, Fatia 6b)                       ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — geologia avançada                               ║
# ║  Framework   : stdlib PURO — NÃO importa Qt (Princípio X)                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Modelo PURO de camadas manuais (N camadas arbitrárias) que o editor de  ║
# ║    camadas (View) edita e o ``_run_simulation`` consome (modo "manual").   ║
# ║    Também converte um perfil canônico (``simulation.validation``) em       ║
# ║    camadas manuais. Validação de fidelidade (ρᵥ≥ρₕ TIV, espessuras > 0).   ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ManualLayersModel                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ManualLayersModel`` — geologia manual N-camadas (PURO; spec 0015 Fatia 6b)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

__all__ = ["ManualLayersModel"]


@dataclass(frozen=True)
class ManualLayersModel:
    """Geologia manual de N camadas (2 semi-espaços + N−2 internas) — PURO/picklable.

    Convenção idêntica ao gerador estocástico e ao ``simulate_batch``:
    ``rho_h``/``rho_v`` têm ``n_layers`` valores; ``thicknesses`` tem ``n_layers−2``
    (só as camadas internas têm espessura finita; os semi-espaços não).

    Attributes:
        n_layers: nº total de camadas (≥ 2; inclui os 2 semi-espaços).
        thicknesses: espessuras internas (m), ``n_layers−2`` valores (> 0).
        rho_h: resistividades horizontais (Ω·m), ``n_layers`` valores (> 0).
        rho_v: resistividades verticais (Ω·m), ``n_layers`` valores (≥ ρₕ; TIV λ≥1).
    """

    n_layers: int
    thicknesses: Tuple[float, ...]
    rho_h: Tuple[float, ...]
    rho_v: Tuple[float, ...]

    def validate(self) -> List[str]:
        """Valida as invariantes físicas/estruturais. Retorna lista de erros (vazia = OK).

        Returns:
            Lista de mensagens (PT-BR). Checa: nº de camadas ≥ 2; tamanhos
            consistentes (ρₕ/ρᵥ = n_layers; espessuras = n_layers−2); positividade
            (ρ > 0, espessura > 0); anisotropia TIV física (ρᵥ ≥ ρₕ ⇔ λ ≥ 1).
        """
        errors: List[str] = []
        if self.n_layers < 2:
            errors.append(f"n_layers deve ser ≥ 2 (got {self.n_layers}).")
        if len(self.rho_h) != self.n_layers:
            errors.append(
                f"ρₕ deve ter {self.n_layers} valores (got {len(self.rho_h)})."
            )
        if len(self.rho_v) != self.n_layers:
            errors.append(
                f"ρᵥ deve ter {self.n_layers} valores (got {len(self.rho_v)})."
            )
        if len(self.thicknesses) != max(0, self.n_layers - 2):
            errors.append(
                f"espessuras devem ter {self.n_layers - 2} valores "
                f"(got {len(self.thicknesses)})."
            )
        if any(t <= 0.0 for t in self.thicknesses):
            errors.append("toda espessura interna deve ser > 0.")
        if any(r <= 0.0 for r in self.rho_h) or any(r <= 0.0 for r in self.rho_v):
            errors.append("toda resistividade (ρₕ/ρᵥ) deve ser > 0.")
        # TIV física: ρᵥ ≥ ρₕ (λ=√(ρᵥ/ρₕ) ≥ 1). Só checa onde os tamanhos casam.
        if len(self.rho_h) == len(self.rho_v):
            if any(rv < rh for rh, rv in zip(self.rho_h, self.rho_v)):
                errors.append("ρᵥ deve ser ≥ ρₕ em cada camada (TIV: λ ≥ 1).")
        return errors

    def to_model_dict(self) -> Dict[str, Any]:
        """Converte para o dict de modelo consumido por ``_simulate_grouped``.

        Returns:
            ``{"n_layers", "rho_h", "rho_v", "thicknesses"}`` (listas) — o mesmo
            formato dos modelos do gerador estocástico (agrupável por n_layers).
        """
        return {
            "n_layers": int(self.n_layers),
            "rho_h": list(self.rho_h),
            "rho_v": list(self.rho_v),
            "thicknesses": list(self.thicknesses),
        }

    @classmethod
    def from_canonical(cls, model: Any) -> "ManualLayersModel":
        """Constrói a partir de um ``CanonicalModel`` (simulation.validation).

        Args:
            model: ``CanonicalModel`` com ``n_layers``/``rho_h``/``rho_v``/``esp``
                (arrays numpy). Os valores são copiados para tuplas (picklable/imutável).

        Returns:
            ``ManualLayersModel`` equivalente ao perfil canônico.
        """
        return cls(
            n_layers=int(model.n_layers),
            thicknesses=tuple(float(x) for x in model.esp),
            rho_h=tuple(float(x) for x in model.rho_h),
            rho_v=tuple(float(x) for x in model.rho_v),
        )
