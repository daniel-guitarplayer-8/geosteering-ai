# -*- coding: utf-8 -*-
"""Perspectiva **Preferências** do SM MVVM (Fatia 6e).

Exporta a perspectiva plugável (``PreferencesPerspective``), o ViewModel PURO
(``PreferencesViewModel``) e o serviço de persistência (``PreferencesService``).
A View Qt (``PreferencesPanel``) é importada localmente em ``build_view`` para
manter o ViewModel testável sem Qt (Princípio X).
"""

from __future__ import annotations

from apps.sim_manager.perspectives.preferences.perspective import (
    PreferencesPerspective,
)
from apps.sim_manager.perspectives.preferences.service import PreferencesService
from apps.sim_manager.perspectives.preferences.viewmodel import PreferencesViewModel

__all__ = [
    "PreferencesPerspective",
    "PreferencesViewModel",
    "PreferencesService",
]
