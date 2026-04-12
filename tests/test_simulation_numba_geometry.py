# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_numba_geometry.py                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Testes Geometria (Sprint 2.3)          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Framework   : pytest 7.x + numpy 2.x                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de _numba/geometry.py (Sprint 2.3).

Baterias:
- TestSanitizeProfile (5): shapes, valores, casos degenerados.
- TestFindLayersTR (7): TX e RX em várias posições relativas.
- TestLayerAtDepth (4): camada para profundidade arbitrária.
"""
from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation._numba.geometry import (
    find_layers_tr,
    layer_at_depth,
    sanitize_profile,
)


# ──────────────────────────────────────────────────────────────────────────────
# TestSanitizeProfile
# ──────────────────────────────────────────────────────────────────────────────
class TestSanitizeProfile:
    """sanitize_profile produz h e prof corretos."""

    def test_3_layers_1_internal(self):
        """n=3, 1 camada interna → h=[0, 5, 0], prof=[-1e300, 0, 5, 1e300]."""
        esp = np.array([5.0])
        h, prof = sanitize_profile(n=3, esp=esp)
        assert h.shape == (3,)
        assert prof.shape == (4,)
        np.testing.assert_allclose(h, [0.0, 5.0, 0.0])
        assert prof[0] == -1.0e300  # sentinel topo (paridade Fortran)
        np.testing.assert_allclose(prof[1:3], [0.0, 5.0])
        assert prof[3] == 1.0e300  # sentinel fundo

    def test_5_layers_3_internal(self):
        """n=5, 3 camadas internas."""
        esp = np.array([1.5, 2.0, 1.0])
        h, prof = sanitize_profile(n=5, esp=esp)
        assert h.shape == (5,)
        assert prof.shape == (6,)
        np.testing.assert_allclose(h, [0.0, 1.5, 2.0, 1.0, 0.0])
        # prof[0]=-1e300, prof[1]=0, prof[2]=1.5, prof[3]=3.5, prof[4]=4.5, prof[5]=1e300
        assert prof[0] == -1.0e300
        np.testing.assert_allclose(prof[1:5], [0.0, 1.5, 3.5, 4.5])
        assert prof[5] == 1.0e300

    def test_2_layers_no_internal(self):
        """n=2, esp vazio → h=[0,0], prof=[-1e300, 0, 1e300]."""
        esp = np.zeros(0, dtype=np.float64)
        h, prof = sanitize_profile(n=2, esp=esp)
        assert h.shape == (2,)
        assert prof.shape == (3,)
        np.testing.assert_allclose(h, [0.0, 0.0])
        assert prof[0] == -1.0e300
        assert prof[2] == 1.0e300

    def test_invalid_n_raises(self):
        """n<2 levanta ValueError."""
        with pytest.raises(ValueError, match="n=1"):
            sanitize_profile(n=1, esp=np.array([1.0]))

    def test_invalid_esp_shape_raises(self):
        """esp com shape errado levanta ValueError."""
        with pytest.raises(ValueError, match="esp.shape"):
            sanitize_profile(n=5, esp=np.array([1.0, 2.0]))


# ──────────────────────────────────────────────────────────────────────────────
# TestFindLayersTR
# ──────────────────────────────────────────────────────────────────────────────
class TestFindLayersTR:
    """find_layers_tr retorna camadas corretas para TX/RX."""

    def _prof_3_layers(self):
        """Perfil 3 camadas: topo + 5m + base."""
        return np.array([0.0, 0.0, 5.0, 1.0e300], dtype=np.float64)

    def test_tx_above_interface_rx_middle(self):
        """TX no ar (h0<0), RX no meio da camada 1."""
        prof = self._prof_3_layers()
        ct, cr = find_layers_tr(n=3, h0=-1.0, z=2.5, prof=prof)
        assert ct == 0
        assert cr == 1

    def test_tx_and_rx_same_middle_layer(self):
        """TX e RX ambos na camada 1."""
        prof = self._prof_3_layers()
        ct, cr = find_layers_tr(n=3, h0=2.5, z=2.5, prof=prof)
        assert ct == 1
        assert cr == 1

    def test_tx_middle_rx_below(self):
        """TX na camada 1, RX na base (camada 2)."""
        prof = self._prof_3_layers()
        ct, cr = find_layers_tr(n=3, h0=2.5, z=7.0, prof=prof)
        assert ct == 1
        assert cr == 2

    def test_tx_base_rx_base(self):
        """Ambos na camada 2 (base)."""
        prof = self._prof_3_layers()
        ct, cr = find_layers_tr(n=3, h0=100.0, z=100.0, prof=prof)
        assert ct == 2
        assert cr == 2

    def test_tx_on_interface_uses_above(self):
        """TX exatamente em prof[1]=0 usa camada *acima* (0)."""
        prof = self._prof_3_layers()
        # h0 == 0 → h0 > prof[1]=0 é falso → camad_t = 0
        ct, _ = find_layers_tr(n=3, h0=0.0, z=0.0, prof=prof)
        assert ct == 0

    def test_rx_at_interface_uses_below(self):
        """RX exatamente em prof[2]=5 usa camada *abaixo* (2)."""
        prof = self._prof_3_layers()
        # z == 5 → z >= prof[2]=5 é verdadeiro mas z < prof[n-1]=5 é falso
        # A lógica é: z >= prof[n-1] → camada n-1
        #             senão percorre n-2..1; primeira com z >= prof[i] ganha
        # prof[2]=5, z=5 → z >= prof[n-1]=prof[2]=5 → camad_r = n-1 = 2
        _, cr = find_layers_tr(n=3, h0=0.0, z=5.0, prof=prof)
        assert cr == 2

    def test_5_layers_tx_middle(self):
        """Perfil 5 camadas, TX na camada 2 (segunda interna)."""
        h, prof = sanitize_profile(n=5, esp=np.array([2.0, 3.0, 1.0]))
        # prof = [0, 0, 2, 5, 6, 1e300]
        # TX em z=3.5 → entre prof[2]=2 e prof[3]=5 → camada 2
        ct, cr = find_layers_tr(n=5, h0=3.5, z=3.5, prof=prof)
        assert ct == 2
        assert cr == 2


# ──────────────────────────────────────────────────────────────────────────────
# TestLayerAtDepth
# ──────────────────────────────────────────────────────────────────────────────
class TestLayerAtDepth:
    """layer_at_depth retorna a camada 0-based correta."""

    def test_4_layers_various_depths(self):
        """Perfil 4 camadas com várias profundidades de teste."""
        h, prof = sanitize_profile(n=4, esp=np.array([5.0, 5.0]))
        # prof = [0, 0, 5, 10, 1e300]
        assert layer_at_depth(4, -1.0, prof) == 0  # acima da 1ª interface
        assert layer_at_depth(4, 2.5, prof) == 1  # meio camada 1
        assert layer_at_depth(4, 7.5, prof) == 2  # meio camada 2
        assert layer_at_depth(4, 100.0, prof) == 3  # profundo

    def test_z_negative_stays_at_topo(self):
        """z negativo (no ar) fica na camada 0 (topo)."""
        h, prof = sanitize_profile(n=3, esp=np.array([5.0]))
        # z < 0 não satisfaz nenhuma condição do loop (todos prof[i] >= 0),
        # então retorna camad_default = 0 (paridade Fortran:
        # "camad=1 when z=0 is preferable", i.e., default inicial).
        assert layer_at_depth(3, -1.0, prof) == 0

    def test_z_exactly_at_interface_uses_below(self):
        """z == prof[i] vai para a camada *abaixo* da interface."""
        h, prof = sanitize_profile(n=3, esp=np.array([5.0]))
        # prof = [0, 0, 5, 1e300]
        # z=5 → z >= prof[n-1]=prof[2]=5 → camada n-1=2
        assert layer_at_depth(3, 5.0, prof) == 2

    def test_2_layers_half_space(self):
        """2 camadas (apenas semi-espaços) — z>0 sempre na camada 1."""
        h, prof = sanitize_profile(n=2, esp=np.zeros(0, dtype=np.float64))
        # prof = [0, 0, 1e300]
        # z >= prof[n-1]=prof[1]=0 → camad = n-1 = 1
        assert layer_at_depth(2, 10.0, prof) == 1
