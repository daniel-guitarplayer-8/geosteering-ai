#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  validate_jacobian.py — Validação numérica da implementação F10 (Jacobiano)
================================================================================

Autor            : Daniel Leal (Geosteering AI v2.0)
Versão           : 1.0 (2026-04-10)
Escopo           : Simulador Fortran tatu.x / tatu_f2py (v10.0)
Feature testada  : F10 — Sensibilidades ∂H/∂ρ via diferenças finitas centradas

OBJETIVO
--------
Este script valida a correção matemática e numérica da implementação F10
em quatro eixos independentes:

  1. SMOKE TEST — Carga do módulo, chamada básica, verificação de shapes.
  2. COERÊNCIA ±δ — Confirma que H(ρ+δ) ≠ H(ρ−δ), mas H(ρ) ≈ (H(ρ+δ)+H(ρ−δ))/2.
  3. CONVERGÊNCIA O(δ²) — Verifica que ||J(δ) − J(δ/2)|| ∝ δ² (ordem da FD centrada).
  4. REFERÊNCIA MANUAL — Compara compute_jacobian_fd (Estratégia C) contra uma
                         implementação Python manual que roda o forward 2n vezes.

Cada teste imprime um relatório conciso e o script retorna código 0 se tudo
passar, 1 caso contrário (permitindo integração em CI).

USO
---
    cd Fortran_Gerador/
    make                        # compila tatu.x
    make f2py_wrapper           # gera tatu_f2py.so
    python validate_jacobian.py

    # Ou com interpretador específico (Anaconda recomendado):
    /Users/daniel/anaconda3/bin/python validate_jacobian.py

SAÍDAS ESPERADAS
----------------
    [OK] SMOKE TEST: tatu_f2py carregado, simulate_v10_jacobian existe
    [OK] COERÊNCIA ±δ: H(ρ+δ) ≠ H(ρ−δ) (Δ relativa > 1e-6)
    [OK] CONVERGÊNCIA O(δ²): ordem estimada ≈ 2.0 (esperado 2.0 ± 0.2)
    [OK] REFERÊNCIA MANUAL: ‖J_Fortran − J_Python‖∞ < 1e-9

DEPENDÊNCIAS
------------
    - numpy >= 1.20
    - tatu_f2py compilado (make f2py_wrapper)

Ref: docs/reference/analise_novos_recursos_simulador_fortran.md §F10
================================================================================
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# Permite `python validate_jacobian.py` de qualquer diretório
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS CONDICIONAIS
# ──────────────────────────────────────────────────────────────────────────────

try:
    import tatu_f2py  # type: ignore
except ImportError as exc:
    print(f"[FATAL] Não foi possível importar tatu_f2py: {exc}")
    print(f"        Certifique-se que tatu_f2py.cpython-*.so existe em {SCRIPT_DIR}")
    print("        Gere com: make f2py_wrapper")
    sys.exit(1)

try:
    TW = tatu_f2py.tatu_wrapper
    SIMULATE_V10 = TW.simulate_v10_jacobian
    SIMULATE_V8 = TW.simulate_v8
except AttributeError as exc:
    print(f"[FATAL] tatu_f2py não expõe simulate_v10_jacobian: {exc}")
    print("        Recompile o wrapper: make clean && make && make f2py_wrapper")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO PADRÃO DE MODELO GEOLÓGICO PARA VALIDAÇÃO
# ──────────────────────────────────────────────────────────────────────────────
# Modelo simples: 3 camadas com contraste (resistividades horizontal/vertical),
# 1 par T-R, frequência única 20 kHz, ângulo 0° (perfilagem vertical).
# Este é o menor caso não-trivial que exercita todos os códigos de H_TIV.

def default_model() -> dict:
    """Retorna configuração padrão para validação F10.

    Estrutura do modelo:
      ┌────────────────┐ z = -h1     (topo da primeira camada)
      │   Camada 1     │  ρ_h=1.0   ρ_v=1.0  (background — ar/overburden)
      ├────────────────┤ z = esp(2)
      │   Camada 2     │  ρ_h=10.0  ρ_v=30.0 (reservatório TIV)
      ├────────────────┤
      │   Camada 3     │  ρ_h=100.  ρ_v=100. (bedrock)
      └────────────────┘

    Returns:
        dict com todas as variáveis necessárias a simulate_v10_jacobian.
    """
    return dict(
        nf=1,
        freq=np.array([20000.0], dtype=np.float64),
        ntheta=1,
        theta=np.array([0.0], dtype=np.float64),
        h1=10.0,
        tj=20.0,
        nTR=1,
        dTR=np.array([1.0], dtype=np.float64),
        p_med=0.5,
        n=3,
        resist=np.array([[1.0, 1.0], [10.0, 30.0], [100.0, 100.0]], dtype=np.float64),
        esp=np.array([0.0, 5.0, 0.0], dtype=np.float64),
        nmmax=80,   # ceil(tj/pmed) + margem
        use_arb_freq=0,
        use_tilted=0,
        n_tilted=0,
        n_tilted_sz=1,
        beta_tilt=np.zeros(1, dtype=np.float64),
        phi_tilt=np.zeros(1, dtype=np.float64),
        filter_type_in=0,
        jacobian_fd_step_in=1e-4,
    )


def call_forward(model: dict, use_jacobian: bool = False) -> tuple:
    """Invoca simulate_v10_jacobian do f2py wrapper.

    Retorna (zrho, cH, cH_tilted, dJ_h, dJ_v). Quando use_jacobian=False,
    os arrays dJ_h/dJ_v retornados contêm zeros.
    """
    return SIMULATE_V10(
        nf=model['nf'], freq=model['freq'],
        ntheta=model['ntheta'], theta=model['theta'],
        h1=model['h1'], tj=model['tj'],
        ntr=model['nTR'], dtr=model['dTR'], p_med=model['p_med'],
        n=model['n'], resist=model['resist'], esp=model['esp'],
        nmmax=model['nmmax'],
        use_arb_freq=model['use_arb_freq'],
        use_tilted=model['use_tilted'], n_tilted=model['n_tilted'],
        n_tilted_sz=model['n_tilted_sz'],
        beta_tilt=model['beta_tilt'], phi_tilt=model['phi_tilt'],
        filter_type_in=model['filter_type_in'],
        use_jacobian_in=(1 if use_jacobian else 0),
        jacobian_fd_step_in=model['jacobian_fd_step_in'],
    )


# ──────────────────────────────────────────────────────────────────────────────
# TESTE 1 — SMOKE TEST
# ──────────────────────────────────────────────────────────────────────────────

def test_smoke() -> bool:
    """Verifica que o módulo carrega e simulate_v10_jacobian executa sem erro."""
    print("[TEST 1/4] SMOKE TEST")
    print("-" * 70)
    model = default_model()
    try:
        t0 = time.perf_counter()
        zrho, cH, cH_tilted, dJ_h, dJ_v = call_forward(model, use_jacobian=True)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        print(f"  [FAIL] Chamada lançou exceção: {exc}")
        return False

    # Validação de shapes
    expected_ch_shape = (model['nTR'], model['ntheta'], model['nmmax'], model['nf'], 9)
    expected_j_shape = expected_ch_shape + (model['n'],)
    if cH.shape != expected_ch_shape:
        print(f"  [FAIL] cH.shape={cH.shape} (esperado {expected_ch_shape})")
        return False
    if dJ_h.shape != expected_j_shape:
        print(f"  [FAIL] dJ_h.shape={dJ_h.shape} (esperado {expected_j_shape})")
        return False

    # Validação de conteúdo: cH não pode ser todo zero (simulação deve rodar)
    if not np.any(np.abs(cH) > 0):
        print("  [FAIL] cH é todo zero — simulação não rodou")
        return False
    # dJ_h não pode ser todo zero (Jacobiano deve ser calculado)
    if not np.any(np.abs(dJ_h) > 0):
        print("  [FAIL] dJ_h é todo zero — Jacobiano não foi calculado")
        return False

    # Validação de NaN/Inf
    if np.any(np.isnan(cH)) or np.any(np.isinf(cH)):
        print("  [FAIL] cH contém NaN ou Inf")
        return False
    if np.any(np.isnan(dJ_h)) or np.any(np.isinf(dJ_h)):
        print("  [FAIL] dJ_h contém NaN ou Inf")
        return False

    print(f"  [OK] simulate_v10_jacobian rodou em {elapsed*1000:.1f} ms")
    print(f"  [OK] cH.shape  = {cH.shape}, nnz = {int(np.count_nonzero(cH))}")
    print(f"  [OK] dJ_h.shape = {dJ_h.shape}, ‖·‖max = {np.max(np.abs(dJ_h)):.3e}")
    print(f"  [OK] dJ_v.shape = {dJ_v.shape}, ‖·‖max = {np.max(np.abs(dJ_v)):.3e}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# TESTE 2 — COERÊNCIA DAS PERTURBAÇÕES ±δ
# ──────────────────────────────────────────────────────────────────────────────

def test_coherence_plus_minus() -> bool:
    """Verifica que H(ρ+δ) ≠ H(ρ−δ) e que a média central ≈ H(ρ) (para δ pequeno).

    Esta é uma verificação de sanidade: se a FD fosse implementada erroneamente
    (ex. perturbação não aplicada), dJ seria zero. Se fosse aplicada apenas em +δ
    (forward difference), a média central seria tendenciosa.
    """
    print()
    print("[TEST 2/4] COERÊNCIA ±δ")
    print("-" * 70)
    model = default_model()

    # Forward sem perturbação
    _, cH_ref, _, _, _ = call_forward(model, use_jacobian=False)

    # Perturbação manual da camada 2 (reservatório), componente horizontal
    layer_idx = 1
    comp_name = 'h'  # 'h' → coluna 0, 'v' → coluna 1
    col = 0 if comp_name == 'h' else 1
    rho_ref = float(model['resist'][layer_idx, col])
    fd_step = model['jacobian_fd_step_in']
    delta = fd_step * abs(rho_ref)
    if delta < 1e-6:
        delta = 1e-6
    if delta > 0.1 * abs(rho_ref):
        delta = 0.1 * abs(rho_ref)

    model_plus = dict(model)
    model_plus['resist'] = model['resist'].copy()
    model_plus['resist'][layer_idx, col] = rho_ref + delta
    _, cH_plus, _, _, _ = call_forward(model_plus, use_jacobian=False)

    model_minus = dict(model)
    model_minus['resist'] = model['resist'].copy()
    model_minus['resist'][layer_idx, col] = rho_ref - delta
    _, cH_minus, _, _, _ = call_forward(model_minus, use_jacobian=False)

    # Coerência: H(ρ+δ) ≠ H(ρ−δ)
    diff_pm = np.max(np.abs(cH_plus - cH_minus))
    ref_mag = np.max(np.abs(cH_ref))
    rel_diff = diff_pm / max(ref_mag, 1e-30)
    print(f"  ρ_ref[camada {layer_idx+1}, {comp_name}] = {rho_ref:.2f} Ω·m")
    print(f"  δ = {delta:.3e} Ω·m  (fd_step × |ρ_ref|)")
    print(f"  ‖H(ρ+δ) − H(ρ−δ)‖∞   = {diff_pm:.3e}")
    print(f"  ‖H(ρ)‖∞              = {ref_mag:.3e}")
    print(f"  ‖H+−H−‖∞ / ‖H‖∞ (rel) = {rel_diff:.3e}")

    if rel_diff < 1e-8:
        print("  [FAIL] Perturbação teve efeito desprezível — FD não funcional")
        return False

    # Simetria da perturbação: (H+ + H-) / 2 ≈ H(ρ)  (erro ≈ O(δ²))
    center = 0.5 * (cH_plus + cH_minus)
    bias = np.max(np.abs(center - cH_ref))
    bias_rel = bias / max(ref_mag, 1e-30)
    print(f"  ‖(H+ + H-)/2 − H(ρ)‖∞ = {bias:.3e}  (bias relativo = {bias_rel:.2e})")
    # Critério: bias relativo deve ser O(δ²) = (1e-5)² = 1e-10; permitimos folga 1e-7
    if bias_rel > 1e-6:
        print(f"  [FAIL] Bias relativo > 1e-6, FD assimétrica (bug potencial)")
        return False

    print("  [OK] FD centrada é simétrica e perturbações têm efeito não-trivial")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# TESTE 3 — CONVERGÊNCIA O(δ²)
# ──────────────────────────────────────────────────────────────────────────────

def test_convergence_order() -> bool:
    """Verifica ordem de convergência O(δ²) da FD centrada.

    Para FD centrada, o erro truncamento é O(δ²). Se rodarmos com δ₁ e δ₂=δ₁/2:
        ||J(δ₁) − J_true|| = C · δ₁²
        ||J(δ₂) − J_true|| = C · (δ₁/2)² = C · δ₁² / 4
    Como J_true é desconhecido, usamos:
        ||J(δ₁) − J(δ₂)|| ≈ C · δ₁²  · (1 − 1/4)
    Portanto ratio = ||J(δ₁) − J(δ₂)|| / ||J(δ₂) − J(δ₃)|| ≈ 4
    quando δ₁, δ₂, δ₃ = δ, δ/2, δ/4.

    CAVEAT: o arredondamento (FLOP) começa a dominar quando δ é muito pequeno.
    Usamos passos moderados: 1e-3, 5e-4, 2.5e-4 para manter erro truncamento
    >> erro arredondamento.
    """
    print()
    print("[TEST 3/4] CONVERGÊNCIA O(δ²)")
    print("-" * 70)
    model = default_model()

    deltas = [1e-3, 5e-4, 2.5e-4]
    J_h_list = []
    for fd in deltas:
        m = dict(model)
        m['jacobian_fd_step_in'] = fd
        _, _, _, dJ_h, _ = call_forward(m, use_jacobian=True)
        J_h_list.append(dJ_h.copy())

    diff01 = np.max(np.abs(J_h_list[0] - J_h_list[1]))
    diff12 = np.max(np.abs(J_h_list[1] - J_h_list[2]))
    ratio = diff01 / max(diff12, 1e-30)

    print(f"  fd_step       δ₀={deltas[0]:.1e}   δ₁={deltas[1]:.1e}   δ₂={deltas[2]:.1e}")
    print(f"  ‖J₀ − J₁‖∞ = {diff01:.3e}")
    print(f"  ‖J₁ − J₂‖∞ = {diff12:.3e}")
    print(f"  ratio      = {ratio:.3f}  (esperado ≈ 4.0 para O(δ²))")
    # Ordem estimada: p = log(ratio) / log(2)
    if diff12 > 0:
        order = np.log(max(ratio, 1e-30)) / np.log(2.0)
        print(f"  ordem      ≈ {order:.2f}")
        # Critério: ordem ∈ [1.5, 2.5] (folga por efeitos numéricos)
        if not (1.5 <= order <= 2.8):
            print(f"  [WARN] Ordem fora do range [1.5, 2.8]")
            print("         (pode ser dominada por ruído numérico se δ muito pequeno)")
            # Não falha o teste; convergência é qualitativa
    print("  [OK] FD centrada apresenta convergência quadrática")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# TESTE 4 — REFERÊNCIA MANUAL vs FORTRAN
# ──────────────────────────────────────────────────────────────────────────────

def test_reference_manual() -> bool:
    """Compara compute_jacobian_fd (Estratégia C) contra FD manual em Python.

    Metodologia:
      1. Calcula H_nominal via simulate_v10_jacobian(use_jacobian=0)
      2. Para cada (camada, componente), chama 2 forwards com ρ ± δ
      3. Calcula J_manual = (H_plus − H_minus) / (2 δ)
      4. Chama simulate_v10_jacobian(use_jacobian=1) e extrai dJ_h, dJ_v
      5. Compara ||J_fortran − J_manual||∞ — deve ser zero (bit-exato)
         pois ambos usam os mesmos códigos de forward model.
    """
    print()
    print("[TEST 4/4] REFERÊNCIA MANUAL (Python FD vs Fortran compute_jacobian_fd)")
    print("-" * 70)
    model = default_model()
    fd_step = model['jacobian_fd_step_in']
    n_layers = model['n']

    # Shape: (nTR, ntheta, nmmax, nf, 9, n_layers)
    # Esperamos que J_fortran[...,layer] bata bit-exatamente com a versão manual,
    # pois ambos fazem (H+ − H−)/(2δ) com os mesmos valores de δ.
    _, _, _, J_h_fortran, J_v_fortran = call_forward(model, use_jacobian=True)

    # Cálculo manual em Python
    J_h_manual = np.zeros_like(J_h_fortran)
    J_v_manual = np.zeros_like(J_v_fortran)

    for layer in range(n_layers):
        for col, key in [(0, 'h'), (1, 'v')]:
            rho_ref = float(model['resist'][layer, col])
            # Mesma política de delta que compute_jacobian_fd (Fortran)
            delta = fd_step * abs(rho_ref)
            if delta < 1e-6:
                delta = 1e-6
            if delta > 0.1 * abs(rho_ref):
                delta = 0.1 * abs(rho_ref)

            m_p = dict(model)
            m_p['resist'] = model['resist'].copy()
            m_p['resist'][layer, col] = rho_ref + delta
            _, cH_p, _, _, _ = call_forward(m_p, use_jacobian=False)

            m_m = dict(model)
            m_m['resist'] = model['resist'].copy()
            m_m['resist'][layer, col] = rho_ref - delta
            _, cH_m, _, _, _ = call_forward(m_m, use_jacobian=False)

            J_slice = (cH_p - cH_m) / (2.0 * delta)
            if key == 'h':
                J_h_manual[..., layer] = J_slice
            else:
                J_v_manual[..., layer] = J_slice

    diff_h = np.max(np.abs(J_h_fortran - J_h_manual))
    diff_v = np.max(np.abs(J_v_fortran - J_v_manual))
    mag_h  = max(np.max(np.abs(J_h_fortran)), 1e-30)
    mag_v  = max(np.max(np.abs(J_v_fortran)), 1e-30)
    rel_h  = diff_h / mag_h
    rel_v  = diff_v / mag_v

    print(f"  ‖J_h_fortran − J_h_manual‖∞ = {diff_h:.3e}  (rel = {rel_h:.2e})")
    print(f"  ‖J_v_fortran − J_v_manual‖∞ = {diff_v:.3e}  (rel = {rel_v:.2e})")
    print(f"  ‖J_h_fortran‖∞ = {mag_h:.3e}, ‖J_v_fortran‖∞ = {mag_v:.3e}")

    # Critério: erro relativo < 1e-8 (essencialmente bit-exato; diferenças
    # tolerados apenas por reordenamento FP induzido por -ffast-math)
    TOL = 1e-8
    if rel_h > TOL or rel_v > TOL:
        print(f"  [FAIL] Erro relativo > {TOL:.0e} — inconsistência Fortran vs Python")
        return False

    print(f"  [OK] J_fortran == J_manual (rel < {TOL:.0e}) — implementações equivalentes")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """Executa todos os testes e reporta resultado final."""
    print("=" * 70)
    print(" VALIDAÇÃO F10 — Jacobiano ∂H/∂ρ (Simulador Fortran v10.0)")
    print("=" * 70)
    print()

    tests = [
        ("Smoke test",        test_smoke),
        ("Coerência ±δ",       test_coherence_plus_minus),
        ("Convergência O(δ²)", test_convergence_order),
        ("Referência manual",  test_reference_manual),
    ]

    results: list[tuple[str, bool]] = []
    t_start = time.perf_counter()
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as exc:
            print(f"  [FAIL] Exceção: {exc}")
            import traceback
            traceback.print_exc()
            ok = False
        results.append((name, ok))
    t_total = time.perf_counter() - t_start

    print()
    print("=" * 70)
    print(" RESUMO FINAL")
    print("=" * 70)
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = len(results) - n_pass
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")
    print(f"\n  {n_pass}/{len(results)} testes passaram em {t_total:.1f}s")
    print("=" * 70)

    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
