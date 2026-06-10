# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/_hwinfo.py                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Coleta de especificações de hardware para a CLI            ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.53 — tabela de resultados)              ║
# ║  Versão      : v2.53                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-02                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Framework   : stdlib (subprocess/os/platform) + probes opcionais         ║
# ║  Dependências: numba (opcional, p/ num_threads), psutil (opcional, RAM),  ║
# ║                nvidia-smi (opcional, GPU)                                  ║
# ║  Padrão      : cada probe em try/except — NUNCA levanta                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Coleta CPU (modelo + cores físicos/lógicos), RAM, threads Numba e      ║
# ║    (opcionalmente) GPU/VRAM/JAX devices para enriquecer a tabela de       ║
# ║    resultados da CLI. Todas as sondagens são defensivas: ferramentas      ║
# ║    ausentes (psutil/nvidia-smi) ou erros de leitura produzem              ║
# ║    ``"desconhecido"``/``None`` em vez de exceção.                         ║
# ║                                                                           ║
# ║  ESTRATÉGIA POR CAMPO (primeira fonte que funcionar)                      ║
# ║    ┌──────────────┬──────────────────────────────────────────────────┐   ║
# ║    │ cpu_model    │ /proc/cpuinfo → sysctl (macOS) → platform.proc()  │   ║
# ║    │ cores        │ _workers.detect_cpu_topology() (psutil→/proc→…)   │   ║
# ║    │ ram_gb       │ /proc/meminfo → psutil → None                     │   ║
# ║    │ numba_threads│ numba.get_num_threads() (try)                     │   ║
# ║    │ gpu/vram     │ nvidia-smi --query-gpu (só se want_gpu)           │   ║
# ║    │ jax_devices  │ jax.devices() (só se want_gpu)                    │   ║
# ║    └──────────────┴──────────────────────────────────────────────────┘   ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    collect_hardware_info: (*, want_gpu) → dict de specs de hardware       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Coleta defensiva de specs de hardware para a CLI (Sprint v2.53).

:func:`collect_hardware_info` retorna um dict com modelo de CPU, cores
físicos/lógicos, RAM, threads Numba e — quando ``want_gpu=True`` — nome/VRAM
da GPU (via ``nvidia-smi``) e devices JAX. Nenhuma sondagem levanta exceção:
fontes ausentes viram ``"desconhecido"`` ou ``None``.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["collect_hardware_info"]


def _cpu_model() -> str:
    """Retorna o nome comercial da CPU (``"desconhecido"`` se indisponível).

    Tenta, em ordem: ``/proc/cpuinfo`` (Linux), ``sysctl`` (macOS),
    ``platform.processor()`` (genérico). Cada fonte é defensiva.

    Returns:
        str: modelo da CPU, ou ``"desconhecido"`` se nenhuma fonte responder.
    """
    # Linux — /proc/cpuinfo "model name"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    # macOS — sysctl machdep.cpu.brand_string
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    # Genérico
    try:
        proc = platform.processor()
        if proc:
            return proc
    except Exception:  # noqa: BLE001 — platform nunca deveria falhar, mas seguro
        pass
    return "desconhecido"


def _ram_gb() -> Optional[float]:
    """Retorna a RAM física total em GB (``None`` se indisponível).

    Tenta ``/proc/meminfo`` (Linux) e depois ``psutil`` (opcional). Retorna
    ``None`` se ambas as fontes falharem.

    Returns:
        Optional[float]: RAM total em GiB, ou ``None``.
    """
    # Linux — /proc/meminfo MemTotal (kB)
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    # psutil (dependência [dev] — opcional)
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total) / (1024.0**3)
    except Exception:  # noqa: BLE001 — psutil ausente / erro de leitura
        return None


def _numba_threads() -> Optional[int]:
    """Retorna ``numba.get_num_threads()`` (``None`` se Numba ausente/erro)."""
    try:
        import numba  # type: ignore

        return int(numba.get_num_threads())
    except Exception:  # noqa: BLE001 — numba ausente ou não-inicializado
        return None


def _gpu_via_nvidia_smi() -> tuple[Optional[str], Optional[float]]:
    """Sonda nome + VRAM (GB) da GPU 0 via ``nvidia-smi`` (graceful).

    Returns:
        Tupla ``(gpu_name, gpu_vram_gb)``. ``(None, None)`` se ``nvidia-smi``
        ausente, sem GPU, ou em qualquer erro de parsing.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None, None
        # Primeira GPU (linha 0): "NVIDIA RTX A6000, 49140"
        first = out.stdout.strip().splitlines()[0]
        name, mem_mib = (p.strip() for p in first.split(","))
        vram_gb = float(mem_mib) / 1024.0  # MiB → GiB
        return name, vram_gb
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None


def _jax_devices() -> Optional[list[str]]:
    """Retorna a lista de devices JAX como strings (``None`` se indisponível)."""
    try:
        import jax  # type: ignore

        return [str(d) for d in jax.devices()]
    except Exception:  # noqa: BLE001 — jax ausente / sem backend
        return None


def collect_hardware_info(*, want_gpu: bool = False) -> dict:
    """Coleta um inventário defensivo de specs de hardware.

    Sondas de CPU/RAM/threads sempre rodam; sondas de GPU (``nvidia-smi`` +
    ``jax.devices()``) só rodam quando ``want_gpu=True`` (evita o overhead de
    subprocess/import JAX no caminho Numba puro). Nenhuma sondagem levanta.

    Args:
        want_gpu: Se ``True``, inclui ``gpu_name``/``gpu_vram_gb``/``jax_devices``
            (relevante para o backend JAX). Default ``False`` (caminho Numba).

    Returns:
        dict com as chaves:
          - ``cpu_model`` (str), ``cpu_physical`` (int|None),
            ``cpu_logical`` (int|None), ``ram_gb`` (float|None),
            ``numba_threads`` (int|None);
          - quando ``want_gpu``: ``gpu_name`` (str|None),
            ``gpu_vram_gb`` (float|None), ``jax_devices`` (list[str]|None).

    Note:
        ``cpu_physical``/``cpu_logical`` reusam
        :func:`geosteering_ai.simulation._workers.detect_cpu_topology`
        (psutil → sysctl → /proc → heurística; nunca falha).

    Example:
        >>> hw = collect_hardware_info(want_gpu=False)
        >>> set(["cpu_model", "cpu_logical", "ram_gb"]).issubset(hw)
        True
    """
    info: dict = {
        "cpu_model": _cpu_model(),
        "cpu_physical": None,
        "cpu_logical": None,
        "ram_gb": _ram_gb(),
        "numba_threads": _numba_threads(),
    }

    # Topologia de CPU (lógicas × físicas) — reuso defensivo do _workers.
    try:
        from geosteering_ai.simulation._workers import detect_cpu_topology

        logical, physical, _has_ht = detect_cpu_topology()
        info["cpu_logical"] = logical
        info["cpu_physical"] = physical
    except Exception:  # noqa: BLE001 — fallback se o módulo não importar
        import os

        info["cpu_logical"] = os.cpu_count()

    if want_gpu:
        gpu_name, gpu_vram = _gpu_via_nvidia_smi()
        info["gpu_name"] = gpu_name
        info["gpu_vram_gb"] = gpu_vram
        info["jax_devices"] = _jax_devices()

    return info
