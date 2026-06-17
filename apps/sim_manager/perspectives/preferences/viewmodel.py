# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/preferences/viewmodel.py                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : PreferencesViewModel — estado PURO de preferências         ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app — perspectiva Preferências (Fatia 6e)               ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: gui.viewmodels.base (BaseViewModel), .service              ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    ViewModel PURO (Princípio X — NÃO importa Qt) das preferências do SM:  ║
# ║    tema, 4 paths, backend de plot e limites do cache LRU. Estado          ║
# ║    observável via ``VMSignal`` + serialização round-trip (to/from dict).  ║
# ║    Persistência delegada ao ``PreferencesService`` (injetado, duck-typed).║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    PreferencesViewModel                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``PreferencesViewModel`` — estado PURO das preferências (Fatia 6e)."""

from __future__ import annotations

from typing import Any, Dict

from geosteering_ai.gui.viewmodels.base import BaseViewModel
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["PreferencesViewModel"]

# Chaves de path expostas (paridade com load_paths()/save_paths() do monólito).
_PATH_KEYS = ("output_dir", "tatu_binary", "python_binary", "geosteering_ai")


def _safe_int(value: Any, fallback: int) -> int:
    """Converte ``value`` para ``int`` com fallback (NÃO levanta).

    Honra o contrato de degradação graciosa do load: um ``preferences.json``
    válido em JSON mas com um numérico de tipo errado (ex.: ``"512MB"``, ``null``)
    NÃO pode crashar o boot — cai para ``fallback`` (o valor-default corrente).

    Args:
        value: valor a converter (qualquer escalar JSON).
        fallback: valor de retorno se a conversão falhar.

    Returns:
        int: ``int(value)`` ou ``fallback`` se ``value`` não for coercível.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


class PreferencesViewModel(BaseViewModel):
    """ViewModel PURO das preferências do SM (tema, paths, backend, cache LRU).

    Attributes:
        theme_changed: ``VMSignal`` emit(theme: str) ao mudar o tema.
        plot_backend_changed: ``VMSignal`` emit(backend: str) ao mudar o backend.
        cache_changed: ``VMSignal`` emit(max_mb: int, max_snapshots: int).
        saved: ``VMSignal`` emit() após persistir com sucesso.

    Example:
        >>> vm = PreferencesViewModel(service=PreferencesService(path="/tmp/p.json"))
        >>> vm.cache_max_mb = 512        # dispara cache_changed
        >>> vm.save(); reloaded = PreferencesViewModel(service=vm._service)
        >>> reloaded.load(); reloaded.cache_max_mb
        512

    Note:
        NÃO importa Qt. A View liga ``changed``/sinais customizados a slots Qt.
        ``load`` atribui o estado DIRETO (sem ``_set``) para não disparar a cadeia
        de sinais durante a restauração — a View ressincroniza via
        :meth:`to_session_dict` após o load (evita laço View↔VM).
    """

    _STATE_FIELDS = (
        "_theme",
        "_plot_backend",
        "_cache_max_mb",
        "_cache_max_snapshots",
        "_paths",
    )

    def __init__(self, service: Any) -> None:
        """Inicializa com o serviço de persistência injetado.

        Args:
            service: objeto com ``defaults()``/``load()``/``save(dict)`` (duck-typed;
                tipicamente :class:`PreferencesService`).
        """
        super().__init__()
        self._service = service

        defaults = service.defaults()
        self._theme: str = defaults["theme"]
        self._plot_backend: str = defaults["plot_backend"]
        self._cache_max_mb: int = int(defaults["cache_max_mb"])
        self._cache_max_snapshots: int = int(defaults["cache_max_snapshots"])
        self._paths: Dict[str, str] = dict(defaults["paths"])

        self.theme_changed: VMSignal = VMSignal()
        self.plot_backend_changed: VMSignal = VMSignal()
        self.cache_changed: VMSignal = VMSignal()
        self.saved: VMSignal = VMSignal()

    # ── Properties + setters (cada um emite o sinal específico via _set) ─────
    @property
    def theme(self) -> str:
        """Nome do tema ativo (ex.: ``"antigravity_dark"``)."""
        return self._theme

    @theme.setter
    def theme(self, value: str) -> None:
        if self._set("_theme", value):
            self.theme_changed.emit(value)

    @property
    def plot_backend(self) -> str:
        """Backend de plot default (ex.: ``"matplotlib"``/``"pyqtgraph"``)."""
        return self._plot_backend

    @plot_backend.setter
    def plot_backend(self, value: str) -> None:
        if self._set("_plot_backend", value):
            self.plot_backend_changed.emit(value)

    @property
    def cache_max_mb(self) -> int:
        """Limite de bytes do cache LRU de plots, em MB (guardrail mín. 32)."""
        return self._cache_max_mb

    @cache_max_mb.setter
    def cache_max_mb(self, value: int) -> None:
        if self._set("_cache_max_mb", max(32, int(value))):
            self.cache_changed.emit(self._cache_max_mb, self._cache_max_snapshots)

    @property
    def cache_max_snapshots(self) -> int:
        """Limite de entradas (maxlen) do cache LRU (guardrail mín. 1)."""
        return self._cache_max_snapshots

    @cache_max_snapshots.setter
    def cache_max_snapshots(self, value: int) -> None:
        if self._set("_cache_max_snapshots", max(1, int(value))):
            self.cache_changed.emit(self._cache_max_mb, self._cache_max_snapshots)

    def get_path(self, key: str) -> str:
        """Retorna o path da chave (``""`` se ausente)."""
        return self._paths.get(key, "")

    def set_path(self, key: str, value: str) -> None:
        """Define o path da chave (emite ``changed`` se mudou)."""
        if key not in _PATH_KEYS:
            raise ValueError(f"path key inválida: {key!r}. Válidas: {_PATH_KEYS}")
        if self._paths.get(key) != value:
            new_paths = dict(self._paths)
            new_paths[key] = value
            # _equal usa == em dict (reduz a bool) → dedupe correto.
            self._set("_paths", new_paths)

    # ── Serialização (round-trip JSON, espelha SimulationViewModel) ──────────
    def to_session_dict(self) -> Dict[str, Any]:
        """Serializa o estado para um dict JSON-serializável (p/ o service)."""
        return {
            "theme": self._theme,
            "plot_backend": self._plot_backend,
            "cache_max_mb": self._cache_max_mb,
            "cache_max_snapshots": self._cache_max_snapshots,
            "paths": dict(self._paths),
        }

    def load_session_dict(self, data: Dict[str, Any]) -> None:
        """Restaura o estado de um dict (atribuição DIRETA — sem emitir sinais).

        Args:
            data: dict no formato de :meth:`to_session_dict` (chaves ausentes
                mantêm o valor atual). A View ressincroniza após o load.
        """
        self._theme = str(data.get("theme", self._theme))
        self._plot_backend = str(data.get("plot_backend", self._plot_backend))
        # _safe_int (não int direto): um JSON válido com numérico de tipo errado
        # (ex.: "512MB", null) cai para o default corrente em vez de crashar o boot
        # (contrato "nunca levanta no load" — a degradação graciosa é do VM aqui).
        self._cache_max_mb = max(
            32, _safe_int(data.get("cache_max_mb"), self._cache_max_mb)
        )
        self._cache_max_snapshots = max(
            1, _safe_int(data.get("cache_max_snapshots"), self._cache_max_snapshots)
        )
        stored_paths = data.get("paths")
        if isinstance(stored_paths, dict):
            self._paths = {
                k: str(stored_paths.get(k, self._paths.get(k, ""))) for k in _PATH_KEYS
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferencesViewModel":
        """NÃO suportado: este VM exige um ``service`` no construtor.

        O ``from_dict`` genérico da base presume ``cls()`` sem argumentos — aqui
        isso levantaria ``TypeError`` (``service`` é obrigatório). A persistência
        das Preferências usa :meth:`load`/:meth:`save` (via ``PreferencesService``)
        + o round-trip :meth:`to_session_dict`/:meth:`load_session_dict`::

            vm = PreferencesViewModel(service=PreferencesService())
            vm.load_session_dict(data)

        Args:
            data: ignorado.

        Raises:
            NotImplementedError: sempre — direciona ao caminho correto (evita o
                ``TypeError`` críptico do contrato genérico herdado).
        """
        raise NotImplementedError(
            "PreferencesViewModel.from_dict não é suportado (exige service); "
            "use PreferencesViewModel(service=...).load_session_dict(data)."
        )

    # ── Comandos (delegam ao service) ───────────────────────────────────────
    def load(self) -> None:
        """Carrega as preferências do serviço e restaura o estado (sem sinais)."""
        self.load_session_dict(self._service.load())

    def save(self) -> None:
        """Persiste o estado atual via serviço e emite ``saved``."""
        self._service.save(self.to_session_dict())
        self.saved.emit()

    def restore_defaults(self) -> None:
        """Restaura os valores-padrão (sem persistir; emite sinais por campo).

        ``cache_changed`` é emitido NO MÁXIMO UMA VEZ, com o par final consistente
        (mb, snapshots) — evita o transitório inconsistente que dois setters
        separados produziriam (1º emitiria ``(novo_mb, snapshots_antigo)``).
        """
        defaults = self._service.defaults()
        self.theme = defaults["theme"]
        self.plot_backend = defaults["plot_backend"]
        mb_changed = self._set("_cache_max_mb", max(32, int(defaults["cache_max_mb"])))
        snaps_changed = self._set(
            "_cache_max_snapshots", max(1, int(defaults["cache_max_snapshots"]))
        )
        if mb_changed or snaps_changed:
            self.cache_changed.emit(self._cache_max_mb, self._cache_max_snapshots)
        for key in _PATH_KEYS:
            self.set_path(key, defaults["paths"].get(key, ""))
