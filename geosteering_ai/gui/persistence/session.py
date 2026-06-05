# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/persistence/session.py                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SessionDocument — estado volátil de UI (.session)          ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — persistência (spec 0007)                            ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação                                        ║
# ║  Framework   : stdlib PURO (json, dataclasses) — NÃO importa Qt           ║
# ║  Dependências: gui.persistence.atomic                                     ║
# ║  Padrão      : Documento de persistência (≠ ViewModel) — JSON, sem pickle  ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Documento ``.session`` (JSON) que guarda o ESTADO VOLÁTIL de UI do SM   ║
# ║    e do Studio — perspectiva ativa, parâmetros, layout, backend de plot   ║
# ║    selecionado, snapshots em cache. Salva de forma ATÔMICA (crash-safe).  ║
# ║    Base do ``.gsproj`` (projeto durável, spec 0018), que o EMBUTE.        ║
# ║                                                                           ║
# ║  INVARIANTES                                                              ║
# ║    • PROIBIDO pickle — só JSON (segurança: pickle = RCE em load).         ║
# ║    • Forward-compat: chaves desconhecidas (top-level e dentro de ``data``) ║
# ║      são PRESERVADAS no round-trip (um Studio novo grava; um SM velho lê   ║
# ║      e re-grava sem perder campos futuros).                               ║
# ║    • PURO (sem Qt) → um ViewModel (0005) serializa-se PARA um             ║
# ║      SessionDocument; o documento em si é testável sem ``pytest-qt``.     ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SessionDocument                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SessionDocument`` — estado volátil de UI em ``.session`` (JSON atômico, sem pickle)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from geosteering_ai.gui.persistence.atomic import atomic_write_text

__all__ = ["SessionDocument"]

# Chaves reservadas do envelope — o resto é tratado como extensão futura.
_ENVELOPE_KEYS = ("schema_version", "data")


@dataclass
class SessionDocument:
    """Documento ``.session`` — estado volátil de UI (perspectiva, params, layout…).

    Attributes:
        schema_version: versão do esquema do envelope (``1`` na fundação).
        data: dicionário ABERTO com o estado de UI (JSON-serializável). Campos
            novos vivem aqui — preservados no round-trip por ser um dict opaco.
        extra: chaves TOP-LEVEL desconhecidas lidas de um ``.session`` futuro,
            preservadas para re-emissão (forward-compat). Não use diretamente.

    Example:
        >>> doc = SessionDocument(data={"perspective": "simulation"})
        >>> doc.save("/tmp/app.session")          # escrita atômica
        >>> back = SessionDocument.load("/tmp/app.session")
        >>> back.data["perspective"]
        'simulation'

    Note:
        Serialização é JSON puro (``ensure_ascii=False`` p/ acentuação PT-BR).
        NUNCA usa pickle (Princípio de segurança — RCE em desserialização).
    """

    schema_version: int = 1
    data: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Valida o invariante de construção: ``data``/``extra`` são dicts.

        Falha-rápido (na construção, não só no ``to_json``) se alguém atribuir
        ``SessionDocument(data=42)`` — pega o erro do programador cedo.

        Raises:
            TypeError: se ``data`` ou ``extra`` não forem ``dict``.
        """
        if not isinstance(self.data, dict):
            raise TypeError(
                f"SessionDocument.data deve ser dict, obtido {type(self.data).__name__}."
            )
        if not isinstance(self.extra, dict):
            raise TypeError(
                f"SessionDocument.extra deve ser dict, obtido {type(self.extra).__name__}."
            )

    def to_json(self) -> str:
        """Serializa o documento para JSON (envelope + extras futuros).

        Returns:
            String JSON: ``{"schema_version": N, "data": {...}, <extras>}``.

        Raises:
            TypeError: se ``data``/``extra`` contiverem valores não-serializáveis
                em JSON (ex.: ``set``, ``datetime``, objetos custom) — fail-fast
                do ``json.dumps`` (a responsabilidade de manter o estado
                JSON-serializável é de quem o popula).
        """
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "data": self.data,
        }
        # Re-emite chaves top-level futuras preservadas (forward-compat).
        payload.update(self.extra)
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "SessionDocument":
        """Reconstrói um ``SessionDocument`` de JSON (forward-compat).

        Args:
            text: conteúdo JSON produzido por :meth:`to_json` (ou versão futura).

        Returns:
            Instância com ``schema_version``/``data`` restaurados; chaves
            top-level desconhecidas vão para ``extra`` (preservadas).

        Raises:
            ValueError: se o JSON não for um objeto (dict) no topo.
        """
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(
                f"`.session` inválido: esperado objeto JSON no topo, obtido {type(obj).__name__}."
            )
        schema_version = obj.get("schema_version", 1)
        data = obj.get("data", {})
        if not isinstance(data, dict):
            raise ValueError(
                f"`.session` inválido: 'data' deve ser objeto JSON (dict), "
                f"obtido {type(data).__name__}."
            )
        # Chaves top-level fora do envelope conhecido → preservadas (forward-compat).
        extra = {k: v for k, v in obj.items() if k not in _ENVELOPE_KEYS}
        return cls(schema_version=schema_version, data=data, extra=extra)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Salva o documento em ``path`` de forma ATÔMICA (crash-safe).

        Args:
            path: caminho do ``.session`` (diretório pai criado se necessário).
        """
        atomic_write_text(path, self.to_json())

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "SessionDocument":
        """Carrega um ``.session`` do disco.

        Args:
            path: caminho do arquivo ``.session``.

        Returns:
            O :class:`SessionDocument` reconstruído (forward-compat).
        """
        with open(os.fspath(path), encoding="utf-8") as handle:
            return cls.from_json(handle.read())
