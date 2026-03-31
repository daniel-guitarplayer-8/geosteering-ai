#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MCP SERVER: Consensus Scientific Paper Search                             ║
# ║  Fase: B — MCP Server Dedicado (Opção A — Padrão)                         ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Integração com Pesquisa Científica                 ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: Model Context Protocol (MCP) via mcp-python-sdk               ║
# ║                                                                            ║
# ║  Propósito:                                                                ║
# ║    • Buscar artigos científicos via Semantic Scholar API                    ║
# ║    • Buscar preprints via ArXiv API                                        ║
# ║    • Classificar relevância para inversão EM / geosteering / DL            ║
# ║    • Cache local de resultados em docs/reference/papers/                   ║
# ║    • Integração nativa com Claude Code via MCP                             ║
# ║                                                                            ║
# ║  Dependências: mcp (pip install mcp), httpx                                ║
# ║  Exports: MCP tools — search_papers, get_paper_details,                    ║
# ║           search_arxiv, list_cached_papers                                 ║
# ║  Ref: docs/reference/consensus_integration.md                             ║
# ║                                                                            ║
# ║  Histórico:                                                                ║
# ║    v1.0.0 (2026-03) — Implementação inicial                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""MCP Server para busca de artigos científicos — Geosteering AI v2.0.

Provê ferramentas de pesquisa científica como capacidades nativas do
Claude Code via Model Context Protocol (MCP). Três fontes de dados:

    1. Semantic Scholar API (acesso aberto, sem API key obrigatória)
    2. ArXiv API (acesso aberto, XML)

Nota: Integração com a Consensus API (https://consensus.app) está planejada
para uma versão futura. Requer CONSENSUS_API_KEY.

Uso via Claude Code:
    O servidor é registrado em .claude/settings.json e expõe ferramentas
    que o Claude Code pode invocar diretamente durante a conversa.

Instalação:
    pip install mcp httpx

Execução standalone (debug):
    python tools/consensus-mcp-server/server.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger("consensus-mcp")

# ── Constantes ────────────────────────────────────────────────────────────

# Semantic Scholar API — acesso aberto (100 req/5 min sem key)
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

# ArXiv API — acesso aberto (sem rate limit estrito)
ARXIV_API = "https://export.arxiv.org/api/query"

# Consensus API — planejada para versão futura (requer CONSENSUS_API_KEY)
# CONSENSUS_API = "https://api.consensus.app/v1"  # TODO(v1.1): implementar

# Campos retornados pelo Semantic Scholar
S2_FIELDS = (
    "title,abstract,year,citationCount,authors,externalIds,"
    "url,openAccessPdf,fieldsOfStudy,publicationTypes"
)

# Categorias ArXiv relevantes ao Geosteering AI
ARXIV_CATEGORIES = {
    "physics.geo-ph": "Geofísica",
    "cs.LG": "Machine Learning",
    "eess.SP": "Processamento de Sinais",
    "physics.comp-ph": "Física Computacional",
    "cs.NE": "Redes Neurais",
    "cs.AI": "Inteligência Artificial",
    "stat.ML": "Estatística/ML",
}

# Palavras-chave para scoring de relevância ao projeto
RELEVANCE_KEYWORDS = {
    # Score 5 — match direto ao projeto
    5: [
        "electromagnetic inversion",
        "resistivity inversion",
        "LWD",
        "logging while drilling",
        "geosteering",
        "1D inversion",
        "triaxial electromagnetic",
    ],
    # Score 4 — PINNs para geofísica
    4: [
        "physics-informed neural network",
        "PINN geophysic",
        "Helmholtz equation",
        "anisotropic resistivity",
        "TIV",
        "transversely isotropic",
    ],
    # Score 3 — inversão geofísica genérica
    3: [
        "geophysical inversion",
        "deep learning inversion",
        "sequence to sequence",
        "well logging",
        "formation evaluation",
    ],
    # Score 2 — ML para geociências
    2: [
        "geoscience",
        "subsurface",
        "petrophysic",
        "borehole",
        "downhole",
    ],
}

# Diretório de cache local (resolvido a partir da raiz do projeto)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "docs" / "reference" / "papers"


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO: FUNÇÕES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════
# Funções utilitárias para scoring, formatação e cache de resultados.
# ──────────────────────────────────────────────────────────────────────────


def compute_relevance_score(title: str, abstract: str) -> int:
    """Calcula score de relevância (1-5) para o Geosteering AI v2.0.

    Combina título e abstract para determinar quão relevante o artigo
    é para inversão EM 1D, geosteering, PINNs e deep learning aplicado.

    Args:
        title: Título do artigo.
        abstract: Resumo do artigo (pode ser string vazia).

    Returns:
        int: Score de 1 (genérico) a 5 (match direto ao projeto).

    Note:
        Keywords hierárquicas: score mais alto prevalece.
        Ref: docs/reference/consensus_integration.md seção de scoring.
    """
    text = f"{title} {abstract}".lower()
    best_score = 1
    for score, keywords in RELEVANCE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                best_score = max(best_score, score)
    return best_score


def format_authors(authors: list[dict]) -> str:
    """Formata lista de autores para exibição compacta.

    Args:
        authors: Lista de dicts com campo 'name'.

    Returns:
        str: Autores formatados (ex: "Silva, J.; Leal, D. et al.").
    """
    if not authors:
        return "N/A"
    names = [a.get("name", "?") for a in authors[:3]]
    result = "; ".join(names)
    if len(authors) > 3:
        result += " et al."
    return result


def slugify(text: str) -> str:
    """Converte texto em slug seguro para nome de arquivo.

    Args:
        text: Texto a converter.

    Returns:
        str: Slug com apenas letras minúsculas, números e hífens.
    """
    slug = re.sub(r"[^a-z0-9\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug[:80].strip("-")


def save_to_cache(query: str, results: list[dict]) -> str:
    """Salva resultados de pesquisa no cache local.

    Args:
        query: Query original da busca.
        results: Lista de artigos encontrados.

    Returns:
        str: Caminho do arquivo salvo.

    Note:
        Cache em docs/reference/papers/<slug>.json.
        Criação do diretório é automática.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(query)
    filepath = CACHE_DIR / f"{slug}.json"
    payload = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "count": len(results),
        "results": results,
    }
    filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Cache salvo: %s (%d resultados)", filepath, len(results))
    return str(filepath)


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO: SEMANTIC SCHOLAR API
# ════════════════════════════════════════════════════════════════════════════
# Interface com a API do Semantic Scholar para busca de artigos.
# Rate limit: 100 requisições por 5 minutos (sem API key).
# Com API key (S2_API_KEY): 1 req/segundo = ~300 req/5 min.
# Ref: https://api.semanticscholar.org/
# ──────────────────────────────────────────────────────────────────────────


async def search_semantic_scholar(
    query: str,
    limit: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
    fields_of_study: str | None = None,
) -> list[dict]:
    """Busca artigos no Semantic Scholar.

    Args:
        query: Texto de busca (ex: "PINN electromagnetic inversion").
        limit: Número máximo de resultados (1-100). Default: 10.
        year_min: Ano mínimo de publicação. Default: None (sem filtro).
        year_max: Ano máximo de publicação. Default: None (sem filtro).
        fields_of_study: Campos de estudo (ex: "Geology,Computer Science").

    Returns:
        list[dict]: Lista de artigos com título, abstract, ano, citações, etc.

    Raises:
        httpx.HTTPStatusError: Se a API retornar erro HTTP.

    Note:
        Rate limit: 100 req/5 min sem key, 1 req/s com S2_API_KEY.
        Ref: https://api.semanticscholar.org/api-docs/
    """
    import httpx

    params: dict[str, Any] = {
        "query": query,
        "limit": min(limit, 100),
        "fields": S2_FIELDS,
    }

    if year_min is not None or year_max is not None:
        yr_min = str(year_min) if year_min is not None else ""
        yr_max = str(year_max) if year_max is not None else ""
        params["year"] = f"{yr_min}-{yr_max}"

    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study

    headers = {}
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{SEMANTIC_SCHOLAR_API}/paper/search",
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    papers = data.get("data", [])
    results = []
    for p in papers:
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        results.append(
            {
                "title": title,
                "authors": format_authors(p.get("authors", [])),
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "abstract": (abstract or "")[:500],
                "doi": (p.get("externalIds") or {}).get("DOI", ""),
                "arxiv_id": (p.get("externalIds") or {}).get("ArXiv", ""),
                "url": p.get("url", ""),
                "pdf_url": (p.get("openAccessPdf") or {}).get("url", ""),
                "fields": p.get("fieldsOfStudy", []),
                "relevance_score": compute_relevance_score(title, abstract or ""),
                "source": "semantic_scholar",
            }
        )

    results.sort(key=lambda x: (-x["relevance_score"], -x["citations"]))
    return results


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO: ARXIV API
# ════════════════════════════════════════════════════════════════════════════
# Interface com a API do ArXiv para busca de preprints.
# Acesso aberto, sem API key. Retorna XML Atom.
# Ref: https://info.arxiv.org/help/api/
# ──────────────────────────────────────────────────────────────────────────


async def search_arxiv(
    query: str,
    limit: int = 10,
    category: str | None = None,
) -> list[dict]:
    """Busca preprints no ArXiv.

    Args:
        query: Texto de busca.
        limit: Número máximo de resultados (1-50). Default: 10.
        category: Categoria ArXiv para filtrar (ex: "physics.geo-ph").
            Se None, busca em todas as categorias.

    Returns:
        list[dict]: Lista de preprints com título, autores, ano, etc.

    Note:
        Categorias relevantes: physics.geo-ph, cs.LG, eess.SP,
        physics.comp-ph, cs.NE, cs.AI, stat.ML.
        Ref: https://info.arxiv.org/help/api/
    """
    import httpx

    # ── Construir query ArXiv ─────────────────────────────────────────
    search_query = f"all:{query}"
    if category:
        search_query = f"cat:{category} AND all:{query}"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": min(limit, 50),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(ARXIV_API, params=params)
        resp.raise_for_status()
        xml_text = resp.text

    # ── Parse XML Atom ────────────────────────────────────────────────
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", ns)

    results = []
    for entry in entries:
        title = (entry.findtext("atom:title", "", ns) or "").strip()
        title = re.sub(r"\s+", " ", title)

        abstract = (entry.findtext("atom:summary", "", ns) or "").strip()
        abstract = re.sub(r"\s+", " ", abstract)[:500]

        authors = []
        for author_el in entry.findall("atom:author", ns):
            name = author_el.findtext("atom:name", "", ns)
            if name:
                authors.append({"name": name})

        published = entry.findtext("atom:published", "", ns)
        year = int(published[:4]) if published and len(published) >= 4 else None

        # ── Extrair ArXiv ID da URL ──────────────────────────────────
        arxiv_id = ""
        for link_el in entry.findall("atom:link", ns):
            href = link_el.get("href", "")
            if "arxiv.org/abs/" in href:
                arxiv_id = href.split("/abs/")[-1]
                break

        # ── Extrair PDF URL ──────────────────────────────────────────
        pdf_url = ""
        for link_el in entry.findall("atom:link", ns):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href", "")
                break

        # ── Extrair categorias ───────────────────────────────────────
        categories = []
        for cat_el in entry.findall(
            "{http://arxiv.org/schemas/atom}primary_category"
        ):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)

        results.append(
            {
                "title": title,
                "authors": format_authors(authors),
                "year": year,
                "citations": 0,  # ArXiv não fornece citações
                "abstract": abstract,
                "doi": "",
                "arxiv_id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "pdf_url": pdf_url,
                "fields": categories,
                "relevance_score": compute_relevance_score(title, abstract),
                "source": "arxiv",
            }
        )

    results.sort(key=lambda x: -x["relevance_score"])
    return results


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO: MCP SERVER
# ════════════════════════════════════════════════════════════════════════════
# Registro das ferramentas MCP que serão expostas ao Claude Code.
# Cada ferramenta é uma função assíncrona decorada com @server.tool().
#
# Ferramentas expostas:
#   1. search_papers — busca multi-fonte (Semantic Scholar + ArXiv)
#   2. get_paper_details — detalhes de um paper por DOI ou ArXiv ID
#   3. search_arxiv_papers — busca exclusiva no ArXiv com filtro de categoria
#   4. list_cached_papers — lista papers salvos no cache local
#
# Ref: https://modelcontextprotocol.io/docs
# ──────────────────────────────────────────────────────────────────────────


def create_server():
    """Cria e configura o servidor MCP com todas as ferramentas.

    Returns:
        FastMCP: Servidor MCP configurado e pronto para execução.

    Note:
        Requer: pip install mcp httpx
        Execução: python tools/consensus-mcp-server/server.py
        API: Usa FastMCP (mcp.server.fastmcp) para decorador @server.tool().
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("consensus-scientific-search")

    # ── Tool 1: search_papers ─────────────────────────────────────────
    @server.tool()
    async def search_papers(
        query: str,
        limit: int = 10,
        year_min: int | None = None,
        year_max: int | None = None,
        save_cache: bool = True,
    ) -> str:
        """Busca artigos científicos em múltiplas fontes.

        Combina resultados do Semantic Scholar e ArXiv, classifica por
        relevância ao Geosteering AI v2.0 (inversão EM, PINNs, DL),
        e opcionalmente salva no cache local.

        Args:
            query: Texto de busca científica.
                Exemplos relevantes ao projeto:
                - "LWD electromagnetic inversion deep learning"
                - "physics-informed neural network geophysics"
                - "geosteering real-time resistivity"
            limit: Número máximo de resultados por fonte (1-50).
            year_min: Ano mínimo de publicação (ex: 2020).
            year_max: Ano máximo de publicação (ex: 2026).
            save_cache: Se True, salva resultados em docs/reference/papers/.

        Returns:
            str: Resultados formatados em Markdown.
        """
        # ── Buscar em paralelo nas duas fontes ────────────────────────
        import asyncio

        s2_task = search_semantic_scholar(
            query, limit=limit, year_min=year_min, year_max=year_max
        )
        arxiv_task = search_arxiv(query, limit=limit)

        s2_results, arxiv_results = await asyncio.gather(
            s2_task, arxiv_task, return_exceptions=True
        )

        # ── Tratar exceções ───────────────────────────────────────────
        all_results = []
        if isinstance(s2_results, list):
            all_results.extend(s2_results)
        else:
            logger.warning("Semantic Scholar falhou: %s", s2_results)

        if isinstance(arxiv_results, list):
            all_results.extend(arxiv_results)
        else:
            logger.warning("ArXiv falhou: %s", arxiv_results)

        # ── Deduplificar por título (fuzzy) ───────────────────────────
        seen_titles: set[str] = set()
        unique_results = []
        for r in all_results:
            title_key = r["title"].lower().strip()[:120]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_results.append(r)

        unique_results.sort(
            key=lambda x: (-x["relevance_score"], -x["citations"])
        )

        # ── Salvar cache ──────────────────────────────────────────────
        if save_cache and unique_results:
            save_to_cache(query, unique_results)

        # ── Formatar saída Markdown ───────────────────────────────────
        lines = [
            f"## Resultados da Pesquisa: \"{query}\"",
            f"**Total**: {len(unique_results)} artigos encontrados",
            f"**Fontes**: Semantic Scholar + ArXiv",
            f"**Data**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "| # | Score | Título | Autores | Ano | Cit. | Fonte |",
            "|:-:|:-----:|:-------|:--------|:---:|:----:|:-----:|",
        ]

        for i, r in enumerate(unique_results[:20], 1):
            score_display = f"{r['relevance_score']}/5"
            # Escapar pipes em títulos/autores para não quebrar tabela Markdown
            title_safe = r["title"][:80].replace("|", "\\|")
            authors_safe = r["authors"][:40].replace("|", "\\|")
            lines.append(
                f"| {i} | {score_display} | {title_safe} | "
                f"{authors_safe} | {r['year'] or '?'} | "
                f"{r['citations']} | {r['source']} |"
            )

        # ── Adicionar abstracts dos top 5 ────────────────────────────
        lines.extend(["", "### Top 5 — Resumos", ""])
        for i, r in enumerate(unique_results[:5], 1):
            lines.append(f"**{i}. {r['title']}**")
            if r["abstract"]:
                lines.append(f"> {r['abstract']}")
            if r["doi"]:
                lines.append(f"DOI: {r['doi']}")
            if r["pdf_url"]:
                lines.append(f"PDF: {r['pdf_url']}")
            lines.append("")

        return "\n".join(lines)

    # ── Tool 2: get_paper_details ─────────────────────────────────────
    @server.tool()
    async def get_paper_details(paper_id: str) -> str:
        """Obtém detalhes completos de um artigo por DOI ou ArXiv ID.

        Args:
            paper_id: Identificador do artigo. Aceita:
                - DOI (ex: "10.1093/gji/ggaf101")
                - ArXiv ID (ex: "2210.09060")
                - Semantic Scholar ID (ex: "S2:12345678")

        Returns:
            str: Detalhes completos formatados em Markdown.
        """
        import httpx

        # ── Validar paper_id (segurança: evitar path traversal) ───────
        if not re.match(r"^[\w./:()@-]+$", paper_id):
            return f"ID inválido: {paper_id}. Aceita DOI, ArXiv ID ou S2 ID."

        # ── Determinar tipo de ID ─────────────────────────────────────
        if paper_id.startswith("10."):
            s2_id = f"DOI:{paper_id}"
        elif re.match(r"^\d{4}\.\d{4,5}", paper_id):
            s2_id = f"ArXiv:{paper_id}"
        else:
            s2_id = paper_id

        headers = {}
        api_key = os.environ.get("S2_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        # ── Buscar com tratamento de erros ────────────────────────────
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{SEMANTIC_SCHOLAR_API}/paper/{s2_id}",
                    params={"fields": S2_FIELDS + ",references,citations"},
                    headers=headers,
                )
                resp.raise_for_status()
                p = resp.json()
        except httpx.HTTPStatusError as e:
            return (
                f"Erro HTTP {e.response.status_code} ao buscar '{paper_id}'. "
                "Paper não encontrado ou rate-limited."
            )
        except httpx.RequestError as e:
            return f"Erro de rede ao buscar '{paper_id}': {e}"

        title = p.get("title", "N/A")
        abstract = p.get("abstract", "N/A")
        authors = format_authors(p.get("authors", []))
        year = p.get("year", "?")
        citations = p.get("citationCount", 0)
        doi = (p.get("externalIds") or {}).get("DOI", "")
        pdf_url = (p.get("openAccessPdf") or {}).get("url", "")
        score = compute_relevance_score(title, abstract or "")

        lines = [
            f"## {title}",
            f"**Autores**: {authors}",
            f"**Ano**: {year} | **Citações**: {citations}",
            f"**DOI**: {doi}" if doi else "",
            f"**PDF**: {pdf_url}" if pdf_url else "",
            "",
            "### Resumo",
            abstract or "(resumo não disponível)",
            "",
            f"### Relevância para Geosteering AI: {score}/5",
        ]

        return "\n".join(lines)

    # ── Tool 3: search_arxiv_papers ───────────────────────────────────
    @server.tool()
    async def search_arxiv_papers(
        query: str,
        category: str = "",
        limit: int = 15,
    ) -> str:
        """Busca exclusiva no ArXiv com filtro de categoria.

        Ideal para preprints recentes e artigos com texto completo
        disponível gratuitamente.

        Args:
            query: Texto de busca.
            category: Categoria ArXiv para filtrar. Opções relevantes:
                - "physics.geo-ph" — Geofísica
                - "cs.LG" — Machine Learning
                - "eess.SP" — Processamento de Sinais
                - "physics.comp-ph" — Física Computacional
                - "" — Todas as categorias (default)
            limit: Número máximo de resultados (1-50).

        Returns:
            str: Resultados formatados com links para PDF.
        """
        try:
            results = await search_arxiv(
                query,
                limit=limit,
                category=category or None,
            )
        except Exception as e:
            return f"Erro ao buscar no ArXiv: {e}"

        lines = [
            f"## ArXiv: \"{query}\"",
            f"**Categoria**: {category or 'todas'}",
            f"**Resultados**: {len(results)}",
            "",
        ]

        for i, r in enumerate(results, 1):
            lines.append(f"### {i}. {r['title']}")
            lines.append(f"**Autores**: {r['authors']} ({r['year'] or '?'})")
            lines.append(f"**Score**: {r['relevance_score']}/5")
            if r["pdf_url"]:
                lines.append(f"**PDF**: {r['pdf_url']}")
            if r["abstract"]:
                lines.append(f"> {r['abstract'][:300]}...")
            lines.append("")

        return "\n".join(lines)

    # ── Tool 4: list_cached_papers ────────────────────────────────────
    @server.tool()
    async def list_cached_papers() -> str:
        """Lista artigos salvos no cache local (docs/reference/papers/).

        Returns:
            str: Inventário do cache com queries e contagens.
        """
        if not CACHE_DIR.exists():
            return "Cache vazio. Nenhuma pesquisa salva ainda."

        files = sorted(CACHE_DIR.glob("*.json"))
        if not files:
            return "Cache vazio. Nenhuma pesquisa salva ainda."

        lines = [
            "## Cache Local de Pesquisas Científicas",
            f"**Diretório**: {CACHE_DIR}",
            f"**Total**: {len(files)} pesquisas salvas",
            "",
            "| # | Arquivo | Query | Resultados | Data |",
            "|:-:|:--------|:------|:----------:|:----:|",
        ]

        for i, f in enumerate(files, 1):
            try:
                data = json.loads(f.read_text())
                query = data.get("query", "?")[:50]
                count = data.get("count", 0)
                ts = data.get("timestamp", "?")[:10]
                lines.append(
                    f"| {i} | {f.name} | {query} | {count} | {ts} |"
                )
            except (json.JSONDecodeError, KeyError):
                lines.append(f"| {i} | {f.name} | (erro) | ? | ? |")

        return "\n".join(lines)

    return server


# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO: PONTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════════
# Execução do servidor MCP via stdio (padrão para Claude Code).
# ──────────────────────────────────────────────────────────────────────────


def main():
    """Ponto de entrada principal — executa o servidor MCP via stdio.

    FastMCP gerencia o event loop e o transporte stdio internamente.
    """
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
