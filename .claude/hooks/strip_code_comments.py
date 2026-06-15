#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove comentários e literais de string (incl. docstrings) de código Python.

Usado pelo hook ``check-anti-patterns-precommit.sh`` para que os padrões de
anti-pattern (ex.: ``globals().get(``, ``print(``, ``TARGET_SCALING = "log"``)
só casem CÓDIGO REAL — nunca o mesmo texto aparecendo num COMENTÁRIO ou
DOCSTRING (que dispararia falso-positivo, como o header D1 "NUNCA
globals().get()" disparava KB-GLB).

Estratégia: tokeniza o fonte e BRANQUEIA (substitui por espaços) os tokens de
``COMMENT`` e ``STRING`` preservando linhas/colunas — o resultado tem a MESMA
estrutura de código, só sem o texto de comentários/strings. Lê o arquivo em
``argv[1]`` e escreve o resultado em stdout.

Fail-open: se o arquivo tiver erro de sintaxe (``tokenize`` falha), devolve o
conteúdo ORIGINAL — preferimos um eventual falso-positivo a DEIXAR PASSAR uma
violação real (arquivos com erro de sintaxe falham noutros gates de qualquer modo).
"""

from __future__ import annotations

import sys
import tokenize


def strip(path: str) -> str:
    """Devolve o fonte com comentários + strings branqueados (estrutura preservada)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    out = list(lines)
    try:
        with open(path, "rb") as fb:
            for tok in tokenize.tokenize(fb.readline):
                if tok.type in (tokenize.COMMENT, tokenize.STRING):
                    (sr, sc), (er, ec) = tok.start, tok.end
                    for r in range(sr, er + 1):
                        if r - 1 >= len(out):
                            continue
                        line = out[r - 1]
                        cs = sc if r == sr else 0
                        ce = ec if r == er else len(line)
                        out[r - 1] = line[:cs] + " " * (ce - cs) + line[ce:]
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return "".join(lines)  # fail-open: preserva detecção
    return "".join(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(0)
    sys.stdout.write(strip(sys.argv[1]))
