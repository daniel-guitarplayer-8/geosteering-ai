# -*- coding: utf-8 -*-
# Marcador explícito de pacote (PEP 8).
#
# Sem este arquivo, ``tests`` seria um *namespace package* implícito (PEP 420)
# — funciona localmente quando ``pythonpath = ["."]`` (pyproject.toml) adiciona
# a raiz ao ``sys.path``, mas falha em ambientes como Colab quando outro
# pacote pré-instalado expõe um diretório ``tests`` que sombreia o nosso,
# causando ``ModuleNotFoundError`` em ``from tests._fortran_helpers import ...``.
#
# Manter este arquivo VAZIO — não exportar nada de ``tests/`` é intencional.
