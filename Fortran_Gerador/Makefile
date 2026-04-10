# SETTINGS VARIABLES
#--------------------------------------------------------------------
# Executable name (main compilation result)
binary = tatu.x
# Source files extension
src_ext = f08
# Object files extension
obj_ext = o
# Build directory path, where to put .$(obj_ext) and .mod files
build = ./build
# If not exist, create build directory
$(shell mkdir -p $(build))


# PROJECT DEPENDENCIES
#--------------------------------------------------------------------
# Dependencies
source_files = parameters filtersv2 utils magneticdipoles PerfilaAnisoOmp RunAnisoOmp
# Adds build path and object files extension to each one of source_files dependencies
object_files = $(patsubst %, $(build)/%.$(obj_ext), $(source_files))


# LINKER WORKAROUND (macOS 26+ / Darwin 25+ / x86_64)
#--------------------------------------------------------------------
# O linker padrão ld-1266.8 (CLT macOS 26) falha com o formato !tapi-tbd
# para x86_64. O ld-classic presente no CLT suporta x86_64 corretamente.
#
# ESTRATÉGIA: exportar COMPILER_PATH apontando para .ld_wrapper/ onde
# existe um script "ld" que delega para ld-classic.
#
# Por que COMPILER_PATH em vez de -B<dir>?
#   - -B<dir> seria relativo ao CWD, mas o f2py executa o link a partir
#     de um tmpdir (/var/tmp/...), tornando -B./build/ ineficaz nesse caso.
#   - COMPILER_PATH é uma variável de ambiente herdada por todos os
#     subprocessos (gfortran -> collect2 -> ld), funcionando corretamente
#     independente do tempdir onde o f2py opera.
#
# .ld_wrapper/ é um diretório permanente (não apagado por make clean),
# criado automaticamente via $(shell ...) abaixo na primeira execução.
#
# Em Linux ou quando ld-classic não existe, COMPILER_PATH fica vazio.
_LD_WRAPPER_DIR := $(abspath .ld_wrapper)
_LD_CLASSIC     := /Library/Developer/CommandLineTools/usr/bin/ld-classic

ifeq ($(shell uname -s),Darwin)
  ifneq ($(wildcard $(_LD_CLASSIC)),)
    $(shell mkdir -p $(_LD_WRAPPER_DIR) && \
            printf '#!/bin/sh\nexec $(_LD_CLASSIC) "$$@"\n' \
                   > $(_LD_WRAPPER_DIR)/ld && \
            chmod +x $(_LD_WRAPPER_DIR)/ld)
    export COMPILER_PATH := $(_LD_WRAPPER_DIR)
  endif
endif


# COMPILER SETTINGS
#--------------------------------------------------------------------
# Fortran Compiler
fc = gfortran
# fc = ifort

# Flags to Fortran Compiler
# -J specifies where to put .mod files for compiled modules

# Para o gfortran:
development_flags_gfortran = -J$(build) -std=f2008 -pedantic -Wall -Wextra -Wimplicit-interface -fPIC \
	-fmax-errors=1 -g -fcheck=all -fbacktrace -fall-intrinsics
production_flags_gfortran  = -J$(build) -std=f2008 -pedantic -Wall -Wextra -Wimplicit-interface -fPIC \
	-fmax-errors=1 -O3 -march=native -ffast-math -funroll-loops -fall-intrinsics
# Validação numérica: -O0 sem fast-math para garantir ausência de reordenamento
# associativo de ponto flutuante. Qualquer divergência de MD5 entre duas builds
# debug_O0 revela um problema matemático real (não artefato do compilador).
debug_O0_flags_gfortran = -J$(build) -std=f2008 -pedantic -Wall -Wextra -Wimplicit-interface -fPIC \
	-fmax-errors=1 -O0 -g -fno-fast-math -fsignaling-nans -fall-intrinsics

# Para o ifort:
# development_flags_ifort = -fPIC -fmax-errors=1 -g
# production_flags_ifort  = -fPIC -werror -fmax-errors=1 -O3 -march=native -ffast-math -funroll-loops -static-libgfortran

# flags = $(development_flags_gfortran)
flags = $(production_flags_gfortran)
# flags = $(development_flags_ifort)
# flags = $(production_flags_ifort)

# TARGETS
#--------------------------------------------------------------------
# Target to create executable binary
$(binary): $(object_files)
# $(@) represents the current target, in this case: $(@) = $(binary)
# $(^) represents all dependencies of the current target, in this case: all .$(obj_ext) files in $(build) directory
	$(fc) -fopenmp $(flags) -o $(@) $(^)

# Target to compile one dependency with different flags
#$(build)/Anderson.$(obj_ext): Anderson.for
#	$(fc) -J$(build) -std=legacy -c $(<) -o $(@)

# Target to compile all others dependencies
$(build)/%.$(obj_ext): %.$(src_ext)
# $(<) represents the first dependency of the current target, in this case: $(<) = %.$(src_ext)
	$(fc) -fopenmp $(flags) -c $(<) -o $(@)

# Target to run Python script
run_python: $(binary)
	python3 	fifthBuildTIVModels.py
# 	python3 	Models2OtherPeople.py
# 	python3 	buildValidamodels.py
# 	python3 	forthBuildTIVmodels.py
#   python3 	thirdBuildTIVmodels.py
# 	python3 	secondBuildTIVModels.py
#	python3 	buildTIVmodels.py

all: $(binary) run_python

# Target f2py: wrapper Python para o simulador EM 1D TIV.
# Gera módulo tatu_f2py.so importável em Python via "import tatu_f2py".
# Uso: make f2py_wrapper
# Pre-requisito: f2py (parte do NumPy) e compilador Fortran com OpenMP.
# Nota: requer compilação prévia dos objetos via "make $(binary)" para
#       gerar os .mod necessários.
#
# NOTA TÉCNICA 1 — Extensão .f08:
#   f2py não reconhece .f08 (apenas .f, .f90, .f95). Criamos um symlink
#   temporário .f90 -> .f08 antes da chamada e o removemos depois.
#
# NOTA TÉCNICA 2 — Flag -J:
#   f2py injeta -J<tmpdir> automaticamente. O gfortran aceita apenas um -J,
#   portanto F2PY_F90FLAGS não inclui -J$(build) (diferente de production_flags).
#
# NOTA TÉCNICA 3 — Linker macOS 26+:
#   COMPILER_PATH já é exportado no bloco do workaround acima, herdado por
#   todos os subprocessos incluindo o gfortran chamado internamente pelo f2py.
#   Não é necessário passar -B aqui.
#
# NOTA TÉCNICA 4 — .f2py_f2cmap:
#   Mapeia real(dp)->double e complex(dp)->complex_double. Como make clean
#   apaga build/, o arquivo é recriado dinamicamente neste target.
#
# NOTA TÉCNICA 5 — Detecção automática do interpretador Python:
#   O Homebrew python3 (3.14 em abril/2026) frequentemente vem SEM numpy,
#   portanto não é possível hardcodar "python3". A variável PYTHON prioriza:
#     1) ambiente (export PYTHON=...) — máxima precedência
#     2) python3.11 se numpy estiver instalado
#     3) python (tipicamente anaconda)
#     4) python3 (fallback)
#   Uso alternativo: make PYTHON=/path/custom/python f2py_wrapper
PYTHON ?= $(shell \
  for p in python3.11 python3.12 python3.13 python python3; do \
    if command -v $$p >/dev/null 2>&1 && $$p -c "import numpy.f2py" >/dev/null 2>&1; then \
      echo $$p; break; \
    fi; \
  done)

F2PY_F90FLAGS = -fopenmp -std=f2008 -pedantic -Wall -Wextra -Wimplicit-interface -fPIC \
	-fmax-errors=1 -O3 -march=native -ffast-math -funroll-loops -fall-intrinsics
f2py_wrapper: $(object_files)
	@if [ -z "$(PYTHON)" ]; then \
	  echo "ERRO: nenhum interpretador Python com numpy/f2py encontrado."; \
	  echo "      Instale numpy ou passe PYTHON=... explicitamente."; \
	  echo "      Exemplo: make PYTHON=/Users/daniel/anaconda3/bin/python f2py_wrapper"; \
	  exit 1; \
	fi
	@echo "==> Usando PYTHON = $(PYTHON) ($$($(PYTHON) --version 2>&1))"
	ln -sf ../tatu_f2py_wrapper.f08 $(build)/tatu_f2py_wrapper.f90
	@echo 'dict(real = dict(dp="double"), complex = dict(dp="complex_double"))' > $(build)/.f2py_f2cmap
	cd $(build) && $(PYTHON) -m numpy.f2py -c -m tatu_f2py tatu_f2py_wrapper.f90 \
	    parameters.$(obj_ext) filtersv2.$(obj_ext) utils.$(obj_ext) \
	    magneticdipoles.$(obj_ext) PerfilaAnisoOmp.$(obj_ext) \
	    --f90flags="$(F2PY_F90FLAGS)" \
	    -lgomp \
	    && mv tatu_f2py*.so ../
	rm -f $(build)/tatu_f2py_wrapper.f90

# Target de validação numérica: rebuild completo com -O0 -fno-fast-math.
# Uso: make clean && make debug_O0
# Produz binário determinístico bit-a-bit idêntico entre versões
# matematicamente equivalentes do código (sem reordenamento FP).
debug_O0:
	$(MAKE) clean
	$(MAKE) flags="$(debug_O0_flags_gfortran)" $(binary)

clean:
	rm -rf $(build)

cleanall:
	rm -rf $(build)
	rm -rf $(binary)
	rm -rf *.dat
	rm -rf *.out
