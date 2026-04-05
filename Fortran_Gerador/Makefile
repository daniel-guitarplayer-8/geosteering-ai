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
# ld-1266.8 (CLT macOS 26) falha com !tapi-tbd para x86_64.
# ld-classic suporta x86_64 corretamente e está presente no CLT.
# Solução: criar build/ld wrapper → ld-classic, e usar -B$(build)/ para
# que o collect2 do gfortran encontre o wrapper antes do ld do sistema.
# Em Linux ou quando ld-classic não existe, LDFLAGS_EXTRA fica vazio.
ifeq ($(shell uname -s),Darwin)
  _LD_CLASSIC := /Library/Developer/CommandLineTools/usr/bin/ld-classic
  ifneq ($(wildcard $(_LD_CLASSIC)),)
    $(shell printf '#!/bin/sh\nexec $(_LD_CLASSIC) "$$@"\n' > $(build)/ld && chmod +x $(build)/ld)
    LDFLAGS_EXTRA := -B$(build)/
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
	$(fc) -fopenmp $(flags) $(LDFLAGS_EXTRA) -o $(@) $(^)

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
