#!/usr/bin/env bash
# =============================================================================
# run_bench.sh — Benchmark reprodutível do simulador Fortran PerfilaAnisoOmp
# =============================================================================
# Executa tatu.x em loop controlado, mede wall-time via /usr/bin/time, gera
# estatísticas (média, desvio-padrão, min, max) e calcula MD5 da saída binária
# para validação numérica posterior.
#
# Uso:
#   bash bench/run_bench.sh [--label NAME] [--iters N] [--threads T] [--keep]
#
# Opções:
#   --label NAME    Rótulo do relatório (ex: baseline, phase1). Default: baseline
#   --iters N       Número de invocações de tatu.x (default 30)
#   --threads T     OMP_NUM_THREADS para o benchmark (default 8 — ótimo i9-9980HK)
#   --keep          Preserva o .dat final da última iteração (usado para md5/validação)
#   --build-flags F Flags adicionais de compilação (ex: "-fopt-info-vec=...")
#
# Plataformas: macOS (Homebrew gfortran) e Linux (gfortran distro).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORTRAN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

# ---------- parse args ----------
LABEL="baseline"
ITERS=30
THREADS=8
KEEP_OUTPUT=0
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)       LABEL="$2"; shift 2 ;;
    --iters)       ITERS="$2"; shift 2 ;;
    --threads)     THREADS="$2"; shift 2 ;;
    --keep)        KEEP_OUTPUT=1; shift ;;
    --build-flags) EXTRA_FLAGS="$2"; shift 2 ;;
    -h|--help)     sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "Opção desconhecida: $1" >&2; exit 1 ;;
  esac
done

# ---------- detectar SO ----------
OS_NAME="$(uname -s)"
case "$OS_NAME" in
  Darwin)
    CORES_PHYS=$(sysctl -n hw.physicalcpu)
    CORES_LOG=$(sysctl -n hw.logicalcpu)
    CPU_BRAND=$(sysctl -n machdep.cpu.brand_string)
    TIME_CMD="/usr/bin/time -p"
    ;;
  Linux)
    CORES_PHYS=$(grep -c '^processor' /proc/cpuinfo || nproc)
    CORES_LOG="$CORES_PHYS"
    CPU_BRAND=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')
    TIME_CMD="/usr/bin/time -p"
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    ;;
  *)
    echo "SO não suportado: $OS_NAME" >&2; exit 1 ;;
esac

export OMP_NUM_THREADS="$THREADS"

# ---------- compilação ----------
echo "==> Compilando tatu.x ($LABEL) em $OS_NAME com $CORES_PHYS cores físicos..."
cd "$FORTRAN_DIR"
make clean >/dev/null 2>&1 || true
rm -f tatu.x
if [[ -n "$EXTRA_FLAGS" ]]; then
  # passa flags extras via variável de ambiente que o Makefile não vê — fallback: rebuild manual
  rm -rf build; mkdir -p build
  FFLAGS="-J./build -std=f2008 -pedantic -Wall -Wextra -Wimplicit-interface -fPIC \
-fmax-errors=1 -O3 -march=native -ffast-math -funroll-loops -fall-intrinsics -fopenmp $EXTRA_FLAGS"
  for src in parameters filtersv2 utils magneticdipoles PerfilaAnisoOmp RunAnisoOmp; do
    gfortran $FFLAGS -c "${src}.f08" -o "build/${src}.o"
  done
  gfortran -fopenmp -o tatu.x build/parameters.o build/filtersv2.o build/utils.o \
                                 build/magneticdipoles.o build/PerfilaAnisoOmp.o build/RunAnisoOmp.o
else
  make >/dev/null
fi

if [[ ! -x ./tatu.x ]]; then
  echo "ERRO: tatu.x não gerado" >&2; exit 2
fi

# ---------- ambiente isolado ----------
WORK="$(mktemp -d -t tatu_bench_XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
cp tatu.x "$WORK/"
cp model.in "$WORK/"
cd "$WORK"

# extrair nome do arquivo de saída do model.in (linha 10)
OUT_NAME=$(sed -n '10p' model.in | awk '{print $1}')
OUT_DAT="${OUT_NAME}.dat"
OUT_OUT="info${OUT_NAME}.out"

# ---------- loop de benchmark ----------
echo "==> Executando $ITERS iterações com OMP_NUM_THREADS=$THREADS..."
TIMES_FILE="$RESULTS_DIR/${LABEL}_times_raw.txt"
: > "$TIMES_FILE"

for i in $(seq 1 "$ITERS"); do
  rm -f "$OUT_DAT" "$OUT_OUT"
  # /usr/bin/time -p imprime 3 linhas em stderr: real X / user Y / sys Z
  REAL=$({ $TIME_CMD ./tatu.x; } 2>&1 1>/dev/null | awk '/^real/ {print $2}')
  echo "$REAL" >> "$TIMES_FILE"
  printf '  iter %02d: %s s\n' "$i" "$REAL"
done

# ---------- estatísticas ----------
STATS=$(python3 - "$TIMES_FILE" <<'PY'
import sys, statistics, math
with open(sys.argv[1]) as f:
    vals = [float(x.replace(',', '.')) for x in f if x.strip()]
n = len(vals)
mean = statistics.fmean(vals)
stdev = statistics.stdev(vals) if n > 1 else 0.0
vmin = min(vals); vmax = max(vals); median = statistics.median(vals)
throughput = 3600.0 / mean
print(f"{n}|{mean:.4f}|{stdev:.4f}|{vmin:.4f}|{vmax:.4f}|{median:.4f}|{throughput:.1f}")
PY
)
IFS='|' read -r N MEAN STDEV VMIN VMAX MEDIAN THROUGHPUT <<< "$STATS"

# ---------- md5 do output ----------
if command -v md5sum >/dev/null 2>&1; then
  MD5=$(md5sum "$OUT_DAT" | awk '{print $1}')
else
  MD5=$(md5 -q "$OUT_DAT")
fi
OUT_SIZE=$(wc -c < "$OUT_DAT" | tr -d ' ')

# preservar saída para validação numérica se solicitado
if [[ "$KEEP_OUTPUT" -eq 1 ]]; then
  cp "$OUT_DAT" "$RESULTS_DIR/${LABEL}_output.dat"
  cp "$OUT_OUT" "$RESULTS_DIR/${LABEL}_output.info"
fi

# ---------- relatório ----------
REPORT="$RESULTS_DIR/${LABEL}_report.md"
GFORTRAN_VER=$(gfortran --version | head -1)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > "$REPORT" <<EOF
# Relatório de Benchmark — ${LABEL}

**Gerado em**: $TIMESTAMP
**Script**: \`bench/run_bench.sh\`

## Ambiente

| Campo                | Valor                                   |
|:---------------------|:----------------------------------------|
| Sistema operacional  | $OS_NAME                                |
| CPU                  | $CPU_BRAND                              |
| Núcleos físicos      | $CORES_PHYS                             |
| Núcleos lógicos      | $CORES_LOG                              |
| Compilador           | $GFORTRAN_VER                           |
| \`OMP_NUM_THREADS\`  | $THREADS                                |
| Flags extras         | ${EXTRA_FLAGS:-"(nenhuma)"}             |

## Configuração do Modelo (\`model.in\`)

- Frequências: 2 (20 kHz, 40 kHz)
- Ângulo: 1 (0°)
- Camadas: 10
- Filtro Hankel: 201 pontos (Werthmüller J0/J1)
- Medidas por modelo: ~600 (janela 120 m, passo 0,2 m)

## Resultados

| Métrica                        | Valor                   |
|:-------------------------------|:------------------------|
| Iterações                      | $N                      |
| Wall-time médio (s/modelo)     | $MEAN                   |
| Desvio-padrão (s)              | $STDEV                  |
| Mínimo (s)                     | $VMIN                   |
| Máximo (s)                     | $VMAX                   |
| Mediana (s)                    | $MEDIAN                 |
| **Throughput (modelos/hora)**  | **$THROUGHPUT**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | \`${OUT_DAT}\`                     |
| Tamanho      | $OUT_SIZE bytes                    |
| MD5          | \`$MD5\`                           |

## Série Bruta (segundos)

\`\`\`
$(cat "$TIMES_FILE")
\`\`\`
EOF

echo ""
echo "==> Relatório gerado: $REPORT"
echo "==> Resumo: mean=$MEAN s, std=$STDEV s, throughput=$THROUGHPUT modelos/h"
echo "==> MD5 ($OUT_DAT): $MD5"
