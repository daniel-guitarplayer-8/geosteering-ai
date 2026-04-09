# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# buildValidamodels.py — Validação do simulador Fortran via modelos geológicos canônicos de referência
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Simula 6 modelos da literatura (Oklahoma, Devine, Hou et al.) com os parâmetros do simulador EM 1D TIV
# (PerfilaAnisoOmp.f08) e gera figuras comparativas do tensor H completo + perfis de resistividade.
#
# Modelos de referência:
#   1. Oklahoma 3 camadas        — TIV simples (ρ_h ≠ ρ_v em camada central)
#   2. Oklahoma 5 camadas        — TIV gradual com 5 contrastes
#   3. Devine 8 camadas          — Isotrópico (ρ_h = ρ_v), complexidade geológica real
#   4. Oklahoma 15 camadas       — Isotrópico, alta estratificação
#   5. Oklahoma 28 camadas       — TIV forte (ρ_v = 2×ρ_h), Tech. Report 32_2011 p.58
#   6. Hou, Mallan & Verdin 2006 — 7 camadas, modelo de referência para validação 3D→1D
#
# Formato de saída binário (22-col):
#   col0  = med (int32)   — índice da medida
#   col1  = z_obs         — profundidade do ponto médio T-R (m)
#   col2  = ρ_h           — resistividade horizontal verdadeira (Ω·m)
#   col3  = ρ_v           — resistividade vertical verdadeira (Ω·m)
#   col4  = Re(Hxx)  col5  = Im(Hxx)   ...   col20 = Re(Hzz)  col21 = Im(Hzz)
#
# Dependências:
#   - numpy, matplotlib, subprocess, shutil
#   - Executável Fortran compilado: tatu.x (gerado por 'make' no Makefile do Fortran_Gerador)
#
# Uso:
#   python buildValidamodels.py
#
# Autor: Daniel Leal
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
import numpy as np
import subprocess as sub
import time
import os
import math
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# ── CONFIGURAÇÃO DE SIMULAÇÃO ──────────────────────────────────────────────────────────────────────────────
# Parâmetros compartilhados com fifthBuildTIVModels.py.
# Para manter consistência entre treinamento e validação, estes valores DEVEM coincidir
# com os definidos no script de geração de modelos TIV.
#
# IMPORTANTE: Ao alterar freqs, angulos ou dTR aqui, altere também em fifthBuildTIVModels.py.
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
freqs   = [20000.]          #[2000., 6000.]   # frequências (Hz) — ex: 2k=δ≈11m, 6k=δ≈6.5m (ρ=10Ω·m)
angulos = [0.]              #[0., 30.]        # ângulos de inclinação (°) — 0°=vertical, 30°=off-diagonal≠0
nf      = len(freqs)        # número de frequências
na      = len(angulos)      # número de ângulos
pmed    = 0.2               # passo entre medidas (m) — resolução vertical do perfil
dTR     = [1.0]             #[8.19, 20.43]    # espaçamentos T-R (m) — investigação profunda (ARC/Periscope)
_nTR    = len(dTR)           # número de pares T-R

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# ── CAMINHOS ───────────────────────────────────────────────────────────────────────────────────────────────
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
mypath       = os.path.dirname(os.path.realpath(__file__)) + '/'
mymodel      = mypath + 'model.in'
fortran_exec = mypath + 'tatu.x'

# Verifica se o executável Fortran existe antes de prosseguir
if not os.path.isfile(fortran_exec):
    raise FileNotFoundError(
        f'Executável Fortran não encontrado: {fortran_exec}\n'
        f'Execute "make" no diretório {mypath} para compilar tatu.x.'
    )

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# ── MODELOS CANÔNICOS DE VALIDAÇÃO ─────────────────────────────────────────────────────────────────────────
# Cada modelo retorna (ncam, esp, resh, resv):
#   ncam  — número de camadas (incluindo semi-infinitas superior e inferior)
#   esp   — lista de espessuras das ncam-2 camadas internas (m)
#   resh  — lista de resistividades horizontais por camada (Ω·m)
#   resv  — lista de resistividades verticais por camada (Ω·m)
#
# Para modelos isotrópicos: resv = resh (λ = ρ_v/ρ_h = 1)
# Para modelos anisotrópicos: resv ≠ resh (λ > 1 típico em folhelhos)
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def modelo_valida(imod):
    """Retorna (ncam, esp, resh, resv) para os 6 modelos canônicos de validação.

    Args:
        imod: Índice do modelo (0-based). Valores válidos: 0..5.

    Returns:
        Tupla (ncam, esp, resh, resv) onde:
        - ncam (int): número total de camadas
        - esp (list[float]): espessuras das ncam-2 camadas internas (m)
        - resh (list[float]): resistividades horizontais por camada (Ω·m)
        - resv (list[float]): resistividades verticais por camada (Ω·m)

    Raises:
        ValueError: Se imod não está no intervalo [0, 5].

    Note:
        Ref (modelos 1-5): Tech. Report 32_2011 — Oklahoma/Devine geological models.
        Ref (modelo 6): Hou, Mallan & Verdin (2006), "Finite-Difference simulation
        of borehole EM measurements in 3D anisotropic media using coupled
        scalar-vector potentials", Geophysics 71(5), F101-F114.
    """
    pe = 0.3048  # conversão pé → metro (1 ft = 0.3048 m)
    if (imod + 1) == 1:      # Oklahoma 3 camadas — TIV simples
        ncam = 3;  esp = [8*pe]
        resh = [1., 20., 1.];  resv = [1., 40., 1.]
    elif (imod + 1) == 2:    # Oklahoma 5 camadas — TIV gradual
        ncam = 5;  esp = [17.*pe, 8.*pe, 4.*pe]
        resh = [5., 50., 2., 15., 4.5];  resv = [10., 100., 4., 30., 9.]
    elif (imod + 1) == 3:    # Devine 8 camadas — isotrópico
        ncam = 8;  esp = [20*pe, 15*pe, 19*pe, 16*pe, 15*pe, 45*pe]
        resh = [2., 3., 1.8, 4.71, 8.89, 5., 3.2, 8.];  resv = list(resh)
    elif (imod + 1) == 4:    # Oklahoma 15 camadas — isotrópico
        ncam = 15
        esp  = [17*pe, 8*pe, 4*pe, 3*pe, 7*pe, 4*pe, 6*pe, 3*pe, 5*pe, 7*pe, 18*pe, 8*pe, 7*pe]
        resh = [5., 50., 2., 15., 4.5, 100., 3.5, 450., 30.12, 602.41, 20., 746.27, 200., 7.5, 500.]
        resv = list(resh)
    elif (imod + 1) == 5:    # Oklahoma 28 camadas — anisotrópico forte (ρ_v = 2×ρ_h)
        ncam = 28
        z    = [46,63,71,75,78,85,89,95,98,103,110,128,136,143,153,157,162,165,169,
                173,177,182,185,187,189,191,203]
        esp  = list(np.diff(z) * pe)
        resh = [10.,100.,4.,30.,9.,200.,7.,909.,60.,1250.,40.,1416.,400.,15.,1000.,
                179.,1000.,15.,75.,9.,20.,100.,18.,200.,75.,149.,7.,11.]
        resv = [2*r for r in resh]
    elif (imod + 1) == 6:    # Hou, Mallan & Verdin (2006) — 7 camadas
        ncam = 7;  esp = [1.52, 2.35, 2.1, 1.88, 0.92]
        resh = [1., 80., 1., 10., 1., 0.3, 1.];  resv = [10., 80., 10., 10., 10., 0.3, 10.]
    else:
        raise ValueError(f'modelo_valida: imod={imod} fora do intervalo [0, 5]')
    return ncam, list(esp), list(resh), list(resv)


# Nomes legíveis dos modelos (usados em prints e títulos das figuras)
NOMES_MODELOS = [
    'Oklahoma 3 cam.', 'Oklahoma 5 cam.', 'Devine 8 cam. (isotrópico)',
    'Oklahoma 15 cam.', 'Oklahoma 28 cam. (anisotrópico)', 'Hou et al. 7 cam.'
]
NMODELS_VALIDA = 6

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# ── UTILITÁRIOS ────────────────────────────────────────────────────────────────────────────────────────────
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def ler_arquivo_com_int_float(arquivo):
    """Lê arquivo texto com colunas mistas (int/float/string) e converte automaticamente.

    Usado para ler o arquivo info*.out gerado pelo Fortran, que contém:
      Linha 1: nt  nf  nmaxmodel  (inteiros)
      Linha 2: theta(1..nt)       (floats — ângulos em graus)
      Linha 3: freq(1..nf)        (floats — frequências em Hz)
      Linha 4: nmeds(1..nt)       (inteiros — nº de medidas por ângulo)

    Args:
        arquivo: Caminho do arquivo a ser lido.

    Returns:
        Lista de listas, cada sublista representando uma linha com valores convertidos.
    """
    dados = []
    with open(arquivo, "r") as informa:
        for lin in informa:
            linha = lin.strip()
            if not linha:
                continue
            colunas = linha.split()
            linha_convertida = []
            for coluna in colunas:
                try:
                    valor = int(coluna)
                except ValueError:
                    try:
                        valor = float(coluna)
                    except ValueError:
                        valor = coluna
                linha_convertida.append(valor)
            dados.append(linha_convertida)
    return dados


# Tipo binário 22-col (formato de saída do PerfilaAnisoOmp.f08):
#   col0 = med (int32), col1..col21 = z_obs, ρ_h, ρ_v, Re/Im dos 9 componentes tensor H (float64)
dtyp = np.dtype([('col0', np.int32)] + [('col{}'.format(i), np.float64) for i in range(1, 22)])


#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL — Simulação Fortran dos 6 modelos de validação
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
print('═'*89)
print('VALIDAÇÃO — Modelos geológicos de referência (buildValidamodels.py)')
print('═'*89)
print(f'  Parâmetros: nf={nf}, freq={freqs} Hz | na={na}, ang={angulos}° | nTR={_nTR}, dTR={dTR} m')

# ── Janela de investigação ──────────────────────────────────────────────────────────────────────────────
# tj_v é determinada pelo modelo com a maior espessura total, acrescida de 20 m de margem
# para garantir que as camadas semi-infinitas (superior e inferior) sejam bem amostradas.
somas_esp_v = [np.sum(modelo_valida(mv)[1]) for mv in range(NMODELS_VALIDA)]
tj_v = max(somas_esp_v) + 20.0
print(f'  tj_v={tj_v:.2f} m (maior espessura={max(somas_esp_v):.2f} m + 20 m margem), pmed={pmed} m')
print('─'*89)

# ── Display de parâmetros (skin depth e DOI) ────────────────────────────────────────────────────────────
print(f'  nf={nf}  Frequências (Hz) : {freqs}')
for fi in freqs:
    delta_10 = 503 * math.sqrt(10 / fi)
    print(f'    f={int(fi):>6} Hz — skin depth (ρ=10Ω·m): δ≈{delta_10:.1f} m')
print(f'  na={na}  Ângulos (°)      : {angulos}')
for ai in angulos:
    if ai == 0.:
        print(f'    θ=0°  → poço vertical, off-diagonal = 0')
    else:
        print(f'    θ={int(ai)}°  → poço desviado, off-diagonal ≠ 0, DOI radial = dTR×|sin({int(ai)}°)|')
print(f'  nTR={_nTR}  Espaçamentos T-R (m): {dTR}')
_ang_max = max(angulos)
for dtr_i in dTR:
    for fi in freqs:
        delta_i = 503 * math.sqrt(10 / fi)
        doi_est = dtr_i * abs(math.sin(math.radians(_ang_max))) if _ang_max > 0 else 0.0
        print(f'    dTR={dtr_i}m, f={int(fi/1000)}kHz → r_k(θ={int(_ang_max)}°)={doi_est:.2f}m, δ≈{delta_i:.1f}m')
print('─'*89)

# ── Configuração OpenMP ─────────────────────────────────────────────────────────────────────────────────
_omp_threads = os.environ.get('OMP_NUM_THREADS', None)
if _omp_threads:
    print(f'[OpenMP] OMP_NUM_THREADS = {_omp_threads} thread(s)  (definido pelo usuário)')
else:
    import multiprocessing as _mp
    print(f'[OpenMP] OMP_NUM_THREADS não definido — OpenMP usará o padrão do sistema '
          f'({_mp.cpu_count()} núcleos lógicos detectados)')
print('─'*89)

# ── Loop de simulação Fortran ───────────────────────────────────────────────────────────────────────────
filename_valida_base = 'validacao'
start_time_v = time.perf_counter()

for mv in range(NMODELS_VALIDA):
    ncam_v, esp_v, resh_v, resv_v = modelo_valida(mv)
    soma_esp_v = np.sum(esp_v)
    h1_v = (tj_v - soma_esp_v) / 2.0
    print(f'  [{mv+1}/{NMODELS_VALIDA}] {NOMES_MODELOS[mv]} '
          f'— {ncam_v} cam., soma_esp={soma_esp_v:.2f}m, h1={h1_v:.2f}m')

    # ── Escrita do model.in para este modelo de validação ─────────────────────────────────────────────
    with open(mymodel, 'w') as f_mod:
        f_mod.write(str(nf) + '                 !número de frequências\n')
        for _idx, _fi in enumerate(freqs, 1):
            f_mod.write(str(_fi) + '           ' + f'!frequência {_idx}\n')
        f_mod.write(str(na) + '                 !número de ângulos de inclinação\n')
        for _idx, _ai in enumerate(angulos, 1):
            f_mod.write(str(_ai) + '               ' + f'!ângulo {_idx}\n')
        f_mod.write(str(h1_v) + '              !altura do primeiro ponto-médio T-R\n')
        f_mod.write(str(tj_v) + '             !tamanho da janela de investigação\n')
        f_mod.write(str(pmed)  + '               !passo entre as medidas\n')
        f_mod.write(str(_nTR) + '                 !número de pares T-R\n')
        for dtr_v in dTR:
            f_mod.write(str(dtr_v) + '               !distância T-R\n')
        f_mod.write(filename_valida_base + '              !nome dos arquivos de saída\n')
        f_mod.write(str(ncam_v) + '                !número de camadas\n')
        for jc in range(ncam_v):
            myrhoh = float(resh_v[jc]);  myrhov = float(resv_v[jc])
            if myrhov < myrhoh: myrhov = myrhoh
            suffix = '     !resistividades horizontal e vertical' if jc == 0 else ''
            f_mod.write(f'{round(myrhoh, 4)}    {round(myrhov, 4)}{suffix}\n')
        for jc in range(ncam_v - 3):
            suffix = '              !espessuras das n-2 camadas' if jc == 0 else ''
            f_mod.write(f'{round(esp_v[jc], 4)}{suffix}\n')
        f_mod.write(f'{round(esp_v[-1], 4)}\n')
        f_mod.write(f'{mv+1} {NMODELS_VALIDA}         !modelo atual e o número máximo de modelos\n')
        # ── F5/F7/F6/Filtro — Defaults desabilitados (v9.0 backward compatible) ──
        f_mod.write('0                 !F5: use_arbitrary_freq (0=desabilitado)\n')
        f_mod.write('0                 !F7: use_tilted_antennas (0=desabilitado)\n')
        f_mod.write('0                 !F6: use_compensation (0=desabilitado)\n')
        f_mod.write('0                 !Filtro: 0=Werthmuller (default)')

    # ── Execução do simulador Fortran ─────────────────────────────────────────────────────────────────
    try:
        sub.run([fortran_exec], check=True, text=True, capture_output=True)
    except sub.CalledProcessError as e:
        print(f'    [ERRO Fortran]: {e.stderr[:300]}')

end_time_v = time.perf_counter()
print('─'*89)
print(f'  Tempo de simulação Fortran (validação): {(end_time_v - start_time_v):.2f} s')

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# ── Geração do .out pelo Python (garantia total, independente do Fortran) ──────────────────────────────
# O Fortran só grava o .out quando modelm == nmaxmodel. O Python gera sempre,
# evitando o bug de int() truncation que impedia a geração para ntmodels não múltiplo de 40.
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
out_src_v = mypath + 'info' + filename_valida_base + '.out'
_nmeds_v  = [math.ceil(tj_v / (pmed * math.cos(math.radians(a)))) for a in angulos]
with open(out_src_v, 'w') as _f_out_v:
    _f_out_v.write(f' {na} {nf} {NMODELS_VALIDA}\n')
    _f_out_v.write(' ' + ' '.join(f'{float(a):.1f}' for a in angulos) + '\n')
    _f_out_v.write(' ' + ' '.join(f'{float(f):.1f}' for f in freqs) + '\n')
    _f_out_v.write(' ' + ' '.join(str(n) for n in _nmeds_v) + '\n')
print(f'  [.out validação gerado pelo Python] {out_src_v}')
print(f'  nt={na}  nf={nf}  nmaxmodel={NMODELS_VALIDA}  nmeds_v={_nmeds_v}')

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# ── Renomeação de arquivos de saída (dTR no nome + .out idêntico ao .dat) ──────────────────────────────
# Formato do Fortran:
#   nTR=1: {filename}.dat         (sem sufixo _TR)
#   nTR>1: {filename}_TR{k}.dat   (com sufixo _TR por par T-R)
# Após renomeação: {filename}_TR{k}_d{dTR}m.dat + .out idêntico
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
valida_dat_files = []
print('  Renomeação de arquivos de validação:')
for itr, dtr in enumerate(dTR):
    dtr_tag    = str(dtr).replace('.', 'p')
    _tr_suf_v  = f'_TR{itr+1}' if _nTR > 1 else ''             # nTR=1: Fortran não adiciona sufixo _TR
    old_dat_v  = mypath + filename_valida_base + f'{_tr_suf_v}.dat'
    new_name_v = filename_valida_base + f'_TR{itr+1}_d{dtr_tag}m'
    new_dat_v  = mypath + new_name_v + '.dat'
    new_out_v  = mypath + new_name_v + '.out'
    if os.path.exists(old_dat_v):
        os.rename(old_dat_v, new_dat_v)
        print(f'    .dat: {os.path.basename(old_dat_v)} → {os.path.basename(new_dat_v)}')
    else:
        print(f'    [AVISO] .dat não encontrado: {os.path.basename(old_dat_v)}')
    if os.path.exists(out_src_v):
        shutil.copy(out_src_v, new_out_v)
        print(f'    .out: info{filename_valida_base}.out → {os.path.basename(new_out_v)}')
    valida_dat_files.append((new_dat_v, f'TR{itr+1} (dTR={dtr} m) — Validação', new_out_v))

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# ── Leitura e validação numérica ───────────────────────────────────────────────────────────────────────────
# Verifica integridade dos arquivos .dat gerados: contagem de registros, NaN e Inf.
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
nmeds_v = [math.ceil(tj_v / (pmed * math.cos(math.radians(a)))) for a in angulos]
records_per_model_v = sum(nmeds_v) * nf
_nmeds_v_str = '+'.join(str(n) for n in nmeds_v)
print('─'*89)
print(f'  Registros esperados/modelo: ({_nmeds_v_str})×{nf}={records_per_model_v} '
      f'| Total={NMODELS_VALIDA*records_per_model_v}')
for dat_path_v, label_v, out_path_v in valida_dat_files:
    try:
        mydat_v   = np.fromfile(dat_path_v, dtype=dtyp)
        myarr_v   = np.array(mydat_v.tolist())
        nrow_v    = myarr_v.shape[0]
        expected  = NMODELS_VALIDA * records_per_model_v
        has_nan_v = np.isnan(myarr_v).any()
        has_inf_v = np.isinf(myarr_v).any()
        status_v  = 'OK' if not has_nan_v and not has_inf_v and nrow_v == expected else 'ATENÇÃO'
        print(f'  [{status_v}] {label_v}')
        print(f'         .dat  : {dat_path_v}')
        print(f'         .out  : {out_path_v}')
        print(f'         Regs  : {nrow_v} (esperado {expected}) | NaN={has_nan_v} | Inf={has_inf_v}')
    except FileNotFoundError:
        print(f'  [ERRO] {label_v} — arquivo não encontrado: {dat_path_v}')
print('─'*89)

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# PLOTAGEM — Tensor H completo + perfil de resistividade para cada modelo de validação
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# Para cada combinação (TR × modelo × ângulo × frequência), gera uma figura com:
#   - Coluna 0 (GridSpec span 3 linhas): perfil de ρ_h e ρ_v em escala log-x
#   - Colunas 1-6 (grade 3×6): Re/Im das 9 componentes do tensor H
# Eixo Y compartilhado entre todos os painéis (profundidade invertida, crescendo para baixo).
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
tensor_names_v = [
    ['H_{xx}', 'H_{xy}', 'H_{xz}'],
    ['H_{yx}', 'H_{yy}', 'H_{yz}'],
    ['H_{zx}', 'H_{zy}', 'H_{zz}']
]
# usetex=False: usa mathtext nativo do matplotlib — suporta Unicode (θ, °, ç, ã)
# sem depender de instalação LaTeX externa. Evita RuntimeError com caracteres acentuados.
plt.rcParams.update({
    "text.usetex": False, "font.family": "serif",
    "font.size": 12, "axes.titlesize": 13,
})

# Diretório para salvar as figuras de validação
plots_dir = mypath + 'validacao_plots/'
os.makedirs(plots_dir, exist_ok=True)

# Layout binário por modelo (ntheta=na, nf):
#   k=0 (θ=angulos[0]): j=0 (freqs[0]): nmeds_v[0] registros | ... | j=nf-1: nmeds_v[0] registros
#   k=1 (θ=angulos[1]): j=0 (freqs[0]): nmeds_v[1] registros | ... | j=nf-1: nmeds_v[1] registros
# offset global do bloco (k,j) dentro do modelo:
#   sum_{k'<k} nmed(k') * nf  +  j * nmed(k)
_angulos_plot = angulos
_freqs_plot   = freqs
_nmeds_plot   = nmeds_v
_palette_re   = ['darkblue', 'navy', 'steelblue', 'royalblue', 'cornflowerblue']
_palette_im   = ['darkred',  'firebrick', 'indianred', 'salmon', 'tomato']
_colors_re    = (_palette_re * (1 + na // len(_palette_re)))[:na]  # seguro para qualquer na
_colors_im    = (_palette_im * (1 + na // len(_palette_im)))[:na]

for dat_path_v, label_v, _ in valida_dat_files:
    try:
        mydat_v = np.fromfile(dat_path_v, dtype=dtyp)
        myarr_v = np.array(mydat_v.tolist())
    except FileNotFoundError:
        print(f'  [AVISO] Arquivo de validação não encontrado para plotagem: {dat_path_v}')
        continue
    tr_tag = os.path.splitext(os.path.basename(dat_path_v))[0].replace('validacao_', '')

    for idx_m in range(NMODELS_VALIDA):
        start_r = idx_m * records_per_model_v
        if start_r + records_per_model_v > len(myarr_v):
            break

        ncam_v2, esp_v2, _, _ = modelo_valida(idx_m)
        interfaces = [0.0] + list(np.cumsum(esp_v2))
        nome_safe  = NOMES_MODELOS[idx_m].replace(' ', '_').replace('.', '')

        # Offset cumulativo por ângulo (em registros dentro do modelo)
        _k_offsets = []
        _cum = 0
        for nmed_k in _nmeds_plot:
            _k_offsets.append(_cum)
            _cum += nmed_k * len(_freqs_plot)

        # Itera sobre todos os blocos (ângulo × frequência)
        for ki, (theta_val, nmed_k, k_off) in enumerate(zip(_angulos_plot, _nmeds_plot, _k_offsets)):
            for ji, freq_val in enumerate(_freqs_plot):
                blk_start = start_r + k_off + ji * nmed_k
                blk_end   = blk_start + nmed_k
                dados_plot = myarr_v[blk_start : blk_end]
                zobs = dados_plot[:, 1]  # col1 = z_obs (formato 22-col)

                # ── Layout: col 0 = perfil de resistividade | cols 1-6 = tensor H ──
                # GridSpec com 7 colunas: a primeira (width_ratio=1.8) exibe ρ_h e ρ_v
                # em escala log-x, abrangendo as 3 linhas da figura; as 6 restantes
                # mantêm a grade 3×6 do tensor EM. O eixo Y é compartilhado entre
                # todos os painéis para alinhamento direto de profundidade.
                fig = plt.figure(figsize=(28, 12))
                _gs = gridspec.GridSpec(3, 7, figure=fig,
                                        width_ratios=[1.8, 1, 1, 1, 1, 1, 1])
                fig.suptitle(
                    f'Tensor H  —  {NOMES_MODELOS[idx_m]} ({ncam_v2} cam.)  |  {label_v}  |  '
                    r'$\theta$' + f'={int(theta_val)}°,  f={int(freq_val/1000)} kHz',
                    fontsize=15, y=0.98
                )

                # ── Painel de resistividade (col 0, span 3 linhas) ──────────────────
                # col2 = ρ_h, col3 = ρ_v (formato 22-col). O perfil step-function
                # emerge naturalmente porque o Fortran repete a resistividade da camada
                # em todos os registros com a mesma profundidade de investigação.
                rho_h_plot = dados_plot[:, 2]  # Ohm·m — resistividade horizontal
                rho_v_plot = dados_plot[:, 3]  # Ohm·m — resistividade vertical
                ax_rho = fig.add_subplot(_gs[:, 0])
                ax_rho.semilogx(rho_h_plot, zobs,
                                color='steelblue', linewidth=2.2, label=r'$\rho_h$')
                ax_rho.semilogx(rho_v_plot, zobs,
                                color='darkorange', linewidth=2.2, linestyle='--',
                                label=r'$\rho_v$')
                for interf in interfaces:
                    ax_rho.axhline(y=interf, color='black', linestyle='--', lw=1, alpha=0.7)
                ax_rho.invert_yaxis()
                ax_rho.set_xlabel(r'Resistividade ($\Omega{\cdot}$m)', fontsize=11)
                ax_rho.set_ylabel('Profundidade (m)', fontsize=11)
                ax_rho.set_title(r'$\rho_h$ e $\rho_v$ (modelo verdadeiro)', fontsize=12)
                ax_rho.legend(fontsize=11, loc='lower right')
                ax_rho.grid(True, which='both', linestyle=':', alpha=0.5)

                # ── Painéis do tensor 3×6 (cols 1-6) compartilhando eixo Y ─────────
                _axs = [[fig.add_subplot(_gs[ti, 1 + tj], sharey=ax_rho)
                          for tj in range(6)] for ti in range(3)]
                col_idx = 4  # tensor começa na col4 (formato 22-col)
                for ti in range(3):
                    for tj2 in range(3):
                        name  = tensor_names_v[ti][tj2]
                        ax_re = _axs[ti][tj2*2]
                        ax_re.plot(dados_plot[:, col_idx], zobs,
                                   color=_colors_re[ki], linewidth=1.5)
                        ax_re.set_title(rf'$\mathrm{{Re}}({name})$')
                        ax_re.grid(True, linestyle=':', alpha=0.5)
                        for interf in interfaces:
                            ax_re.axhline(y=interf, color='black', linestyle='--', lw=1, alpha=0.7)
                        if ti == 2: ax_re.set_xlabel('Amplitude')
                        ax_re.tick_params(labelleft=False)
                        col_idx += 1
                        ax_im = _axs[ti][tj2*2 + 1]
                        ax_im.plot(dados_plot[:, col_idx], zobs,
                                   color=_colors_im[ki], linewidth=1.5)
                        ax_im.set_title(rf'$\mathrm{{Im}}({name})$')
                        ax_im.grid(True, linestyle=':', alpha=0.5)
                        for interf in interfaces:
                            ax_im.axhline(y=interf, color='black', linestyle='--', lw=1, alpha=0.7)
                        if ti == 2: ax_im.set_xlabel('Amplitude')
                        ax_im.tick_params(labelleft=False)
                        col_idx += 1
                plt.tight_layout(rect=[0, 0.02, 1, 0.96])

                ang_tag  = f'ang{int(theta_val)}'
                freq_tag = f'f{int(freq_val/1000)}k'
                fig_base = plots_dir + f'valida_mod{idx_m+1:02d}_{nome_safe}_{tr_tag}_{ang_tag}_{freq_tag}'
                plt.savefig(fig_base + '.png', dpi=400, bbox_inches='tight')
                print(f'  [SALVO] mod{idx_m+1} {tr_tag} θ={int(theta_val)}° f={int(freq_val/1000)}kHz: '
                      f'{os.path.basename(fig_base)}.png')
                plt.close(fig)  # libera memória — evita acúmulo de figuras em loops longos

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# ── Relatório final ────────────────────────────────────────────────────────────────────────────────────────
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
total_time = time.perf_counter() - start_time_v
print('═'*89)
print(f'Validação concluída. {NMODELS_VALIDA} modelos de referência processados em {total_time:.2f} s.')
print(f'  Figuras salvas em: {plots_dir}')
print('═'*89)
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
