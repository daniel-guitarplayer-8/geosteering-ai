#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orquestrador Sequencial Paralelo e Gerador Estocástico para Geração Massiva de Datasets Geofísicos.

Módulo: batch_runner.py
Versão: 3.0 (Metamorfose Dinâmica Clean Way)

Descrição Arquitetural:
-----------------------
Este módulo atua como o Controlador Mestre (Orquestrador) do simulador de perfis eletromagnéticos 
1D utilizado no projeto Geosteering AI. Ele implementa o paradigma "Central Master-Plan, Parallel Execution",
substituindo a antiga operação linear e presa a I/O baseada em modelos estáticos.

Fluxo de Trabalho Primário:
1. Geração Centralizada (Master CPU): Importa o motor matemático nativo `fifthBuildTIVModels` e 
   compila, unicamente em memória RAM, dezenas de milhares de dicionários contendo distribuições 
   Quasi-Monte Carlo (Sobol) para as variáveis de engenharia de poço (resistividades anisotrópicas 
   e espessuras estratigráficas).
2. Distribuição Computacional (Chunks): Fatiamento equitativo dos perfis matemáticos por processos.
3. Execução Isolada (Workers Sandboxes): Cada worker multiprocessado recria o arquivo de entrada (`model.in`)
   de forma dinâmica, reescrevendo estocasticamente a estrutura geológica antes de cada acionamento 
   do Fortran. Instancia o binário compilado (`tatu.x`) isoladamente, impedindo Data Race e colisões.
4. Concatenação Binária e Inspeção: Realiza um merge atômico dos resíduos gerados localmente e inspeciona 
   o binário unificado à procura de de NaNs/Infs (falências numéricas) no espaço complexo eletromagnético,
   empregando Mapeamento de Memória (Memory Mapping) para eficiência algorítmica `O(1)` na RAM.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Injeta o diretório do próprio script no PATH, garantindo que módulos adjacentes
# (como fifthBuildTIVModels.py) sejam localizados independentemente do diretório de
# invocação. Usar __file__ em vez de os.getcwd() evita falha quando o script é
# chamado de outro diretório (ex.: python Fortran_Gerador/batch_runner.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ==============================================================================
# SUB-ROTINAS DO ORQUESTRADOR CENTRAL
# ==============================================================================

def fabricar_universos_aleatorios(ntmodels):
    """
    Sintetiza de forma determinística-aleatória (distribuição estocástica) o conjunto completo 
    de modelos geológicos matemáticos aplicados à propagação EM.

    Utilização:
        A matriz base do universo simulado repousa sobre a divisão fracionária rigorosa projetada 
        para garantir que a Deep Learning que ingira os dados reconheça variabilidades críticas. O motor
        interno apoia-se num hipercubo de amostras Sobol, garantindo cobertura espacial densa para o problema inverso.

    Args:
        ntmodels (int): A volumetria alvo do dataset informada pelo usuário via ArgumentParser.

    Returns:
        list[dict]: Uma lista linear de dicionários estruturados, onde cada dicionário carrega a assinatura de uma formação:
                    - 'n_layers': int, o número de camdas estáticas.
                    - 'rho_h': list de float, Condutividade transversal (Horizontal).
                    - 'rho_v': list de float, Condutividade vertical da fratura orientada.
                    - 'thicknesses': list de float, as espessuras sequenciais relativas à profundidade total (tji).

    Raises:
        ImportError: Capturada caso o gerador de matrizes matemáticas original seja corrompido ou falte no ecossistema.
    """
    try:
         import Fortran_Gerador.fifthBuildTIVModels as fbt
    except ImportError:
        try:
           import fifthBuildTIVModels as fbt
        except ImportError as e:
           print("[FATAL ERROR] Falha Crítica: O motor raiz de Entropia Estocástica ('fifthBuildTIVModels.py') não é alcançável.")
           raise e

    print(f"\n[MATRIZ ESTOCÁSTICA] Injetando a Fatura de Geologias e Sorteios ({ntmodels} perfis via Sobol)...")
    t0 = time.perf_counter()

    # O pipeline constrói lotes fracionados usando arquiteturas patológicas desenhadas 
    # por geofísicos para forçar invariância nos tensores EM das NNs. As frações são fixas:
    m1 = fbt.baseline_empirical_2(nmodels = int(ntmodels * 0.15))
    m2 = fbt.baseline_ncamuniform_2(nmodels = int(ntmodels * 0.10))
    m3 = fbt.baseline_thick_thicknesses_2(nmodels = int(ntmodels * 0.10))
    m4 = fbt.unfriendly_empirical_2(nmodels = int(ntmodels * 0.125))
    m5 = fbt.unfriendly_noisy_2(nmodels = int(ntmodels * 0.175))
    m6 = fbt.generate_pathological_models_2(nmodels = int(ntmodels * 0.1))
    
    m7 = fbt.generate_pathological_models_2(
        nmodels = int(ntmodels * 0.125), rho_h_min = 0.1, rho_h_max = 1500, 
        lambda_min = 1, lambda_max = 1.001, min_thickness_internal = 10, p_camfina = 0.3)
        
    m8 = fbt.baseline_empirical_2(
        nmodels = int(ntmodels * 0.125), rho_h_min = 0.1, rho_h_max = 1500, 
        rhohdistr = 'loguni', lambda_min = 1, lambda_max = 1.0001, min_thickness = 0.5)

    # Concatenação atômica no escopo Mestre de toda a cadeia produzida.
    models_list = m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8
    
    # Reparação Algoritmica: Devido ao arredondamento da função de truncamento `int()` operando 
    # em floats (frações do total demandado), é matematicamente previsível a falta de alguns modelos
    # nas decimais somadas. O delta algébrico é saneado adicionando profiles na categoria baseline puro.
    faltante = ntmodels - len(models_list)
    if faltante > 0:
        models_list += fbt.baseline_empirical_2(nmodels = faltante)

    print(f"  -> {len(models_list)} Universos computados de ponta a ponta na Memória Primária Mestre. (T= {time.perf_counter() - t0:.2f}s)")
    return models_list


def extract_model_in_header(model_in_path):
    """
    Extrator estático do prólogo de configuração primária física (`model.in`).

    Funcionamento:
        Filtra recursivamente da linha 1 até o terminador designado `!nome dos arquivos de saída`. 
        Tais linhas prescrevem ao precondicionador numérico do Fortran parâmetros intransigentes (ferramenta):
        - Frequências harmônicas.
        - Ângulos de mergulho (Dip / Anisotropia Relativa).
        - Espaçamentos T-R nativos para cálculo DOI (Profundidade de Investigação).
        Tais informações não devem sofrer metamorfose por perfil para que a ferramenta permaneça em equivalência.

    Args:
        model_in_path (str): URI absolutada ou relativa do arquivo template (ex: `model.in`).
    
    Returns:
        list[str]: Array vetorial de strings, mantendo inclusive os delimitadores `\n` originais para
                   posterior despejo no disco.
    """
    header_lines = []
    with open(model_in_path, 'r') as f:
        for line in f:
            header_lines.append(line)
            if '!nome dos arquivos de saída' in line:
                break
    return header_lines


def parse_model_in(filepath):
    """
    Decodificador semi-analítico das informações visuais indexadas do cabeçalho de simulação.

    Objetivo:
        Extrair os números absolutos operados na janela (Para Logs em terminal amigáveis no Console GUI).

    Args:
        filepath (str): Caminho local contendo a rota do arquivo estruturado (model.in).

    Returns:
        dict ou None: Um dicionário de mapeamento descrevendo os inputs geométricos da ferramenta de Logging.
                      Devolve None caso haja anomalia severa estrutural (ex: quebra de contrato numérico).
    """
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        idx = 0
        nf = int(lines[idx].split()[0]); idx += 1
        freqs = []
        for _ in range(nf):
            freqs.append(float(lines[idx].split()[0])); idx += 1
            
        ntheta = int(lines[idx].split()[0]); idx += 1
        angles = []
        for _ in range(ntheta):
            angles.append(float(lines[idx].split()[0])); idx += 1
            
        idx += 3 
        nTR = int(lines[idx].split()[0]); idx += 1
        trs = []
        for _ in range(nTR):
            trs.append(float(lines[idx].split()[0])); idx += 1
            
        return {'nf': nf, 'freqs': freqs, 'ntheta': ntheta, 'angles': angles, 'nTR': nTR, 'trs': trs}
    except Exception as e:
        return None


def validar_integridade(filepath):
    """
    Scanner C++ Integrado Preventivo contra Infs/NaNs com complexidade Assintótica `O(1)` de RAM.

    Conceito Matemático:
        No processo da solução recursiva das matrizes de Reflexão/Transmissão Eletromagnética nas interfaces,
        solvers diferenciais tendem a estourar assintotas gerando `NaN`. Identificar vazamentos em bases
        de dados textuais gigantescas afaga o sistema. Por isso o simulador despeja binários brutos cravados.

    Engenharia (Memmap):
        Configura o dtype de Numpy em mapeamento vetorial na placa de disco (memmap), transladando a stream 
        sequencial do Fortran (1 `Int32` para ID + 21 `Float64` para Tensões) em visualizações mascaradas super-rápidas. 
        Permite validar Petabytes em segundos avaliando blocos virtuais de 172 bytes.

    Args:
        filepath (str): Arquivo `.dat` final ou fragmento nativo cujo encoding de matriz será varrido.
    """
    try:
        # Assinatura contratual entre C/Fortran Compiler e a ponte Python. 172 Bytes/Frame:
        dtyp = np.dtype([('col0', np.int32)] + [('col{}'.format(i), np.float64) for i in range(1, 22)])
        
        # O Memory Map anexa o disco diretamente no ponteiro interno da CPython.
        mmap = np.memmap(filepath, dtype=dtyp, mode='r')
        
        has_error = False
        error_logs = []
        
        # Ignora a coluna [0] por se tratar de integer indexador natural. Pula diretamente aos campos elétricos complexos.
        for i in range(1, 22):
            col_name = f'col{i}'
            col_vuln = mmap[col_name]
            
            anomalia = np.isnan(col_vuln).any() or np.isinf(col_vuln).any()
            if anomalia:
                has_error = True
                error_logs.append(f"Detectado Vazamento Funcional Inf/NaN no Escopo do Tensor da Coluna: {col_name}")
                break 
                
        if has_error:
            print(f"    [!] AVISO CRÍTICO DE SOLVER: Falhas matemáticas numéricas (NaN/Inf) na Matriz Discreta de {filepath}")
            for err in error_logs:
                print(f"        -> {err}")
        else:
            print(f"    [+] CERTIFICADO DE INTEGRIDADE: Auditoria finalizada. Validação rigorosa reporta 0 falhas no Log Binário Mestre.")
            
        del mmap # Limpeza estrita e imediata do Buffer Virtual (desaloca referências ativas ao arquivo).
    except Exception as e:
        print(f"    [!] Colapso Operacional no Leitor Binário {filepath}: Diagnóstico Inválido. Razão: {e}")


def run_model_batch(args):
    """
    Fábrica Executora do Perímetro Assíncrono (Worker Sandbox).
    
    Finalidade:
        Esta rotina encarna a verdadeira Metamorfose Estocástica. Ela opera em um cordão umbilical 
        completamente desconectado da Main. Possui seu próprio executável `.x` manipulado clonado no escopo 
        da pasta `/tmp`, gerindo chamadas multithreading OpenMP que o Sistema Operacional dispacher.
        
    Execução e Lógica Dinâmica Tática:
        Em lugar da leitura rígida do `arquivo.in` e simples incrementação de contador da iteração passada, este motor:
        1. Resgata e grava o Cabeçalho Estático.
        2. Analisa o `geo_dict` do loop de Chunk para inferir topograficamente Resistividades Finais e Espessuras Variáveis.
        3. Realiza o Print-Out numérico para as linhas exclusivas do `model.in` local, instanciando uma infraestrutura unicamente 
           exclusiva para os solvers Fortran subjacentes engajarem sua marcha.
        4. Transfere sub-arquivos binários retornando o tracking da finalização para o `Future`.

    Args:
        args (tuple): Contrato empacotado que engloba (worker_id, chunk_models_list, header_lines, model_start, model_end, tatu_path, omp_threads)

    Returns:
        tuple: (ID Original do Worker Ativo, Offset de Início, Offset do Fim Sub-pacote, Array de tuplas dos DATs Fragmentados, Array das Tags OUT)
    """
    # F5/F7: args estendidos com flags opcionais (backward compat: len(args)==7 → defaults)
    if len(args) == 9:
        worker_id, chunk_models_list, header_lines, model_start, model_end, tatu_path, omp_threads, use_arb_freq, feature_tilted = args
    else:
        worker_id, chunk_models_list, header_lines, model_start, model_end, tatu_path, omp_threads = args
        use_arb_freq = 0
        feature_tilted = None  # None ou dict {'use_tilted': 1, 'configs': [(beta, phi), ...]}
    env = {**os.environ, 'OMP_NUM_THREADS': str(omp_threads)}
    
    header_text = "".join(header_lines)

    # Sub-diretório imutável auto-descartável mitigador de conflitos IOs.
    with tempfile.TemporaryDirectory(prefix=f'tatu_w{worker_id}_') as tmpdir:
        tatu_dst = os.path.join(tmpdir, 'tatu.x')
        shutil.copy2(tatu_path, tatu_dst)
        os.chmod(tatu_dst, 0o755)

        model_dst = os.path.join(tmpdir, 'model.in')

        # ITERAÇÃO DA METAMORFOSE: Compilando as Entropias para Texto Físico.
        for i, geo_dict in enumerate(chunk_models_list):
            current_model_index = model_start + i
            
            ncam = geo_dict['n_layers']
            rho_h = geo_dict['rho_h']
            rho_v = geo_dict['rho_v']
            thick = geo_dict['thicknesses']
            
            with open(model_dst, 'w') as f:
                # Topologia invariante.
                f.write(header_text)
                # Configuração Dinâmica da Interface de Domínio da Propagação:
                f.write(str(ncam) + '                !número de camadas\n')
                
                # Desenhando as Condutividades no Espaço Euclidiano `(horizontal, vertical)`
                for j in range(ncam):
                    rh = round(rho_h[j], 2)
                    rv = round(rho_v[j], 2)
                    if rv < rh: rv = rh
                    if j == 0:
                        f.write(f"{rh}    {rv}     !resistividades horizontal e vertical\n")
                    else:
                        f.write(f"{rh}    {rv}\n")
                
                # Desenhando as camadas delimitadoras (`thicknesses_n-2` fatias delimitadoras intrínsecas)
                for j in range(len(thick)):
                    t_val = round(thick[j], 2)
                    if j == 0:
                        f.write(f"{t_val}              !espessuras das n-2 camadas\n")
                    else:
                        f.write(f"{t_val}\n")
                
                # Trave Lógica do Motor Fortran para Inserções Sequenciais de Acumuladores (`current_id max_id`):
                f.write(f"{current_model_index} {model_end}         !modelo atual e o número máximo de modelos\n")
                # F5/F7 — Flags opcionais v8.0 (backward compatible)
                f.write(f"{use_arb_freq}                 !F5: use_arbitrary_freq (0=desabilitado, 1=habilitado)\n")
                _use_tilted = 0
                _tilted_configs = []
                if feature_tilted and feature_tilted.get('use_tilted', 0) == 1:
                    _use_tilted = 1
                    _tilted_configs = feature_tilted.get('configs', [])
                f.write(f"{_use_tilted}                 !F7: use_tilted_antennas (0=desabilitado, 1=habilitado)\n")
                if _use_tilted == 1 and len(_tilted_configs) > 0:
                    f.write(f"{len(_tilted_configs)}                 !F7: n_tilted\n")
                    for _it_idx, (_beta_t, _phi_t) in enumerate(_tilted_configs):
                        f.write(f"{_beta_t}  {_phi_t}            !F7: beta({_it_idx+1}) phi({_it_idx+1})\n")

            # Acionamento Ativo: O código fonte base reprocessa tudo embasado na Matemática Recriada acima.
            # timeout=120s por modelo (>60× o tempo esperado de ~0.06s) previne deadlock
            # de threads OpenMP em casos de divergência numérica ou falha de hardware.
            subprocess.run([tatu_dst], cwd=tmpdir, env=env,
                           capture_output=True, check=True, timeout=120)

        original_dir = os.getcwd() 
        results_dat = []
        results_out = []

        # Salvamento Parametrizado. Protegendo as Saídas Brutas movendo-as prefixadas perante sobreposição:
        for f in Path(tmpdir).glob('*.dat'):
            new_name = f'w{worker_id}_{f.name}'
            shutil.move(str(f), os.path.join(original_dir, new_name))
            results_dat.append((f.name, new_name))
            
        for f in Path(tmpdir).glob('*.out'):
            new_name = f'w{worker_id}_{f.name}'
            shutil.move(str(f), os.path.join(original_dir, new_name))
            results_out.append((f.name, new_name))

        return worker_id, model_start, model_end, results_dat, results_out


# ==============================================================================
# ALGORITMO MESTRE (ENTRYPOINT)
# ==============================================================================
def main():
    """
    Função Primordial que estabelece a Configuração Modular Dinâmica via Terminal.
    Faz parsing lógico de dependências e distribui o pipeline estocástico OMP, lidando com fusões e post-runs.
    """
    parser = argparse.ArgumentParser(description='Geosteering AI Simulator Orchestrator - Modulação Altamente Paralela e Entrópica.')
    parser.add_argument('--models', type=int, default=1000, help='Teto-alvo (Volumetria) Total de modelagens sintéticas geradas no Batch.')
    parser.add_argument('--workers', type=int, default=4, help='Paralelização Assíncrona Total (Núcleos virtuais simultâneos alocados p/ Sandbox).')
    parser.add_argument('--omp-threads', type=int, default=2, help='Granularidade do Solver: Threads OpenMP delegadas por processo Worker.')
    parser.add_argument('--tatu', type=str, default='./tatu.x', help='Endereço Relativo do binário C++/Fortran (`tatu.x`).')
    parser.add_argument('--model-in', type=str, default='./model.in', help='Endereço Relativo do esqueleto descritor Topológico (`model.in`).')
    parser.add_argument('--no-concat', action='store_true', help='By-pass paramétrico: Preserva os Data-chunks descentralizados, suprimindo o Merge Binário.')
    # F5/F7 — Feature flags v8.0
    parser.add_argument('--use-arb-freq', type=int, default=0, choices=[0, 1],
                        help='F5: Frequências arbitrárias (0=desabilitado, 1=habilitado). Default: 0.')
    parser.add_argument('--use-tilted', type=int, default=0, choices=[0, 1],
                        help='F7: Antenas inclinadas (0=desabilitado, 1=habilitado). Default: 0.')
    parser.add_argument('--tilted-configs', type=str, default='',
                        help='F7: Configurações tilted "beta1,phi1;beta2,phi2" em graus. Ex: "45,0;30,90".')
    
    args = parser.parse_args()

    total = args.models
    n_workers = args.workers
    omp_t = args.omp_threads

    # Validação antecipada de dependências críticas: falhar rápido antes de alocar
    # workers evita erros crípticos dentro do ProcessPoolExecutor.
    tatu_abs = os.path.abspath(args.tatu)
    if not os.path.isfile(tatu_abs):
        print(f"[FATAL] Binário tatu.x não encontrado: {tatu_abs}")
        print(f"        Compile com `make` em {os.path.dirname(tatu_abs)} e tente novamente.")
        sys.exit(1)
    if not os.path.isfile(args.model_in):
        print(f"[FATAL] Arquivo model.in não encontrado: {args.model_in}")
        sys.exit(1)

    # Setup Logístico e Front-End Mapeável
    print("="*80)
    print(" BATCH RUNNER ORQUESTRADOR METAMÓRFICO — GEOSTEERING AI EXPERT SYSTEM")
    print("="*80)
    
    info = parse_model_in(args.model_in)
    if info:
        print("[ANÁLISE DE TOPOLOGIA DE ENGENHARIA] Carregando Header Restrito do Model.in Base:")
        print(f"  -> Quantidade Frequencial  [nf]: {info['nf']} un | Assinaturas {info['freqs']} Hz")
        print(f"  -> Ângulos Ponto Dip     [ndip]: {info['ntheta']} rot | Disposições {info['angles']} Gr")
        print(f"  -> Tratos Emissores-Receptores : {info['nTR']} Pares | Profundas em {info['trs']} mts")
    else:
        print("[WARNING] Arquitetura de Header base corrompida, suprimindo metadados de Tool GUI.")

    header_lines_estaticas = extract_model_in_header(args.model_in)

    # F5/F7 — Parse de feature flags
    use_arb_freq_flag = args.use_arb_freq
    feature_tilted_dict = None
    if args.use_tilted == 1 and args.tilted_configs:
        tilted_pairs = []
        for pair_str in args.tilted_configs.split(';'):
            parts = pair_str.strip().split(',')
            if len(parts) == 2:
                tilted_pairs.append((float(parts[0]), float(parts[1])))
        if tilted_pairs:
            feature_tilted_dict = {'use_tilted': 1, 'configs': tilted_pairs}
            print(f"[F7] Antenas inclinadas habilitadas: {len(tilted_pairs)} configuração(ões)")
            for idx_t, (b_t, p_t) in enumerate(tilted_pairs):
                print(f"     tilted({idx_t+1}): beta={b_t}° phi={p_t}°")
    if use_arb_freq_flag == 1:
        print(f"[F5] Frequências arbitrárias habilitadas (nf={info['nf'] if info else '?'})")

    # Injeta Criação Numérica e Entidade Monte-Carlo Assíncrona no Fluxo
    todos_modelos_gerados = fabricar_universos_aleatorios(total)
        
    print(f"\n[ENGAJAMENTO ASSÍNCRONO] Alocando Process Pool Paralelo para modelagem de {total} Perfis Estocásticos.")
    print(f"       -> {n_workers} Instâncias Operacionais Sand-Boxes Independentes limitadas a {omp_t} threads C/U.")
    print("-" * 80)

    # Lógica Algorítmica de Distribuição e Empilhamento O(n):
    chunk = total // n_workers
    remainder = total % n_workers
    batches = []
    start = 1
    for w in range(n_workers):
        # Repassa um diferencial de contagem residual para garantir somatórias perfeitas `N == M`
        end = start + chunk - 1 + (1 if w < remainder else 0)
        
        # Desmembra os perfis matemáticos por worker. Slice na Lista Mestre no Range Específico.
        chunk_dict_list = todos_modelos_gerados[start-1 : end]
        batches.append((w, chunk_dict_list, header_lines_estaticas, start, end, tatu_abs, omp_t,
                        use_arb_freq_flag, feature_tilted_dict))
        start = end + 1

    t0 = time.perf_counter()
    chunks_to_concat = {}

    # Pool de Execuções e Fechamento Místico de Contextos Python. Responde Eventos Concluidos em Tempos Ociosos.
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_model_batch, b): b for b in batches}
        for future in as_completed(futures): 
            wid, ms, me, res_dat, res_out = future.result()
            print(f'  [✓] Nó Analítico Worker-{wid}: Compilação Estrutural {ms}-{me} fechada e exportada.')
            
            # Ancoragem Base de Lixo para futura mesclagem indexada:
            for base, chunkf in res_dat + res_out:
                if base not in chunks_to_concat:
                    chunks_to_concat[base] = []
                chunks_to_concat[base].append((wid, chunkf))

    elapsed = time.perf_counter() - t0
    throughput = total / (elapsed or 1) * 3600
    print("-" * 80)
    print(f"[THROUGHPUT MENSURADO] Dispersão Solucionada global em {elapsed:.1f}s — Rendimento: {throughput:.0f} Modelos/Hr")

    # Finalização Pós-Engenharia: Mesclagem Shutil Binária de Acoplamento Limpo.
    if not args.no_concat:
        print("\n[POST-PROCESS OPERATION] Inciando Protocolo de Concatenação Aderente...")
        for basename, chunks in chunks_to_concat.items():
            
            # Reorganiza topografias garantindo Profundidades Lineares Contínuas no Array.
            chunks.sort(key=lambda x: x[0]) 
            final_path = os.path.join(os.getcwd(), basename)
            if os.path.exists(final_path):
                 os.remove(final_path) 
            
            print(f"  -> Calibrando merge de {len(chunks)} sub-sinalizações particionais rumo ao Arquivo Base: {basename}")
            with open(final_path, 'wb') as outfile: 
                # Bypass Completo Nativo em C do Modulo Shutil Operacional de IO garantindo FileBlock Swap Memory sem perdas
                for wid, chunkf in chunks:
                    chunk_path = os.path.join(os.getcwd(), chunkf)
                    with open(chunk_path, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
                    os.remove(chunk_path) 
            
            # Barreira Protetora Numérica Restrita (Validador Memmap):
            if basename.endswith('.dat'):
                validar_integridade(final_path)
    else:
        print("\n[POST-PROCESS OPERATION] Sub-rotina de Supressão ativada (By-Pass Merge). Elementos Isolados resguardados.")
    
    print("="*80)
    print("Encerramento Executável com Resolução Sucesso. Datasets Inteligentes Subsequentes Disponíveis!")

if __name__ == '__main__':
    main()
