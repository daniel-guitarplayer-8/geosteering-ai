# V versão de um programa que gera modelos aleatórios de camadas geoelétricas horizontais. 
# Nesta versão, os modelos são gerados totalmente pelo método Sobol de Quasi-Monte Carlo
import numpy as np
# import pandas as pd         #usado apenas para teste de uso
import subprocess as sub    #permite executar o programa fortran, criar e acessar arquivos sobre o código de lá
import time                 #usado para apresentar o tempo de processamento
import os
from scipy.stats import qmc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#os.environ["OMP_NUM_THREADS"] = "12"

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def log_transform_2(samples, min_val, max_val):
    """Transforma amostras em [0,1] para escala logarítmica [min_val, max_val]"""
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    return np.exp(log_min + samples * (log_max - log_min))

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def uniform_transform_2(samples, min_val, max_val):
    """Transforma amostras em [0,1] para escala linear [min_val, max_val]"""
    return min_val + samples * (max_val - min_val)

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def numeros_aleatorios_soma_constante_dirichlet(n, a1, an, k):
    # Gera n números aleatórios que somam 1 usando a distribuição Dirichlet
    numeros = np.random.dirichlet(np.ones(n))
    
    # Escala os números para o intervalo [a1, an] e ajusta a soma para k
    numeros = a1 + (an - a1) * numeros
    fator_escala = k / sum(numeros)
    numeros *= fator_escala
    # Arredonda para duas casas decimais
    for i in range(n):
        numeros[i] = round(numeros[i], 2)
    return numeros

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def add_gaussian_noise_2(data, std_dev = 0.05):
    """Adiciona ruído gaussiano multiplicativo (ex: 5% de erro relativo)."""
    return data * (1 + np.random.normal(0, std_dev, size = data.shape))

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def add_lognormal_noise_2(data, sigma = 0.1):
    """Adiciona ruído log-normal (útil para variáveis positivas)."""
    noise = np.random.lognormal(mean = 0, sigma = sigma, size = data.shape)
    return data * noise

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def generate_thicknesses_2(internal_layers, sobol_portion, total_depth = 100, min_thickness = 0.3):
    """
    Gera espessuras com base em sequência de Sobol fornecida, garantindo:
    - Espessuras mínimas (ex: 0.5m).
    - Soma total = total_depth.
    sobol_portion: Uma porção da amostra Sobol multidimensional (já deve ter internal_layers-1 pontos).
    """
    if internal_layers <= 0:
        return np.array([])
    
    # Usa a porção Sobol fornecida para as amostras e ordena
    samples = np.sort(sobol_portion.flatten()) # Garante que seja 1D
    
    # Adiciona limites e calcula espessuras iniciais (método de quebra de bastão)
    points = np.concatenate(([0], samples, [1]))
    thicknesses = np.diff(points) * total_depth
    
    # Garante espessura mínima (ajuste iterativo)
    for i in range(10):  # Limite de iterações para evitar loop infinito
        idx_small = np.where(thicknesses < min_thickness)[0]
        if len(idx_small) == 0:
            break  # Sai se todas as espessuras estão acima do mínimo
        excess = (min_thickness - thicknesses[idx_small]).sum()
        thicknesses[idx_small] = min_thickness
        # Distribui o excesso entre as outras camadas
        idx_rest = np.where(thicknesses >= min_thickness)[0]
        if len(idx_rest) > 0:
            thicknesses[idx_rest] -= excess / len(idx_rest)
        else:
            # Se todas as camadas estão abaixo do mínimo, ajusta diretamente
            thicknesses += (total_depth - thicknesses.sum()) / internal_layers
    
    # Reajusta para garantir soma total
    thicknesses = thicknesses / thicknesses.sum() * total_depth
    return thicknesses

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def generate_thick_thicknesses_2(internal_layers, sobol_portion, total_depth = 100, min_thickness = 10, p_thick = 0.7):
    """
    Gera espessuras com probabilidade aumentada de camadas grossas, usando Sobol fornecido.
    - p_thick: Probabilidade de forçar uma camada a ser grossa (> 10m).
    sobol_portion: Uma porção da amostra Sobol multidimensional (já deve ter internal_layers-1 pontos).
    """
    if internal_layers <= 0:
        return []
    
    # Usa a porção Sobol fornecida
    samples = np.sort(sobol_portion.flatten())
    points = np.concatenate(([0], samples, [1]))
    thicknesses = np.diff(points) * total_depth
    
    # Perturbação condicional para camadas grossas
    for i in range(internal_layers):
        if np.random.rand() < p_thick: # Nota: np.random.rand() ainda é usado para a probabilidade condicional
            thicknesses[i] = np.random.uniform(15, 50)  # Camadas grossas (15–50m)
    
    # Garante espessura mínima e soma total
    thicknesses = np.clip(thicknesses, min_thickness, None)
    thicknesses = thicknesses / thicknesses.sum() * total_depth
    return thicknesses

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def generate_thin_thicknesses_2(internal_layers, sobol_portion, total_depth = 100, min_thickness = 0.5, p_thin = 0.4):
    """
    Gera espessuras com probabilidade aumentada de camadas finas, usando Sobol fornecido.
    - min_thickness: Espessura mínima absoluta (ex: 0.5m).
    - p_thin: Probabilidade de forçar uma camada a ser fina (mas não menor que min_thickness).
    sobol_portion: Uma porção da amostra Sobol multidimensional (já deve ter internal_layers-1 pontos).
    """
    if internal_layers <= 0:
        return []
    
    # Usa a porção Sobol fornecida
    samples = np.sort(sobol_portion.flatten())
    points = np.concatenate(([0], samples, [1]))
    thicknesses = np.diff(points) * total_depth

    # Perturbação condicional para camadas finas
    for i in range(internal_layers):
        if np.random.rand() < p_thin: # Nota: np.random.rand() ainda é usado para a probabilidade condicional
            thicknesses[i] = min_thickness * np.random.uniform(0.9, 1.1)
    
    # Garante espessura mínima e máxima absoluta
    thicknesses = np.clip(thicknesses, min_thickness, None)
    
    # Reajusta a soma total
    thicknesses = thicknesses / thicknesses.sum() * total_depth
    return thicknesses

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def conditional_rho_h_sampling_core_2(rho_h, contrast_threshold = 2, p_high = 0.3, infratio = 3.8e-3, supratio = 260,
                                    rhoh_min = 0.1, rhoh_max = 1500):
    """
    Força contrastes extremos entre camadas adjacentes.
    - contrast_threshold: Contraste mínimo para considerar uma inversão.
    - p_high: Probabilidade de forçar um contraste alto.
    """
    rho_h = rho_h.copy()
    for i in range(1, len(rho_h)):
        current_ratio = max(rho_h[i-1], rho_h[i]) / min(rho_h[i-1], rho_h[i])
        if current_ratio < contrast_threshold:
            if np.random.rand() < p_high:
                if np.random.rand() < 0.5:
                    rhoold = rho_h[i]
                    rho_h[i] = rho_h[i-1] * np.random.choice([infratio, supratio])
                    if rho_h[i] < rhoh_min or rho_h[i] > rhoh_max: rho_h[i] = rhoold
                else:
                    rhoold = rho_h[i-1]
                    rho_h[i-1] = rho_h[i] * np.random.choice([infratio, supratio])
                    if rho_h[i-1] < rhoh_min or rho_h[i-1] > rhoh_max: rho_h[i-1] = rhoold
    for j in range(len(rho_h)):
        if rho_h[j] < rhoh_min: rho_h[j] = rhoh_min
        if rho_h[j] > rhoh_max: rho_h[j] = rhoh_max
    return rho_h

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def conditional_rho_h_with_thickness_2(rho_h, thicknesses, thick_threshold = 10, p_high = 0.3, 
                                     rhohminb = 0.5, rhohmina = 10., rhohmaxb = 10., rhohmaxa = 100.):
    """
    Força resistividades altas/baixas em camadas grossas.
    - thick_threshold: Espessura mínima para considerar uma camada grossa (ex: 10m).
    - p_high: Probabilidade de forçar resistividade alta/baixa em camadas grossas.
    """
    rho_perturbed = rho_h.copy()
    for i in range(len(thicknesses)):
        if thicknesses[i] > thick_threshold and np.random.rand() < p_high:
            if np.random.rand() < 0.5:
                rho_perturbed[i] = np.random.uniform(rhohmina, rhohmaxa)
            else:
                rho_perturbed[i] = np.random.uniform(rhohminb, rhohmaxb)
    return rho_perturbed

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def correlated_lambda_sampling_core_2(rho_h, sobol_sample_portion, lambda_min, lambda_max, rho_threshold = 100,
    lambda_low_range = (1.0, 1.1), lambda_high_range = None):
    # lambda_low_range = (1.0, 1.5), lambda_high_range = None):
    """
    Gera λ com base na resistividade horizontal, garantindo que os intervalos estejam dentro de `lambda_min` e `lambda_max`.
    sobol_sample_portion: Porção da amostra Sobol para gerar lambda.
    """
    lambda_vals = np.zeros_like(rho_h, dtype=float)

    if lambda_high_range is None:
        # lambda_high_range = (1.5, lambda_max)
        lambda_high_range = (1.1, lambda_max)
    else:
        lambda_high_range = (
            max(lambda_high_range[0], lambda_min),
            min(lambda_high_range[1], lambda_max)
        )
    
    lambda_low_range = (
        max(lambda_low_range[0], lambda_min),
        min(lambda_low_range[1], lambda_max)
    )

    # Use a porção Sobol para gerar os valores de lambda
    # É importante que sobol_sample_portion tenha o mesmo tamanho que rho_h
    if len(sobol_sample_portion) != len(rho_h):
        raise ValueError("sobol_sample_portion must have the same length as rho_h.")

    for j in range(len(rho_h)):
        if rho_h[j] > rho_threshold:
            low, high = lambda_high_range
        else:
            low, high = lambda_low_range
        lambda_vals[j] = low + (high - low) * sobol_sample_portion[j]
    
    lambda_vals = np.clip(lambda_vals, lambda_min, lambda_max)
    return lambda_vals

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def extract_data(models):
    """Extrai dados de uma lista de modelos."""
    all_rho_h = []
    all_rho_v = []
    all_lambda = []
    all_thicknesses = []
    all_n_layers = []

    for model in models:
        all_n_layers.append(model['n_layers'])
        all_rho_h.extend(model['rho_h'])
        all_rho_v.extend(model['rho_v'])
        all_lambda.extend(model['lambda'])
        all_thicknesses.extend(model['thicknesses'])

    return {
        'rho_h': np.array(all_rho_h),
        'rho_v': np.array(all_rho_v),
        'lambda': np.array(all_lambda),
        'thicknesses': np.array(all_thicknesses),
        'n_layers': np.array(all_n_layers)
            }

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def plot_model(model_idx, modelos):
    """Plota um modelo específico com resistividades e espessuras."""

    # Habilita o suporte a LaTeX
    plt.rcParams.update({
        "text.usetex": True,                      # Habilita o uso de LaTeX para renderizar texto
        "font.family": "serif",                   # Usa uma fonte serifada (compatível com LaTeX)
        "font.serif": ["Computer Modern Roman"],  # Fonte padrão do LaTeX
        # Configurando o tamanho da fonte globalmente
        "font.size": 14,                          # Tamanho padrão para todos os textos
        "axes.titlesize": 24,                     # Tamanho do título
        "axes.labelsize": 22,                     # Tamanho dos rótulos dos eixos
        "xtick.labelsize": 16,                    # Tamanho dos ticks do eixo x
        "ytick.labelsize": 16,                    # Tamanho dos ticks do eixo y
        "legend.fontsize": 18                     # Tamanho da fonte da legenda
        })

    # Extrai dados do modelo específico
    model = modelos[model_idx]
    ncam = model['n_layers']
    rho_h = model['rho_h']
    rho_v = model['rho_v']
    lambda_vals = model['lambda']
    thicknesses = model['thicknesses']
    
    # Camadas (inclui infinitas)
    layers = np.arange(ncam)
    labels = [f"{i+1}" for i in range(ncam-1,-1,-1)]
    
    # Espessuras (primeira e última infinitas, não plotadas)
    thicknesses_plot = np.zeros(ncam)
    if len(thicknesses) > 0:
        thicknesses_plot[1:-1] = thicknesses  # Preenche camadas internas
        totalesp = np.sum(thicknesses_plot)
    
    # Criação do gráfico
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    
    # Plotagem das espessuras
    axes[0].barh(layers, thicknesses_plot, height=0.8, color='skyblue', edgecolor='black')
    axes[0].set_title(rf"$Espessuras\ (total = {totalesp:.2f}\ m)$")
    axes[0].set_xlabel(r"$m$")
    axes[0].set_ylabel("Camadas")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_yticks(layers)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()  # Primeira camada no topo
    
    # Plotagem de resistividade horizontal
    axes[1].barh(layers, rho_h, height=0.8, color='darkgreen', edgecolor='black')
    axes[1].set_title(r"$\rho_h$")
    axes[1].set_xlabel(r"Resistividade $(\Omega\cdot m)$")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_xscale('log')
    axes[1].invert_yaxis()  # Primeira camada no topo
    
    # Plotagem de resistividade vertical
    axes[2].barh(layers, rho_v, height=0.8, color='darkorange', edgecolor='black')
    axes[2].set_title(r"$\rho_v$")
    axes[2].set_xlabel(r"$\Omega\cdot m$")
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].set_xscale('log')
    axes[2].invert_yaxis()  # Primeira camada no topo
    
    # Plotagem do coeficiente de anisotropia
    axes[3].barh(layers, lambda_vals, height=0.8, color='purple', edgecolor='black')
    axes[3].set_title(r"$\lambda$")
    # axes[3].set_xlabel(r"Valor de $\lambda$")
    axes[3].grid(True, linestyle='--', alpha=0.7)
    axes[3].invert_yaxis()  # Primeira camada no topo
    
    plt.suptitle(f"Modelo {model_idx+1} | Total de Camadas: {ncam}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def plots2myscenario(nmodels, models, mystr = 'Modelos Base'):
    # Extrair todos os valores de ρ_h, ρ_v, λ e espessuras
    all_n_cam = np.zeros(nmodels)
    for i in range(nmodels):
        all_n_cam[i] = models[i]['n_layers']
    all_rho_h = np.concatenate([m['rho_h'] for m in models])
    all_rho_v = np.concatenate([m['rho_v'] for m in models])
    all_lambda= np.concatenate([m['lambda'] for m in models])
    all_thick = np.concatenate([m['thicknesses'] for m in models])
    print('-------------------------------------------------------------------------')
    print('Valores limites:')
    print('{} é o valor mínimo de resistividade horizontal'.format(np.min(all_rho_h)))
    print('{} é o valor máximo de resistividade horizontal'.format(np.max(all_rho_h)))
    print('{} é o valor mínimo de resistividade vertical'.format(np.min(all_rho_v)))
    print('{} é o valor máximo de resistividade vertical'.format(np.max(all_rho_v)))
    print('{} é o valor mínimo de espessura'.format(np.min(all_thick)))
    print('{} é o valor máximo de espessura'.format(np.max(all_thick)))
    print('{} é o valor mínimo de lambda'.format(np.min(all_lambda)))
    print('{} é o valor máximo de lambda'.format(np.max(all_lambda)))
    print('-------------------------------------------------------------------------')
    #-------------------------------------------------------------------------------------------------------------
    # Habilita o suporte a LaTeX
    plt.rcParams.update({
        "text.usetex": True,                      # Habilita o uso de LaTeX para renderizar texto
        "font.family": "serif",                   # Usa uma fonte serifada (compatível com LaTeX)
        "font.serif": ["Computer Modern Roman"],  # Fonte padrão do LaTeX
        # Configurando o tamanho da fonte globalmente
        "font.size": 14,                          # Tamanho padrão para todos os textos
        "axes.titlesize": 24,                     # Tamanho do título
        "axes.labelsize": 22,                     # Tamanho dos rótulos dos eixos
        "xtick.labelsize": 16,                    # Tamanho dos ticks do eixo x
        "ytick.labelsize": 16,                    # Tamanho dos ticks do eixo y
        "legend.fontsize": 18                     # Tamanho da fonte da legenda
        })
    #-------------------------------------------------------------------------------------------------------------
    # --- Gráfico 1: Distribuição do número de camadas ---
    plt.figure(figsize = (10, 5))
    plt.hist(all_n_cam, bins = 50, edgecolor = 'black', alpha = 0.7)
    plt.xlabel('Números de camadas')
    plt.ylabel('Frequência')
    plt.title('Distribuição do número de camadas em {0} {1}'.format(nmodels,mystr))
    plt.grid(True, linestyle = '--', alpha = 0.3)
    plt.show()

    # --- Gráfico 1: Distribuição de ρ_h (escala logarítmica) ---
    plt.figure(figsize = (10, 5))
    plt.hist(np.log10(all_rho_h), bins = 50, edgecolor = 'black', alpha = 0.7)
    plt.xlabel(r'$\log_{10}[\rho_h]\ (\Omega\ m)$')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Resistividade Horizontal em  {0} {1}'.format(nmodels,mystr))
    plt.grid(True, linestyle = '--', alpha = 0.3)
    plt.show()

    # --- Gráfico 2: Distribuição de ρ_v (escala logarítmica) ---
    plt.figure(figsize = (10, 5))
    plt.hist(np.log10(all_rho_v), bins = 50, edgecolor = 'black', color = 'orange', alpha = 0.7)
    plt.xlabel(r'$\log_{10}[\rho_v]\ (\Omega\ m)$')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Resistividade Vertical em  {0} {1}'.format(nmodels,mystr))
    plt.grid(True, linestyle = '--', alpha = 0.3)
    plt.show()

    # --- Gráfico 3: Distribuição de λ ---
    plt.figure(figsize=(10, 5))
    plt.hist(all_lambda, bins = 50, edgecolor = 'black', color = 'green', alpha = 0.7)
    # plt.axvline(x = 1.0, color = 'red', linestyle = '--', label = r'$\lambda$ mínimo')
    # plt.axvline(x = np.sqrt(2), color = 'blue', linestyle = '-.', label = r'$\lambda$ máximo')
    plt.xlabel(r'Coeficiente de Anisotropia $\lambda$')
    plt.ylabel('Frequência')
    plt.title('Distribuição do Coeficiente de Anisotropia em  {0} {1}'.format(nmodels,mystr))
    # plt.legend()
    # # Definindo xticks específicos
    # yticks_escolhidos = [1, np.sqrt(2)]  # Valores personalizados
    # plt.xticks(yticks_escolhidos)
    plt.grid(True, linestyle = '--', alpha = 0.3)
    plt.show()

    # --- Gráfico 4: Distribuição de espessuras ---
    plt.figure(figsize = (10, 5))
    plt.hist(all_thick, bins = 50, edgecolor = 'black', alpha = 0.7)
    plt.xlim(0,50)
    # plt.xlim(0,1.1*max(all_thick))
    plt.xlabel('h [m]')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Espessuras em  {0} {1}'.format(nmodels,mystr))
    plt.grid(True, linestyle = '--', alpha = 0.3)
    plt.show()

    # Exibe um modelo aleatório
    plot_model(np.random.randint(nmodels), models)
    # plot_model(mymodel, models)

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def baseline_empirical_2(tji = 100.,
                       rho_h_min = 0.5, 
                       rho_h_max = 1000.,
                       rhohdistr = 'loguni',
                       lambda_min = 1., 
                       lambda_max = np.sqrt(2),
                       min_thickness = 1.,
                       nmodels = 18000):
    ncams = np.arange(3, 31).tolist()
    pesosncam = [
        0.005, 0.01, 0.015, 
        0.02, 0.025, 0.03, 0.035, 0.04, 0.04, 0.04, 0.045,
        0.045, 0.045, 0.045, 0.05, 0.045, 0.045, 0.045, 0.045, 0.045, 0.04,
        0.04, 0.04, 0.04, 0.035, 0.035, 0.03, 0.025
        ]
    pesosncam = np.array(pesosncam) / np.sum(pesosncam)

    models = []
    for i in range(nmodels):
        ncam = np.random.choice(ncams, p = pesosncam)
        ncamint = ncam - 2 # Camadas internas (finitas)

        num_sobol_for_thicknesses = max(0, ncamint - 1)
        # Determinar a dimensão total para o Sobol para este modelo
        dim_total_model = ncam + ncam + num_sobol_for_thicknesses
        
        # Gerar UMA ÚNICA AMOSTRA SOBOL para todas as variáveis deste modelo
        sobol_sampler_modelo = qmc.Sobol(d=dim_total_model, scramble=True)
        amostra_unica_sobol = sobol_sampler_modelo.random(1).flatten()

        # Fatiar a amostra única para as variáveis
        current_idx = 0
        rho_h_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam

        lambda_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam

        thicknesses_sobol_portion = amostra_unica_sobol[current_idx : current_idx + num_sobol_for_thicknesses]

        # Transformação para resistividade horizontal
        if rhohdistr == 'unifor':
            rhoh = uniform_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        else:
            rhoh = log_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        
        # Transformação para coeficiente de anisotropia
        lambdas = uniform_transform_2(lambda_sobol_portion, lambda_min, lambda_max)
        
        # Cálculo das resistividades verticais
        rhov = (lambdas**2) * rhoh

        thicknesses = generate_thicknesses_2(ncamint, thicknesses_sobol_portion, total_depth = tji, min_thickness = min_thickness)

        models.append({
            'n_layers': ncam,
            'rho_h': rhoh.tolist(),
            'lambda': lambdas.tolist(),
            'rho_v': rhov.tolist(),
            'thicknesses': thicknesses.tolist()
            })
    return models

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def baseline_ncamuniform_2(tji = 100.,
                      ncamin = 3,
                      ncamax = 81, # ncamax é exclusivo, então 81 significa até 80 camadas
                      min_thickness = 0.5,
                      rho_h_min = 0.1, 
                      rho_h_max = 1400., 
                      rhohdistr = 'loguni',
                      lambda_min = 1., 
                      lambda_max = np.sqrt(2), #1.7, #> np.sqrt(2),
                      nmodels = 9000):
    models = []
    for i in range(nmodels):
        ncam = np.random.randint(ncamin, ncamax)
        ncamint = ncam - 2

        num_sobol_for_thicknesses = max(0, ncamint - 1)
        # Determinar a dimensão total para o Sobol para este modelo
        dim_total_model = ncam + ncam + num_sobol_for_thicknesses
        
        # Gerar UMA ÚNICA AMOSTRA SOBOL para todas as variáveis deste modelo
        sobol_sampler_modelo = qmc.Sobol(d=dim_total_model, scramble=True)
        amostra_unica_sobol = sobol_sampler_modelo.random(1).flatten()

        # Fatiar a amostra única para as variáveis
        current_idx = 0
        rho_h_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam

        lambda_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam

        thicknesses_sobol_portion = amostra_unica_sobol[current_idx : current_idx + num_sobol_for_thicknesses]

        # Transformação para resistividade horizontal
        if rhohdistr == 'unifor':
            rhoh = uniform_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        else:
            rhoh = log_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        
        # Transformação para coeficiente de anisotropia
        lambdas = uniform_transform_2(lambda_sobol_portion, lambda_min, lambda_max)
        
        # Cálculo das resistividades verticais
        rhov = lambdas**2 * rhoh

        # Geração de espessuras usando a porção Sobol para elas
        thicknesses = generate_thicknesses_2(ncamint, thicknesses_sobol_portion, total_depth = tji, min_thickness = min_thickness)

        models.append({
            'n_layers': ncam,
            'rho_h': rhoh.tolist(),
            'lambda': lambdas.tolist(),
            'rho_v': rhov.tolist(),
            'thicknesses': thicknesses.tolist()
            })
    return models

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def baseline_thick_thicknesses_2(tji = 100.,
                       ncamin = 3,
                       ncamax = 15,
                       rho_h_min = 0.1, 
                       rho_h_max = 1400, #2000.,
                       rhohdistr = 'loguni',
                       rhoh_min_b = 0.1, 
                       rhoh_min_a = 100., 
                       rhoh_max_b = 50., 
                       rhoh_max_a = 1800,    #2000,
                       lambda_min = 1., 
                       lambda_max = np.sqrt(2),
                       nmodels = 9000):
    models = []
    for i in range(nmodels):
        ncam = np.random.randint(ncamin, ncamax)
        ncamint = ncam - 2

        num_sobol_for_thicknesses = max(0, ncamint - 1)
        # Determinar a dimensão total para o Sobol para este modelo
        dim_total_model = ncam + ncam + num_sobol_for_thicknesses
        
        # Gerar UMA ÚNICA AMOSTRA SOBOL para todas as variáveis deste modelo
        sobol_sampler_modelo = qmc.Sobol(d=dim_total_model, scramble=True)
        amostra_unica_sobol = sobol_sampler_modelo.random(1).flatten()

        # Fatiar a amostra única para as variáveis
        current_idx = 0
        rho_h_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam
        lambda_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam
        thicknesses_sobol_portion = amostra_unica_sobol[current_idx : current_idx + num_sobol_for_thicknesses]

        # Geração de espessuras usando a porção Sobol para elas
        thicknesses = generate_thick_thicknesses_2(ncamint, thicknesses_sobol_portion, total_depth = tji, min_thickness = 10)
        
        # Transformação para resistividade horizontal
        if rhohdistr == 'unifor':
            rhoh = uniform_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        else:
            rhoh = log_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        rhoh = conditional_rho_h_with_thickness_2(rhoh, thicknesses, 
                                                rhohminb = rhoh_min_b, rhohmina = rhoh_min_a, 
                                                rhohmaxb = rhoh_max_b, rhohmaxa = rhoh_max_a)

        # Transformação para coeficiente de anisotropia
        lambdas = uniform_transform_2(lambda_sobol_portion, lambda_min, lambda_max)
        
        # Cálculo das resistividades verticais
        rhov = lambdas**2 * rhoh

        models.append({
            'n_layers': ncam,
            'rho_h': rhoh.tolist(),
            'lambda': lambdas.tolist(),
            'rho_v': rhov.tolist(),
            'thicknesses': thicknesses.tolist()
            })
    return models

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def unfriendly_empirical_2(tji = 100.,
                        rho_h_min = 0.1,
                        rho_h_max = 1400,    #2000.,
                        rhohdistr = 'loguni',
                        lambda_min = 1.0,
                        lambda_max = np.sqrt(2),   #np.sqrt(3),
                        nmodels = 12000,
                        contraste = 5, #10,
                        inf_ratio = 1/150,    #1/500,
                        sup_ratio = 150,    #500,
                        rho_limite = 100,
                        p_contraste = 0.5,  #0.7,
                        p_camfina = 0.6,    #0.7,
                        min_thickness = 0.2):    #0.1):
    ncams = np.arange(3, 31).tolist()
    pesosncam = [
        0.005, 0.005, 0.01,
        0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
        0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.11, 0.10, 0.09,
        0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02
    ]
    pesosncam = np.array(pesosncam) / np.sum(pesosncam)

    models = []
    for _ in range(nmodels):
        ncam = np.random.choice(ncams, p = pesosncam)
        ncamint = ncam - 2

        num_sobol_for_thicknesses = max(0, ncamint - 1)
        # Determinar a dimensão total para o Sobol para este modelo
        dim_total_model = ncam + ncam + num_sobol_for_thicknesses
        
        # Gerar UMA ÚNICA AMOSTRA SOBOL para todas as variáveis deste modelo
        sobol_sampler_modelo = qmc.Sobol(d=dim_total_model, scramble=True)
        amostra_unica_sobol = sobol_sampler_modelo.random(1).flatten()

        # Fatiar a amostra única para as variáveis
        current_idx = 0
        rho_h_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam
        lambda_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam
        thicknesses_sobol_portion = amostra_unica_sobol[current_idx : current_idx + num_sobol_for_thicknesses]

        # Resistividades horizontais
        if rhohdistr == 'unifor':
            rho_h = uniform_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        else:
            rho_h = log_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        rho_h = conditional_rho_h_sampling_core_2(rho_h, contrast_threshold = contraste, p_high = p_contraste,
                                                infratio = inf_ratio, supratio = sup_ratio,
                                                rhoh_min = rho_h_min, rhoh_max = rho_h_max) # Passando limites para clipping interno

        # Coeficiente de anisotropia
        lambda_vals = correlated_lambda_sampling_core_2(rho_h, lambda_sobol_portion, lambda_min, lambda_max, 
                                                      rho_threshold = rho_limite)

        # Resistividade vertical
        rho_v = lambda_vals**2 * rho_h
        for i in range(ncam):
            if rho_v[i] < rho_h_min * 2: rho_v[i] = rho_h_min * 2 # Garante um mínimo razoável para rho_v
            if rho_v[i] > rho_h_max * 2: rho_v[i] = rho_h_max * 2 # Garante um máximo razoável para rho_v

        # Espessuras
        thicknesses = generate_thin_thicknesses_2(ncamint, thicknesses_sobol_portion, total_depth = tji, 
                                                   min_thickness = min_thickness, p_thin = p_camfina)

        models.append({
            'n_layers': ncam,
            'rho_h': rho_h.tolist(),
            'lambda': lambda_vals.tolist(),
            'rho_v': rho_v.tolist(),
            'thicknesses': thicknesses.tolist()
        })
    return models

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def unfriendly_noisy_2(tji = 100.,
                      rho_h_min = 0.1,
                      rho_h_max = 1400,  #2000.,
                      rhohdistr = 'loguni', # Changed from 'uni' to 'loguni' for consistency
                      lambda_min = 1.0,
                      lambda_max = np.sqrt(2), #np.sqrt(3),
                      nmodels = 12000,
                      contraste = 5, #10,
                      inf_ratio = 1/150,  #1/500,
                      sup_ratio = 150,  #500,
                      rho_limite = 100,
                      p_contraste = 0.5,    #0.7,
                      p_camfina = 0.6,  #0.7,
                      min_thickness = 0.2,  #0.1,
                      noise_std_rho = 0.05,
                      noise_std_lambda = 0.02,
                      noise_std_thickness = 0.03):
    ncams = np.arange(3, 31).tolist()
    pesosncam = [
        0.005, 0.005, 0.01,
        0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
        0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.11, 0.10, 0.09,
        0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02
    ]
    pesosncam = np.array(pesosncam) / np.sum(pesosncam)

    models = []
    for _ in range(nmodels):
        ncam = np.random.choice(ncams, p = pesosncam)
        ncamint = ncam - 2

        num_sobol_for_thicknesses = max(0, ncamint - 1)
        # Determinar a dimensão total para o Sobol para este modelo
        dim_total_model = ncam + ncam + num_sobol_for_thicknesses
        
        # Gerar UMA ÚNICA AMOSTRA SOBOL para todas as variáveis deste modelo
        sobol_sampler_modelo = qmc.Sobol(d=dim_total_model, scramble=True)
        amostra_unica_sobol = sobol_sampler_modelo.random(1).flatten()

        # Fatiar a amostra única para as variáveis
        current_idx = 0
        rho_h_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam
        lambda_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam
        thicknesses_sobol_portion = amostra_unica_sobol[current_idx : current_idx + num_sobol_for_thicknesses]

        # Resistividades horizontais (geração limpa)
        if rhohdistr == 'unifor':
            rho_h_clean = uniform_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        else:
            rho_h_clean = log_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        rho_h_clean = conditional_rho_h_sampling_core_2(rho_h_clean, contrast_threshold = contraste, p_high = p_contraste,
                                                      infratio = inf_ratio, supratio = sup_ratio,
                                                      rhoh_min = rho_h_min, rhoh_max = rho_h_max)
        rho_h_noisy = add_gaussian_noise_2(rho_h_clean, std_dev = noise_std_rho)
        rho_h_noisy = np.clip(rho_h_noisy, rho_h_min, rho_h_max) # Clipe final para garantir limites

        # Coeficiente de anisotropia (geração limpa)
        lambda_vals_clean = correlated_lambda_sampling_core_2(rho_h_clean, lambda_sobol_portion, lambda_min, lambda_max, 
                                                            rho_threshold = rho_limite)
        lambda_vals_noisy = add_gaussian_noise_2(lambda_vals_clean, std_dev = noise_std_lambda)
        lambda_vals_noisy = np.clip(lambda_vals_noisy, lambda_min, lambda_max) # Clipe final

        # Resistividade vertical
        rho_v_clean = (lambda_vals_clean**2) * rho_h_clean
        rho_v_noisy = (lambda_vals_noisy**2) * rho_h_noisy # Ruído propagado

        for i in range(ncam):
            if rho_v_clean[i] < rho_h_min * 2: rho_v_clean[i] = rho_h_min * 2
            if rho_v_clean[i] > rho_h_max * 2: rho_v_clean[i] = rho_h_max * 2 # Nova garantia de limite superior
            
            if rho_v_noisy[i] < rho_h_min * 2: rho_v_noisy[i] = rho_h_min * 2
            if rho_v_noisy[i] > rho_h_max * 2: rho_v_noisy[i] = rho_h_max * 2 # Nova garantia de limite superior


        # Espessuras (geração limpa)
        thicknesses_clean = generate_thin_thicknesses_2(ncamint, thicknesses_sobol_portion, total_depth = tji, 
                                        min_thickness = min_thickness, p_thin = p_camfina)
        thicknesses_noisy = thicknesses_clean * (1 + np.random.normal(0, noise_std_thickness, size = thicknesses_clean.shape))
        thicknesses_noisy = np.clip(thicknesses_noisy, min_thickness, tji) # Clipe final com min_thickness

        models.append({
            'n_layers': ncam,
            'rho_h': rho_h_noisy.tolist(),
            'lambda': lambda_vals_noisy.tolist(),
            'rho_v': rho_v_noisy.tolist(),
            'thicknesses': thicknesses_noisy.tolist()
        })
    return models

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
def generate_pathological_models_2(tji = 100.,
                                 ncamin = 3,
                                 ncamax = 28, # Manter um bom range de camadas
                                 rho_h_min = 0.05,
                                 rho_h_max = 1500.,
                                 rhohdistr = 'loguni',
                                 lambda_min = 1.0,
                                 lambda_max = np.sqrt(2.5), # Usar o limite superior de anisotropia
                                 nmodels = 4500, # Proporção dos modelos patológicos (ex: 15%)
                                 p_total_depth_fill = 0.7, # Probabilidade de NÃO preencher a janela (resulta em última camada grossa)
                                 min_fill_ratio = 0.4, # Mínimo da janela preenchida, ex: 40% de tji
                                 min_thickness_internal = 0.1, # Permitir camadas internas muito finas
                                 contraste = 10,
                                 inf_ratio = 1/150,
                                 sup_ratio = 150,
                                 rho_limite = 1700,
                                 p_contraste = 0.7,
                                 p_camfina = 0.8,
                                 noise_std_rho = 0.07,
                                 noise_std_lambda = 0.03,
                                 noise_std_thickness = 0.04,
                                 p_extreme_semispace_rho = 0.5,
                                 extreme_semispace_rho_range_low = (0.05, 0.5),
                                 extreme_semispace_rho_range_high = (1000, 3000),
                                 p_semispace_anisotropy = 0.4,
                                 semispace_anisotropy_range = (1.5, 2.5)
                                ):
    models = []
    ncams_range = np.arange(ncamin, ncamax + 1).tolist()

    for _ in range(nmodels):
        ncam = np.random.choice(ncams_range)
        ncamint = ncam - 2

        current_total_depth_for_internal = tji
        if np.random.rand() < p_total_depth_fill:
            current_total_depth_for_internal = np.random.uniform(min_fill_ratio * tji, 0.95 * tji)

        height_for_internal_layers_sum = current_total_depth_for_internal

        num_sobol_for_thicknesses = max(0, ncamint - 1)
        dim_total_model = ncam + ncam + num_sobol_for_thicknesses
        
        sobol_sampler_modelo = qmc.Sobol(d=dim_total_model, scramble=True)
        amostra_unica_sobol = sobol_sampler_modelo.random(1).flatten()

        current_idx = 0
        rho_h_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam
        lambda_sobol_portion = amostra_unica_sobol[current_idx : current_idx + ncam]
        current_idx += ncam
        thicknesses_sobol_portion = amostra_unica_sobol[current_idx : current_idx + num_sobol_for_thicknesses]

        # --- Geração de Espessuras ---
        thicknesses = np.array([])
        if ncamint > 0: # Se há camadas internas a serem geradas
            # Gera as espessuras iniciais com base na distribuição Sobol
            thicknesses = generate_thicknesses_2(ncamint, thicknesses_sobol_portion,
                                                 total_depth = height_for_internal_layers_sum,
                                                 min_thickness = min_thickness_internal)
            # Garantir que a soma total seja height_for_internal_layers_sum
            thicknesses = thicknesses / np.sum(thicknesses) * height_for_internal_layers_sum if np.sum(thicknesses) > 1e-9 else np.ones(ncamint) * (height_for_internal_layers_sum / ncamint)

            original_thicknesses_for_p_camfina = thicknesses.copy()

            # >>> LÓGICA para p_camfina ter ROBUSTEZ <<<
            if ncamint > 1: # A lógica de afinar e redistribuir só faz sentido com múltiplas camadas
                for i in range(ncamint): # Itera sobre todas as camadas internas
                    if np.random.rand() < p_camfina:
                        # Tenta forçar a espessura para um valor muito próximo de min_thickness
                        new_thin_thick_val = min_thickness_internal * np.random.uniform(0.9, 1.1)
                        
                        # Garante que a nova espessura não seja menor que o mínimo e não exceda a espessura original
                        thicknesses[i] = np.clip(new_thin_thick_val, min_thickness_internal, original_thicknesses_for_p_camfina[i])

                # Após tentar afinar camadas, re-normalizar o array inteiro para a soma alvo.
                # Isso distribui o "espaço" liberado ou as "necessidades" adicionais.
                if np.sum(thicknesses) > 1e-9:
                    thicknesses = thicknesses / np.sum(thicknesses) * height_for_internal_layers_sum
                else: # Edge case: if all thicknesses ended up zero or extremely small
                    thicknesses = np.ones(ncamint) * (height_for_internal_layers_sum / ncamint)
                    thicknesses = np.clip(thicknesses, min_thickness_internal, None)
                    thicknesses = thicknesses / np.sum(thicknesses) * height_for_internal_layers_sum # Final normal.
            
            # Última garantia de soma total após todas as manipulações
            if np.sum(thicknesses) > 1e-9:
                thicknesses = thicknesses / np.sum(thicknesses) * height_for_internal_layers_sum
            else: # Fallback for extreme cases (all layers very tiny)
                 thicknesses = np.ones(ncamint) * (height_for_internal_layers_sum / ncamint)
                 thicknesses = np.clip(thicknesses, min_thickness_internal, None)
                 thicknesses = thicknesses / np.sum(thicknesses) * height_for_internal_layers_sum # Final normal.

        # --- Resistividades e Anisotropia ---
        if rhohdistr == 'unifor':
            rho_h_clean = uniform_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        else:
            rho_h_clean = log_transform_2(rho_h_sobol_portion, rho_h_min, rho_h_max)
        
        lambda_vals_clean = correlated_lambda_sampling_core_2(rho_h_clean, lambda_sobol_portion, lambda_min, lambda_max, 
                                                            rho_threshold = rho_limite)

        # Aplicar variações extremas nos semiespaços (Camadas 0 e ncam-1)
        # Camada do Topo (índice 0)
        if np.random.rand() < p_extreme_semispace_rho:
            if np.random.rand() < 0.5:
                rho_h_clean[0] = np.random.uniform(extreme_semispace_rho_range_low[0], extreme_semispace_rho_range_low[1])
            else:
                rho_h_clean[0] = np.random.uniform(extreme_semispace_rho_range_high[0], extreme_semispace_rho_range_high[1])
            rho_h_clean[0] = np.clip(rho_h_clean[0], rho_h_min, rho_h_max)
        
        if np.random.rand() < p_semispace_anisotropy:
            lambda_vals_clean[0] = np.random.uniform(semispace_anisotropy_range[0], semispace_anisotropy_range[1])
            lambda_vals_clean[0] = np.clip(lambda_vals_clean[0], lambda_min, lambda_max)

        # Camada da Base (índice ncam-1)
        if np.random.rand() < p_extreme_semispace_rho:
            if np.random.rand() < 0.5:
                rho_h_clean[ncam-1] = np.random.uniform(extreme_semispace_rho_range_low[0], extreme_semispace_rho_range_low[1])
            else:
                rho_h_clean[ncam-1] = np.random.uniform(extreme_semispace_rho_range_high[0], extreme_semispace_rho_range_high[1])
            rho_h_clean[ncam-1] = np.clip(rho_h_clean[ncam-1], rho_h_min, rho_h_max)
        
        if np.random.rand() < p_semispace_anisotropy:
            lambda_vals_clean[ncam-1] = np.random.uniform(semispace_anisotropy_range[0], semispace_anisotropy_range[1])
            lambda_vals_clean[ncam-1] = np.clip(lambda_vals_clean[ncam-1], lambda_min, lambda_max)

        rho_h_clean = conditional_rho_h_sampling_core_2(rho_h_clean, contrast_threshold = contraste, p_high = p_contraste,
                                                      infratio = inf_ratio, supratio = sup_ratio,
                                                      rhoh_min = rho_h_min, rhoh_max = rho_h_max)
        
        rho_h_noisy = add_gaussian_noise_2(rho_h_clean, std_dev = noise_std_rho)
        rho_h_noisy = np.clip(rho_h_noisy, rho_h_min, rho_h_max)

        lambda_vals_noisy = add_gaussian_noise_2(lambda_vals_clean, std_dev = noise_std_lambda)
        lambda_vals_noisy = np.clip(lambda_vals_noisy, lambda_min, lambda_max)

        rho_v_clean = (lambda_vals_clean**2) * rho_h_clean
        rho_v_noisy = (lambda_vals_noisy**2) * rho_h_noisy

        for j in range(ncam):
            if rho_v_clean[j] < rho_h_min * 2: rho_v_clean[j] = rho_h_min * 2
            if rho_v_clean[j] > rho_h_max * 2: rho_v_clean[j] = rho_h_max * 2
            if rho_v_noisy[j] < rho_h_min * 2: rho_v_noisy[j] = rho_h_min * 2
            if rho_v_noisy[j] > rho_h_max * 2: rho_v_noisy[j] = rho_h_max * 2

        thicknesses_noisy = np.array([])
        if len(thicknesses) > 0:
            thicknesses_noisy = thicknesses * (1 + np.random.normal(0, noise_std_thickness, size = thicknesses.shape))
            thicknesses_noisy = np.clip(thicknesses_noisy, min_thickness_internal, tji)

        models.append({
            'n_layers': ncam,
            'rho_h': rho_h_noisy.tolist(),
            'lambda': lambda_vals_noisy.tolist(),
            'rho_v': rho_v_noisy.tolist(),
            'thicknesses': thicknesses_noisy.tolist()
        })
    return models

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# Function que faz a leitura do arquivo que contém as informações de nº de thetas, nº de frequências e nº de medidas
# gerado pelo código fortran, e que facilita então o endereçamento das saídas do arquivo binário gerado por lá.
def ler_arquivo_com_int_float(arquivo):
    dados = []
    with open(arquivo,"r") as informa:
        for lin in informa:
            # Remove espaços em branco no início e no final da linha
            linha = lin.strip()
            # Divide a linha em colunas
            colunas = linha.split()
            # Converte cada coluna para int ou float, conforme possível
            linha_convertida = []
            for coluna in colunas:
                try:
                    # Tenta converter para int
                    valor = int(coluna)
                except ValueError:
                    try:
                        # Se não for int, tenta converter para float
                        valor = float(coluna)
                    except ValueError:
                        # Se não for possível converter, mantém como string
                        valor = coluna
                linha_convertida.append(valor)
            dados.append(linha_convertida)
    return dados

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
if __name__ == '__main__':
    start_time = time.perf_counter()
    mypath = os.path.dirname(os.path.realpath(__file__)) + '/'
    model = 'model.in'
    mymodel = mypath + model
    exec = 'tatu.x' #Gera o executável do fortran. Com o comando make apenas ele é gerado. Com make all, gera ele e roda este script também
    fortran_exec = mypath + exec
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    # Nem todas as variáveis abaixo são usadas, a não ser que você use na chamada de baseline, unfriendly e unfriendly_noisy
    tj = 120.                 #Tamanho da janela de investigação.
    h1 = 10.                  #Altura do primeiro ponto médio de leitura acima da primeira interface.
    hn = 10.                  #Profundidade do último ponto médio da última interface
    tji = tj - (h1 + hn)      #h1 m serão para acima da 1ª, e outros hn m para abaixo da última.
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    ntmodels = 3000#10#3000 #30000
    nmodels1 = int(ntmodels * 0.15)
    distrho = 'loguni'   #'unifor'  #
    models1 = baseline_empirical_2(nmodels = nmodels1)

    # nmodels1 = int(ntmodels * 0.19)
    # models1 = baseline_empirical_2( rho_h_min = 0.5,            #0.5, 
    #                                 rho_h_max = 1000,           #1000.,
    #                                 lambda_min = np.sqrt(2),    #1., 
    #                                 lambda_max = 2,             #np.sqrt(2),
    #                                 min_thickness = 1.,         #1.,
    #                                 nmodels = nmodels1)
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    nmodels2 = int(ntmodels * 0.1)
    models2 = baseline_ncamuniform_2(nmodels = nmodels2)
    # nmodels2 = int(ntmodels * 0.14)
    # models2 =baseline_ncamuniform_2(ncamin = 3,                 #3,
    #                                 ncamax = 81,                #81,
    #                                 min_thickness = 0.5,        #0.5,
    #                                 rho_h_min = 0.1,            #0.1,
    #                                 rho_h_max = 1400,           #1400., 
    #                                 lambda_min = np.sqrt(2),    #1., 
    #                                 lambda_max = 2,             #np.sqrt(2),
    #                                 nmodels = nmodels2)
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    nmodels3 = int(ntmodels * 0.1)
    models3 = baseline_thick_thicknesses_2(nmodels = nmodels3)
    # nmodels3 = int(ntmodels * 0.14)
    # models3 = baseline_thick_thicknesses_2(ncamin = 3,          #3,
    #                                 ncamax = 15,                #15,
    #                                 rho_h_min = 0.1,            #0.1,
    #                                 rho_h_max = 1400,           #1400.,
    #                                 rhoh_min_b = 0.1,           #0.1,
    #                                 rhoh_min_a = 100.,          #100., 
    #                                 rhoh_max_b = 50.,           #50.,
    #                                 rhoh_max_a = 1800.,         #1800,
    #                                 lambda_min = np.sqrt(2),    #1.,
    #                                 lambda_max = 2,             #np.sqrt(2),
    #                                 nmodels = nmodels3)
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    nmodels4 = int(ntmodels * 0.125)
    models4 = unfriendly_empirical_2(nmodels = nmodels4)
    # nmodels4 = int(ntmodels * 0.165)
    # models4 = unfriendly_empirical_2(rho_h_min = 0.1,           #0.1,
    #                                 rho_h_max = 1400.,          #1400,
    #                                 lambda_min = np.sqrt(2),    #1.0,
    #                                 lambda_max = 2,             #np.sqrt(2),
    #                                 nmodels = nmodels4,
    #                                 contraste = 5,              #5,
    #                                 inf_ratio = 1/150.,         #1/150,
    #                                 sup_ratio = 150.,           #150,
    #                                 rho_limite = 100,           #100,
    #                                 p_contraste = 0.5,          #0.5,
    #                                 p_camfina = 0.6,            #0.6,
    #                                 min_thickness = 0.2)
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    nmodels5 = int(ntmodels * 0.175)
    models5 = unfriendly_noisy_2(nmodels = nmodels5)
    # nmodels5 = int(ntmodels * 0.215)
    # models5 = unfriendly_noisy_2(rho_h_min = 0.1,               #0.1,
    #                                 rho_h_max = 1400,           #1400.,
    #                                 lambda_min = np.sqrt(2),    #1.,
    #                                 lambda_max = 2,             #np.sqrt(2),
    #                                 nmodels = nmodels5,
    #                                 contraste = 5,              #5,
    #                                 inf_ratio = 1/150,          #1/150,
    #                                 sup_ratio = 150,            #150,
    #                                 rho_limite = 100.,          #100,
    #                                 p_contraste = 0.5,          #0.5,
    #                                 p_camfina = 0.6,            #0.6,
    #                                 min_thickness = 0.1,        #0.1,
    #                                 noise_std_rho = 0.05,        #0.05,
    #                                 noise_std_lambda = 0.02,    #0.02,
    #                                 noise_std_thickness = 0.03) #0.03
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    nmodels6 = int(ntmodels * 0.1)
    models6 = generate_pathological_models_2(nmodels = nmodels6)
    # nmodels6 = int(ntmodels * 0.15)
    # models6 = generate_pathological_models_2(ncamin = 3,           #3,
    #                                 ncamax = 28,                   #28,
    #                                 rho_h_min = 0.05,              #0.05,
    #                                 rho_h_max = 1500.,             #1500.,
    #                                 lambda_min = np.sqrt(2),       #1.,
    #                                 lambda_max = np.sqrt(4.5),     #np.sqrt(1.5)
    #                                 nmodels = nmodels6,
    #                                 p_total_depth_fill = 0.7,      #0.7,   #Probabilidade de NÃO preencher a janela (resulta em última camada grossa)
    #                                 min_fill_ratio = 0.4,          #0.4,   #Mínimo da janela preenchida, ex: 40% de tji
    #                                 min_thickness_internal = 0.1,  #0.1,   #Permitir camadas internas muito finas
    #                                 contraste = 10,                #10.,
    #                                 inf_ratio = 1./150,            #1./150,
    #                                 sup_ratio = 150,               #150.,
    #                                 rho_limite = 1700,             #1700.,
    #                                 p_contraste = 0.7,             #0.7,
    #                                 p_camfina = 0.8,               #0.8,
    #                                 noise_std_rho = 0.07,          #0.07,
    #                                 noise_std_lambda = 0.03,       #0.03,
    #                                 noise_std_thickness = 0.04,    #0.04,
    #                                 p_extreme_semispace_rho = 0.5, #0.5,
    #                                 extreme_semispace_rho_range_low = (0.05, 0.5),     #(0.05, 0.5),
    #                                 extreme_semispace_rho_range_high = (1000, 3000),   #(1000, 3000),
    #                                 p_semispace_anisotropy = 0.4,                      #0.4,
    #                                 semispace_anisotropy_range = (1.5, 2.5)            #(1.5, 2.5)
    #                                 )
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    nmodels7 = int(ntmodels * 0.125)
    models7 = generate_pathological_models_2(nmodels = nmodels7, rho_h_min = 0.1, rho_h_max = 1500, lambda_min = 1,
                                            lambda_max = 1.001, min_thickness_internal = 10, p_camfina = 0.3)
    #models7 = generate_pathological_models_2(ncamin = 3,            #3,
    #                                ncamax = 28,                   #28,
    #                                rho_h_min = 0.1,               #0.05,
    #                                rho_h_max = 1500.,             #1500.,
    #                                lambda_min = 1.,               #1.,
    #                                lambda_max = 1.01,             #np.sqrt(1.5)
    #                                nmodels = nmodels7,
    #                                p_total_depth_fill = 0.7,      #0.7,   #Probabilidade de NÃO preencher a janela (resulta em última camada grossa)
    #                                min_fill_ratio = 0.4,          #0.4,   #Mínimo da janela preenchida, ex: 40% de tji
    #                                min_thickness_internal = 10.,  #0.1,   #Permitir camadas internas muito finas
    #                                contraste = 10,                #10.,
    #                                inf_ratio = 1./150,            #1./150,
    #                                sup_ratio = 150,               #150.,
    #                                rho_limite = 1700,             #1700.,
    #                                p_contraste = 0.7,             #0.7,
    #                                p_camfina = 0.3,               #0.8,
    #                                noise_std_rho = 0.07,          #0.07,
    #                                noise_std_lambda = 0.03,       #0.03,
    #                                noise_std_thickness = 0.04,    #0.04,
    #                                p_extreme_semispace_rho = 0.5, #0.5,
    #                                extreme_semispace_rho_range_low = (0.1, 1.),      #(0.05, 0.5),
    #                                extreme_semispace_rho_range_high = (200, 1500),   #(1000, 3000),
    #                                p_semispace_anisotropy = 0.4,                     #0.4,
    #                                semispace_anisotropy_range = (3., 5.)             #(1.5, 2.5)
    #                                )
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••• •••••••••»
    nmodels8 = int(ntmodels * 0.125) #int(ntmodels * 0.075)  #
    models8 = baseline_empirical_2(nmodels = nmodels8, rho_h_min = 0.1, rho_h_max = 1500, rhohdistr = 'loguni',
                                  lambda_min = 1, lambda_max = 1.0001, min_thickness = 0.5)
    #models8 = baseline_empirical_2( rho_h_min = 0.1,            #0.5, 
    #                                rho_h_max = 1500,           #1000.,
    #                                lambda_min = 1.,            #1., 
    #                                lambda_max = 1.01,          #np.sqrt(2),
    #                                min_thickness = 0.5,        #1.,
    #                                rhohdistr = 'loguni',
    #                                nmodels = nmodels8)
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    models = models1 + models2 + models3 + models4 + models5 + models6 + models7 + models8  #
    nmodels_actual = len(models)   # contagem real após truncamento por int()
    # Aviso quando int() causa discrepância (ntmodels não é múltiplo de 40)
    if nmodels_actual != ntmodels:
        print(f'[AVISO] ntmodels={ntmodels} → int(fração×ntmodels) → {nmodels_actual} modelos reais '
              f'(diferença={ntmodels - nmodels_actual}). '
              f'nmaxmodel={nmodels_actual} será usado no model.in para garantir que o .out seja gerado.')
    print('Modelos criados!')
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    # ── Parâmetros físicos da simulação (GeradorComMultiplosReceptores-equivalente) ─────────────────────────────
    # Frequências baixas (2/6 kHz) + espaçamentos longos (8.19/20.43 m) = investigação profunda (DOI ~4–15 m)
    # Dois ângulos (0° e 30°) ativam componentes off-diagonal do tensor H (Hxy, Hxz, Hzx, etc.)
    # Saída: arquivos separados por par T-R (_TR1.dat / _TR2.dat), formato 22-col (sem freq/theta no binário)
    freqs   = [20000.]         #[2000., 6000.]  # frequências (Hz) — ex: 2k=δ≈11m, 6k=δ≈6.5m (ρ=10Ω·m)
    angulos = [0.]        #[0., 30.]        # ângulos de inclinação (°) — 0°=vertical, 30°=off-diagonal≠0
    nf      = len(freqs)       # número de frequências
    na      = len(angulos)     # número de ângulos
    pmed    = 0.2              # passo entre medidas (m)
    dTR     = [1.0]    #[8.19, 20.43]    # espaçamentos T-R (m) — investigação profunda (tipo ARC/Periscope)

    # ── F5/F7/F6/Filtro — Flags de features opcionais (v9.0) ─────────────────────
    # F5: Frequências arbitrárias — quando 1, permite nf > 2 sem aviso no Fortran.
    #     Quando 0 (default), o Fortran emite aviso se nf > 2.
    # F7: Antenas inclinadas — quando 1, calcula H_tilted(β,φ) para cada configuração.
    #     H_tilted(β,φ) = cos(β)·Hzz + sin(β)·[cos(φ)·Hxz + sin(φ)·Hyz]
    #     Saída estendida: 22 + 2×n_tilted colunas por registro binário.
    # F6: Compensação midpoint — quando 1, calcula medições compensadas entre pares T-R.
    #     Requer nTR ≥ 2. Gera arquivos _COMP{i}.dat e _COMP{i}_ATT.dat adicionais.
    #     Cancelamento de efeitos ambientais (rugosidade, excentricidade, invasão).
    # Filtro: Seleção do filtro de Hankel.
    #     0=Werthmuller 201pt (default), 1=Kong 61pt (rápido), 2=Anderson 801pt (preciso).
    use_arbitrary_freq  = 0      # F5: 0=desabilitado (default), 1=habilitado
    use_tilted_antennas = 0      # F7: 0=desabilitado (default), 1=habilitado
    tilted_configs      = []     # F7: lista de (beta, phi) em graus — ex: [(45., 0.), (30., 90.)]
    use_compensation    = 0      # F6: 0=desabilitado (default), 1=habilitado
    comp_pairs          = []     # F6: lista de (near_itr, far_itr) — ex: [(1, 3)] para nTR≥3
    filter_type         = 0      # Filtro: 0=Werthmuller, 1=Kong, 2=Anderson

    # Nome base dos arquivos: encoda frequências e ângulos para rastreabilidade total
    _freq_tag = '-'.join(f'{int(fi/1000)}k'  for fi in freqs)   # ex: "2k-6k"
    _ang_tag  = '-'.join(f'{int(ai)}'        for ai in angulos)  # ex: "0-30"
    filename  = f"Inv_f{_freq_tag}_a{_ang_tag}_{ntmodels}"        # ex: "Inv_f2k-6k_a0-30_3000"

    # ── Display de parâmetros de entrada (espelho do model.in a ser gerado) ──────────────────────────────────────
    import math as _math
    print('═'*89)
    print('PARÂMETROS DE ENTRADA — fifthBuildTIVModels.py')
    print('═'*89)
    print(f'  Executável Fortran : {fortran_exec}')
    print(f'  Arquivo model.in   : {mymodel}')
    print(f'  Arquivo de saída   : {filename}_TR{{i}}_d{{dTR}}m.dat  (i=1..{len(dTR)})')
    print('─'*89)
    print(f'  nf={nf}  Frequências (Hz) : {freqs}')
    for fi in freqs:
        delta_10 = 503 * _math.sqrt(10 / fi)
        print(f'    f={int(fi):>6} Hz — skin depth (ρ=10Ω·m): δ≈{delta_10:.1f} m')
    print(f'  na={na}  Ângulos (°)      : {angulos}')
    for ai in angulos:
        if ai == 0.:
            print(f'    θ=0°  → poço vertical, off-diagonal = 0')
        else:
            print(f'    θ={int(ai)}°  → poço desviado, off-diagonal ≠ 0, DOI radial = dTR×|sin({int(ai)}°)|')
    print(f'  nTR={len(dTR)}  Espaçamentos T-R (m): {dTR}')
    _ang_max = max(angulos)
    for dtr_i in dTR:
        for fi in freqs:
            delta_i = 503 * _math.sqrt(10 / fi)
            doi_est = dtr_i * abs(_math.sin(_math.radians(_ang_max))) if _ang_max > 0 else 0.0
            print(f'    dTR={dtr_i}m, f={int(fi/1000)}kHz → r_k(θ={int(_ang_max)}°)={doi_est:.2f}m, δ≈{delta_i:.1f}m')
    print(f'  pmed={pmed} m   h1={h1} m   tj={tj} m   tji={tji:.1f} m')
    print(f'  ntmodels={ntmodels} → nmodels_actual={nmodels_actual} '
          f'({nmodels1}+{nmodels2}+{nmodels3}+{nmodels4}+{nmodels5}+{nmodels6}+{nmodels7}+{nmodels8})'
          + (f'  [AVISO: diferença={ntmodels-nmodels_actual}]' if nmodels_actual != ntmodels else ''))
    nmeds = [_math.ceil(tj / (pmed * _math.cos(_math.radians(a)))) for a in angulos]
    regs_per_model = sum(nmeds) * nf
    _nmed_str = ' + '.join(f'nmed({int(a)}°)={n}' for a, n in zip(angulos, nmeds))
    print(f'  Registros/modelo/TR: {_nmed_str} × nf={nf} = {regs_per_model}')
    print(f'  Total registros/TR : {regs_per_model * nmodels_actual:,}  (~{regs_per_model * nmodels_actual * 172 / 1e9:.2f} GB)')
    print('═'*89)
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    # ── Configuração OpenMP ───────────────────────────────────────────────────────
    _omp_threads = os.environ.get('OMP_NUM_THREADS', None)
    if _omp_threads:
        print(f'[OpenMP] OMP_NUM_THREADS = {_omp_threads} thread(s)  (definido pelo usuário)')
    else:
        import multiprocessing as _mp
        print(f'[OpenMP] OMP_NUM_THREADS não definido — OpenMP usará o padrão do sistema '
              f'({_mp.cpu_count()} núcleos lógicos detectados)')
    print('[OpenMP] A distribuição interna threads_k × threads_j será exibida pelo Fortran abaixo.')
    print('─' * 89)
    _block_start = time.perf_counter()
    for i, m in enumerate(models):
        ncam = m['n_layers']
        with open(mymodel, "w") as f:
            f.write(str(nf) + '                 ' + '!número de frequências' + '\n')
            for _idx, _fi in enumerate(freqs, 1):
                f.write(str(_fi) + '           ' + f'!frequência {_idx}' + '\n')
            f.write(str(na) + '                 ' + '!número de ângulos de inclinação' + '\n')
            for _idx, _ai in enumerate(angulos, 1):
                f.write(str(_ai) + '               ' + f'!ângulo {_idx}' + '\n')
            f.write(str(h1) + '              ' + '!altura do primeiro ponto-médio T-R, acima da primeira interface de camadas' + '\n')
            f.write(str(tj) + '             ' + '!tamanho da janela de investigação' + '\n')
            f.write(str(pmed) + '               ' + '!passo entre as medidas' + '\n')
            # Feature 1 (Multi-TR): escreve nTR + nTR valores de dTR
            # Para nTR=1, backward-compatible com formato anterior
            if isinstance(dTR, (list, tuple, np.ndarray)):
                nTR_val = len(dTR)
                f.write(str(nTR_val) + '                 ' + '!número de pares T-R' + '\n')
                for itr_idx, dtr_val in enumerate(dTR):
                    f.write(str(dtr_val) + '               ' + '!distância T-R ' + str(itr_idx+1) + '\n')
            else:
                f.write('1                 ' + '!número de pares T-R' + '\n')
                f.write(str(dTR) + '               ' + '!distância T-R 1' + '\n')
            f.write(filename + '              ' + '!nome dos arquivos de saída' + '\n')
            f.write(str(ncam) + '                ' + '!número de camadas' + '\n')
            for j in range(ncam):
                myrhoh = np.array(m['rho_h'])[j]
                myrhov = np.array(m['rho_v'])[j]
                if myrhov < myrhoh: myrhov = myrhoh
                if j == 0:
                    f.write(str(round(myrhoh,2)) + '    ' + str(round(myrhov,2)) + '     ' + '!resistividades horizontal e vertical' + '\n')
                else:
                    f.write(str(round(myrhoh,2)) + '    ' + str(round(myrhov,2)) + '\n')
            for j in range(ncam-3):
                if j == 0:
                    f.write(str(round(np.array(m['thicknesses'])[j],2)) + '              ' + '!espessuras das n-2 camadas' + '\n')
                else:
                    f.write(str(round(np.array(m['thicknesses'])[j],2)) + '\n')
            f.write(str(round(np.array(m['thicknesses'])[-1],2)) + '\n')
            f.write(str(i+1) + ' ' + str(nmodels_actual) + '         ' + '!modelo atual e o número máximo de modelos' + '\n')
            # ── F5/F7/F6/Filtro — Flags opcionais v9.0 (backward compatible) ──
            f.write(str(use_arbitrary_freq)  + '                 ' + '!F5: use_arbitrary_freq (0=desabilitado, 1=habilitado)' + '\n')
            f.write(str(use_tilted_antennas) + '                 ' + '!F7: use_tilted_antennas (0=desabilitado, 1=habilitado)\n')
            if use_tilted_antennas == 1 and len(tilted_configs) > 0:
                f.write(str(len(tilted_configs)) + '                 ' + '!F7: n_tilted (número de configurações tilted)\n')
                for it_idx, (beta_t, phi_t) in enumerate(tilted_configs):
                    f.write(f'{beta_t}  {phi_t}' + '            ' + f'!F7: beta({it_idx+1}) phi({it_idx+1}) em graus\n')
            # F6 — Compensação midpoint
            f.write(str(use_compensation) + '                 ' + '!F6: use_compensation (0=desabilitado, 1=habilitado)\n')
            if use_compensation == 1 and len(comp_pairs) > 0:
                f.write(str(len(comp_pairs)) + '                 ' + '!F6: n_comp_pairs\n')
                for cp_idx, (near_i, far_i) in enumerate(comp_pairs):
                    f.write(f'{near_i}  {far_i}' + '              ' + f'!F6: near({cp_idx+1}) far({cp_idx+1})\n')
            # Filtro Adaptativo
            f.write(str(filter_type) + '                 ' + '!Filtro: 0=Werthmuller, 1=Kong, 2=Anderson\n')
        #-------------------------------------------------------------------------------------------------------------
        # Executando o programa Fortran
        try:
            result = sub.run([fortran_exec] , check = True, text = True, capture_output = True)
            if i == 0 and result.stdout.strip():   # exibe info de threads só no 1º modelo
                print(result.stdout.strip())
        except sub.CalledProcessError as e:
            print("Erro ao executar o programa Fortran:")
            print(e.stderr)
        if (i+1)%100==0:
            _block_end = time.perf_counter()
            _block_s   = _block_end - _block_start
            _mean_s    = _block_s / 100
            _eta_s     = _mean_s * (nmodels_actual - (i+1))
            print(f'  Iterações {i-98:>4d}–{i+1:<4d} de {ntmodels} | '
                  f'bloco={_block_s:6.1f}s | '
                  f'média={_mean_s:.3f}s/modelo | '
                  f'ETA={_eta_s/60:.1f}min')
            _block_start = _block_end
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    end_time = time.perf_counter()
    print('─'*89)
    print(f"Tempo de execução (treinamento): {(end_time - start_time)/3600:.4f} horas  "
          f"({(end_time - start_time):.1f} s)")
    print("AVISO: ao acrescentar dados a um dataset existente, atualize nmaxmodel no .out.")
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    # ── Geração garantida do .out pelo Python ─────────────────────────────────────────────────────────────────────
    # O Fortran só escreve o .out na última iteração (modelm == nmaxmodel). Se o Fortran falhar no
    # último modelo, o .out nunca é criado. Para garantia total, o Python regenera o .out aqui,
    # com os mesmos campos e formato que o Fortran produziria:
    #   Linha 1: nt  nf  nmaxmodel
    #   Linha 2: theta(1..nt)       — ângulos em graus (real64)
    #   Linha 3: freq(1..nf)        — frequências em Hz (real64)
    #   Linha 4: nmeds(1..nt)       — nº de medidas por ângulo (inteiro)
    # Nota: nmeds(k) = ceil(tj / (pmed × cos(θ_k))) — mesmo cálculo do Fortran
    out_src = mypath + 'info' + filename + '.out'
    _nmeds_list = [_math.ceil(tj / (pmed * _math.cos(_math.radians(a)))) for a in angulos]
    with open(out_src, 'w') as _f_out:
        _f_out.write(f' {na} {nf} {nmodels_actual}\n')
        _f_out.write(' ' + ' '.join(f'{float(a):.1f}' for a in angulos) + '\n')
        _f_out.write(' ' + ' '.join(f'{float(f):.1f}' for f in freqs) + '\n')
        _f_out.write(' ' + ' '.join(str(n) for n in _nmeds_list) + '\n')
    print(f'[.out gerado pelo Python] {out_src}')
    print(f'  nt={na}  nf={nf}  nmaxmodel={nmodels_actual}')
    print(f'  theta={angulos}°  freq={freqs} Hz  nmeds={_nmeds_list}')
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    # ── Pós-processamento: renomear arquivos com dTR e gerar .out idênticos ao .dat ──────────────────────────────
    # O Python gera: info{filename}.out  (acima, garantido).
    # Após renomear : {filename}_TR{i}_d{dTR}m.dat  e  {filename}_TR{i}_d{dTR}m.out  (cópia por TR).
    import shutil
    final_dat_files = []  # lista de (dat_path, label, out_path)
    print('─'*89)
    print('Renomeação de arquivos de saída (freq + ângulo + dTR nos nomes):')
    _nTR = len(dTR)
    for itr, dtr in enumerate(dTR):
        dtr_tag  = str(dtr).replace('.', 'p')                        # "8p19"
        _tr_suf  = f'_TR{itr+1}' if _nTR > 1 else ''               # nTR=1: Fortran não adiciona sufixo _TR
        old_dat  = mypath + filename + f'{_tr_suf}.dat'
        new_name = filename + f'_TR{itr+1}_d{dtr_tag}m'             # ex: "Inv_f2k-6k_a0-30_3000_TR1_d8p19m"
        new_dat  = mypath + new_name + '.dat'
        new_out  = mypath + new_name + '.out'
        if os.path.exists(old_dat):
            os.rename(old_dat, new_dat)
            print(f'  .dat : {os.path.basename(old_dat)}\n'
                  f'       → {os.path.basename(new_dat)}')
        else:
            print(f'  [AVISO] .dat não encontrado: {old_dat}')
        if os.path.exists(out_src):
            shutil.copy(out_src, new_out)
            print(f'  .out : info{filename}.out\n'
                  f'       → {os.path.basename(new_out)}')
        else:
            print(f'  [AVISO] .out fonte não encontrado: {out_src}')
        final_dat_files.append((new_dat, f'TR{itr+1} (dTR={dtr} m)', new_out))
    print('─'*89)
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    # ── Validação numérica dos arquivos .dat de treinamento ──────────────────────────────────────────────────────
    # Formato binário (Fortran_Gerador): 1 int32 + 21 float64 = 22 colunas / 172 bytes por registro.
    # Layout: col0=med | col1=z_obs | col2=rho_h | col3=rho_v | col4..21 = Re/Im dos 9 componentes do tensor H.
    # Ordem no arquivo: k(ângulo) → j(freq) → i(medida).
    # Para ntheta=2, nf=2: blocos θ=angulos[0]/freqs[0], θ=angulos[0]/freqs[1], θ=angulos[1]/freqs[0], θ=angulos[1]/freqs[1].
    dtyp = np.dtype([('col0', np.int32)] + [('col{}'.format(i), np.float64) for i in range(1, 22)])
    print(f'Validação dos arquivos .dat de treinamento ({len(final_dat_files)} par(es) T-R):')
    for dat_path, label, out_path in final_dat_files:
        try:
            mydat  = np.fromfile(dat_path, dtype=dtyp)
            myarr  = np.array(mydat.tolist())
            nrow   = myarr.shape[0]
            has_nan = np.isnan(myarr).any()
            has_inf = np.isinf(myarr).any()
            status  = 'OK' if not has_nan and not has_inf else 'ATENÇÃO'
            print(f'  [{status}] {label}')
            print(f'         .dat  : {dat_path}')
            print(f'         .out  : {out_path}')
            print(f'         Regs  : {nrow:,} | NaN={has_nan} | Inf={has_inf}')
            print(f'         H33   : Re={myarr[-1,-2]:.4e}  Im={myarr[-1,-1]:.4e}')
        except FileNotFoundError:
            print(f'  [ERRO] {label} — arquivo não encontrado: {dat_path}')
    print('─'*89)
    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
    # ── Tempo total de execução ────────────────────────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - start_time
    print('═'*89)
    print(f'fifthBuildTIVModels.py concluído. {nmodels_actual} modelos gerados em {total_elapsed:.2f} s.')
    print(f'Para validação com modelos canônicos (Oklahoma, Devine, Hou et al.), execute:')
    print(f'  python buildValidamodels.py')
    print('═'*89)

    #«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
