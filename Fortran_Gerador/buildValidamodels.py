import numpy as np
import pandas as pd         #usado apenas para teste de uso
import subprocess as sub    #permite executar o programa fortran, criar e acessar arquivos sobre o código de lá
import time                 #usado para apresentar o tempo de processamento
import os
import sys
import matplotlib.pyplot as plt

#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
#-------------------------------------------------------------------------------------------------------------
# Function que cria modelo de entrada para o código fortran da perfilagem.
def modelo(imod):
# Function que cria modelo de entrada para o código fortran da perfilagem.
    pe = 0.3048
    if (imod + 1) == 1: #modelo de Oklahoma de 3 camadas
        ncam = 3
        esp = [8 * pe]
        resh = np.ones(ncam)
        resv = np.ones(ncam)
        resh[1] = 20.
        resv[1] = 40.
    if (imod + 1) == 2:   #modelo de Oklahoma de 5 camadas
        ncam = 5
        esp = [17. * pe, 8. * pe, 4. * pe]
        resh = [5.,  50.,  2., 15., 4.5]
        resv = [10., 100., 4., 30., 9.]
    elif (imod + 1) == 3:   #modelo isotrópico Devine de 8 camadas
        ncam = 8
        esp = [20 * pe, 15 * pe, 19 * pe, 16 * pe, 15 * pe, 45 * pe]
        resh = [2., 3., 1.8, 4.71, 8.89, 5., 3.2, 8.]
        resv = [2., 3., 1.8, 4.71, 8.89, 5., 3.2, 8.]
    elif (imod + 1) == 4:   #modelo isotrópico Oklahoma de 15 camadas
        ncam = 15
        esp = [17 * pe, 8 * pe, 4 * pe, 3 * pe, 7 * pe, 4 * pe, 6 * pe, 3 * pe, 5 * pe, 7 * pe, 18 * pe, 8 * pe, 7 * pe]
        resh = [5., 50., 2., 15., 4.5, 100., 3.5, 450., 30.12, 602.41, 20., 746.27, 200., 7.5, 500.]
        resv = [5., 50., 2., 15., 4.5, 100., 3.5, 450., 30.12, 602.41, 20., 746.27, 200., 7.5, 500.]
    elif (imod + 1) == 5: #modelo de Oklahoma de 28 camadas anisotrópicas (informado em Techical Report 32_2011, pagina 58)
        ncam = 28       #número de camadas
        pe = 0.3048     #pé em metros
        z = [46, 63, 71, 75, 78, 85, 89, 95, 98, 103, 110, 128, 136, 143, 153, 157, 162, 165, 169,
             173, 177, 182, 185, 187, 189, 191, 203] #profundidades das interfaces das camadas
        esp = np.diff(z) * pe   #espessuras das camadas internas
        resh = np.array([10., 100., 4., 30., 9., 200., 7., 909., 60., 1250., 40., 1416., 400., 15., 1000., 179., 1000., 15., 
                75., 9., 20., 100., 18., 200., 75., 149., 7., 11.])    #resistividades horizontais
        resv = 2 * resh                                    #resistividades verticais
    elif (imod + 1) == 6: #modelo de Hou, Mallan & Verdin (2006) "Finite-Difference simulation of borehole EM measurements in 3D anistropic media"
                          #"using coupled scalar-vector potentials"
        ncam = 7          #número de camadas
        esp = [1.52, 2.35, 2.1, 1.88, 0.92]       #espessuras das camadas internas
        resh = [1., 80., 1., 10., 1., 0.3, 1.]    #resistividades horizontais
        resv = [10., 80., 10., 10., 10., 0.3, 10.]  #resistividades verticais
    return ncam, esp, resh, resv
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
#-------------------------------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------------------------------
start_time = time.perf_counter()
mypath = os.path.dirname(os.path.realpath(__file__)) + '/'
model = 'model.in'
mymodel = mypath + model
#-------------------------------------------------------------------------------------------------------------
somas_esp = []
nmodels = 6     #Número de modelos que serão criados, e gerarão saídas através do código fortran
for m in range(nmodels):
    _, esp_temp, _, _ = modelo(m)
    somas_esp.append(np.sum(esp_temp))
maior_espessura = max(somas_esp)
tj = maior_espessura + 10 #20
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
exec = 'tatu.x' #Gera o executável do fortran. Com o comando make apenas ele é gerado. Com make all, gera ele e roda este script também
fortran_exec = mypath + exec
filename = 'validacao'
for m in range(nmodels):
    #-------------------------------------------------------------------------------------------------------------
    ncam, esp, resh, resv = modelo(m)
    soma_esp_modelo = np.sum(esp)
    espaco_restante = tj - soma_esp_modelo
    h1 = espaco_restante / 2.0
    hn = espaco_restante / 2.0
    tji = tj - (h1 + hn)      #h1 m serão para acima da 1ª, e outros hn m para abaixo da última.
    nf = 1                   #número de frequências de investigação
    freq = np.array([2.e4])
    na = 1                   #número de ângulos de inclinação
    ang = np.array([0])
    pmed = 0.2               #passo entre medidas
    nAr = 1                  #quantidade de arranjos TR
    dTRs = np.array([1.,2.,3.], dtype = float)
    print('Modelo {}'.format(m+1))
    #-------------------------------------------------------------------------------------------------------------
    f = open(mymodel,"w")
    f.write(str(nf) + '                 ' + '!número de frequências' + '\n')
    for j in range(nf):
        f.write(str(round(freq[j],4)) + '\n')
    f.write(str(na) + '                 ' + '!número de ângulos de inclinação' + '\n')
    for j in range(na):
        f.write(str(round(ang[j],2)) + '\n')
    f.write(str(h1) + '              ' + '!altura do primeiro ponto-médio T-R, acima da primeira interface de camadas' + '\n')
    f.write(str(tj) + '             ' + '!tamanho da janela de investigação' + '\n')
    f.write(str(pmed) + '               ' + '!passo entre as medidas' + '\n')
    f.write(str(nAr)+ '              ' + '!número de arranjos T-R, com um T' + '\n')
    for j in range(nAr):
        f.write(str(round(dTRs[j],2)) + '\n')
    f.write(filename + '              ' + '!nome dos arquivos de saída' + '\n')
    f.write(str(ncam) + '                ' + '!número de camadas' + '\n')
    for j in range(ncam):
        myrhoh = np.array(resh)[j]
        myrhov = np.array(resv)[j]
        if myrhov < myrhoh: myrhov = myrhoh
        if j == 0:
            f.write(str(round(myrhoh,2)) + '    ' + str(round(myrhov,2)) + '     ' + '!resistividades horizontal e vertical' + '\n')
        else:
            f.write(str(round(myrhoh,2)) + '    ' + str(round(myrhov,2)) + '\n')
    for j in range(ncam-3):
        if j == 0:
            f.write(str(round(esp[j],2)) + '              ' + '!espessuras das n-2 camadas' + '\n')
        else:
            f.write(str(round(esp[j],2)) + '\n')
    f.write(str(round(esp[-1],2)) + '\n')
    #Abaixo foi acrescentada uma linha nova para possibilitar a criação do arquivo informa.out com números de elementos da modelagem
    f.write(str(m+1) + ' ' + str(nmodels) + '         ' + '!modelo atual e o número máximo de modelos')
    f.close()
    #-------------------------------------------------------------------------------------------------------------
    # Executando o programa Fortran
    try:
        result = sub.run([fortran_exec] , check = True, text = True, capture_output = True)
        # print("Saída do Fortran:")
        # print(result.stdout)
    except sub.CalledProcessError as e:
        print("Erro ao executar o programa Fortran:")
        print(e.stderr)
    # if (m+1)%100==0:
    #     print('Ocorreram {0} iterações do total de {1}'.format(m+1,nmodels))
#-------------------------------------------------------------------------------------------------------------
end_time = time.perf_counter()
print('-----------------------------------------------------------------------------------------------------')
print(f"Tempo de execução: {(end_time - start_time)/3600:.6f} horas")
print("Por favor, se a execução deste programa não é a primeira vez e você está acrescentando")
print("mais dados ao seu dataset, você obrigatoriamente tem que mudar o número de modelos do")
print("do arquivo 'informa.dat' (última coluna da primeira linha)")
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# Define o tipo de dados (dtype) que corresponde ao formato do arquivo binário (existem 24 colunas)
# Neste caso, a primeira coluna é inteira (int32) e as demais são reais de precisão dupla (float64).
dtyp = np.dtype([('col0', np.int32)] + [('col{}'.format(i), np.float64) for i in range(1, 24)])
# Em cada linha o conteúdo do arquivo binário é:
# med, freq, theta, zobs, resh, resv, Re{Hxx}, Im{Hxx}, Re{Hxy}, Im{Hxy}, Re{Hxz}, Im{Hxz},
#                                     Re{Hyx}, Im{Hyx}, Re{Hyy}, Im{Hyy}, Re{Hyz}, Im{Hyz}
#                                     Re{Hzx}, Im{Hzx}, Re{Hzy}, Im{Hzy}, Re{Hzz}, Im{Hzz}
# O arquivo foi construído por 4 loops encaixados. O mais externo compreende à variação dos ângulos de inclinação.
# O segundo é sobre o número de frequências. O terceiro é sobre as medidas, enquanto o quarto é sobre o número de arranjos.
#-------------------------------------------------------------------------------------------------------------
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# Leitura do arquivo binário pelo numpy:
myout1 = mypath + filename + '.dat'
mydat = np.fromfile(myout1, dtype=dtyp)
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
#-------------------------------------------------------------------------------------------------------------
# Exemplo de conversão do arquivo binário para trabalhar com o numpy:
#-------------------------------------------------------------------------------------------------------------
# Converta para um array numpy estruturado como lista
myarr = np.array(mydat.tolist())
# Exiba o array
nrow = myarr.shape[0]
# ncol = myarr.shape[1]
print('-----------------------------------------------------------------------------------------------------')
print('Número de linhas do arquivo binário:')
print(nrow)
print('-----------------------------------------------------------------------------------------------------')
# Verifique se há algum NaN no array
has_nan = np.isnan(myarr).any()
# Exiba o resultado de haver ou não NaN
if has_nan:
   print("O array CONTÉM valores NaN.")
else:
   print("O array NÃO contém valores NaN.")
print(myarr[-1,-2],myarr[-1,-1])    #últimos valores de Hzz
print('-----------------------------------------------------------------------------------------------------')
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
# PLOTAGEM DE TODAS AS COMPONENTES DO TENSOR COM INTERFACES PARA TODOS OS MODELOS
#-------------------------------------------------------------------------------------------------------------

# Habilita o suporte a LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "axes.titlesize": 16,
})

# 1. Separando os dados por modelo:
z_diff = np.diff(myarr[:, 3])
split_indices = np.where(z_diff < 0)[0] + 1
modelos_dados = np.split(myarr, split_indices)

tensor_names = [
    ['H_{xx}', 'H_{xy}', 'H_{xz}'],
    ['H_{yx}', 'H_{yy}', 'H_{yz}'],
    ['H_{zx}', 'H_{zy}', 'H_{zz}']
]

# 2. LOOP SOBRE TODOS OS MODELOS
for idx_modelo in range(len(modelos_dados)):
    dados_plot = modelos_dados[idx_modelo]
    zobs = dados_plot[:, 3]

    # 3. EXTRAINDO AS INTERFACES DA FUNÇÃO modelo()
    ncam, esp, resh, resv = modelo(idx_modelo)

    interfaces = [0.0]
    interfaces.extend(np.cumsum(esp))

    # 4. Configurando a figura
    fig, axs = plt.subplots(3, 6, figsize=(22, 12), sharey=True)
    fig.suptitle(f'Componentes do Tensor Magnético - Modelo {idx_modelo + 1} (com {ncam} camadas)', fontsize=22, y=0.96)

    col_idx = 6  # Reinicia o índice da coluna para as saídas de cada novo modelo

    for i in range(3):
        for j in range(3):
            name = tensor_names[i][j]
            
            # --- Plot da Parte Real ---
            ax_re = axs[i, j*2]
            ax_re.plot(dados_plot[:, col_idx], zobs, color='darkblue', linewidth=2)
            ax_re.set_title(rf'$Re\{{{name}\}}$')
            ax_re.grid(True, linestyle=':', alpha=0.5) 
            
            for interf in interfaces:
                ax_re.axhline(y=interf, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
                
            if i == 2: ax_re.set_xlabel('Amplitude')
            col_idx += 1
            
            # --- Plot da Parte Imaginária ---
            ax_im = axs[i, j*2 + 1]
            ax_im.plot(dados_plot[:, col_idx], zobs, color='darkred', linewidth=2)
            ax_im.set_title(rf'$Im\{{{name}\}}$')
            ax_im.grid(True, linestyle=':', alpha=0.5) 
            
            for interf in interfaces:
                ax_im.axhline(y=interf, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
                
            if i == 2: ax_im.set_xlabel('Amplitude')
            col_idx += 1

    # Inverte o eixo Y para que a profundidade cresça para baixo
    axs[0, 0].invert_yaxis()

    for i in range(3):
        axs[i, 0].set_ylabel('Profundidade (m)')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    # Exibe a figura na tela
    plt.show()
#«•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••»
