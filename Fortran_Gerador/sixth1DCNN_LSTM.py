# Programa 6 para uso nos dados de perfilagem eletromagnética. Aqui fazemos tratamento nas componentes de campo antes do treinamento
#------------------------------------------------------------------------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Opcional: reduz mensagens verbosas do TF
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Opcional: garante ordem de detecção
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Opcional: para garantir que use a primeira GPU

# Colocando 7.5, você força a compilação para sua GPU.
#os.environ["TF_CUDA_COMPUTE_CAPABILITIES"] = "7.5"
#------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from keras import layers, models, losses , callbacks    #, metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler
from keras.regularizers import L1, L2, L1L2
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import joblib
# from keras import backend as K
import time                 #usado para apresentar o tempo de processamento
import matplotlib.pyplot as plt
#import sys


#TF_ENABLE_ONEDNN_OPTS=0
#------------------------------------------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------------------------------------------
def r2_metric(y_true, y_pred):
    # Implementação customizada do R²:
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

#------------------------------------------------------------------------------------------------------------------
def shuffleinblocks(array,block_size,myseed = 42):
    # Certificando-se de que o array pode ser dividido em blocos de tamanho fixo
    if len(array) % block_size != 0:
        raise ValueError("O número de linhas do array não é divisível pelo tamanho do bloco.")
    # Dividindo o array em blocos
    num_blocks = len(array) // block_size
    blocks = np.split(array, num_blocks)

    np.random.seed(myseed)
    # Embaralhando os blocos
    np.random.shuffle(blocks)   #a I vez
    # np.random.shuffle(blocks)   #uma II vez

    # Reunindo os blocos novamente
    shuffled_array = np.vstack(blocks)
    return shuffled_array

#------------------------------------------------------------------------------------------------------------------
def rmse_loss(y_true, y_pred):
    # Calcula o erro quadrático médio
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    # Retorna a raiz quadrada do MSE
    return tf.sqrt(mse)

#------------------------------------------------------------------------------------------------------------------
def huber_loss(y_true, y_pred, delta = 1.0):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear)

#------------------------------------------------------------------------------------------------------------------
def pseudo_huber_loss(y_true, y_pred, delta = 1.0):
    error = y_true - y_pred
    return delta**2 * (tf.sqrt(1 + (error / delta)**2) - 1)

#------------------------------------------------------------------------------------------------------------------
def log_cosh_loss(y_true, y_pred):
    error = y_true - y_pred
    return tf.reduce_mean(tf.math.log(tf.math.cosh(error)))

#------------------------------------------------------------------------------------------------------------------
def quantile_loss(y_true, y_pred, tau = 0.5):
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error))

#------------------------------------------------------------------------------------------------------------------
def tukeys_biweight_loss(y_true, y_pred, c = 4.685):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    
    # Aplica a função dentro do limite c
    mask = abs_error <= c
    loss_within_c = (c**2 / 6) * (1 - (1 - (error / c)**2)**3)
    
    # Para erros fora do limite, retorna o valor máximo da função
    loss_outside_c = (c**2 / 6) * tf.ones_like(error)
    
    # Combina os dois casos
    loss = tf.where(mask, loss_within_c, loss_outside_c)
    return tf.reduce_mean(loss)

#------------------------------------------------------------------------------------------------------------------
def cauchy_loss(y_true, y_pred, c = 1.0):
    error = y_true - y_pred
    loss = tf.math.log(1 + (error / c)**2)
    return tf.reduce_mean(loss)

#------------------------------------------------------------------------------------------------------------------
#def weighted_rmse(y_true, y_pred):
#    # weights = tf.where(y_true > threshold, high_weight, 1.0)  # Ex.: threshold = valor de resistividade alta
#    # weights = tf.where(y_true > 4.6, 0.1, 1.)  # Ex.: threshold = valor de resistividade alta
#    weights = tf.where(y_true > 2.3, 0.1, 1.)  # Ex.: threshold = valor de resistividade alta
#    mse = tf.reduce_mean(weights * tf.square(y_true - y_pred), axis=-1)
#    return tf.sqrt(mse)
def weighted_rmse(y_true, y_pred):
    # 1. Define um limiar (threshold) para separar "baixas" vs. "altas" resistividades
    threshold = np.log(500.)
    
    # 2. Atribui pesos maiores para valores acima do limiar
    # high_weight = 2.0  # Peso para altas resistividades ou weighst = 1.0 + (y_true / threshold)
    #weights = tf.where(y_true > threshold, high_weight, 1.0)  # Se y_true > threshold, usa high_weight; senão, 1.0
    weights = tf.where(y_true > threshold, 1.0 + (y_true / threshold), 1.0)
    #weights = tf.where(y_true > threshold, 2.0 - (threshold / y_true), 1.0)
    
    # 3. Calcula o MSE ponderado
    mse = tf.reduce_mean(weights * tf.square(y_true - y_pred), axis=-1)
    
    # 4. Retorna a raiz quadrada (RMSE)
    return tf.sqrt(mse)

#------------------------------------------------------------------------------------------------------------------
def weighted_mse(y_true, y_pred):
    weights = tf.where(y_true > 4.6, 0.1, 1.)  # 10x mais peso para ρ_h > 100 Ω.m
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))

#------------------------------------------------------------------------------------------------------------------
class HybridLoss(losses.Loss):
    def __init__(self, alpha=1.0, beta=0.5, name="hybrid_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        # 1. Calcular a perda do perfil (RMSE)
        # Usamos tf.reduce_mean para obter um único valor escalar
        loss_perfil = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

        # 2. Calcular o gradiente do perfil verdadeiro e predito
        # O resultado terá uma dimensão a menos na sequência
        grad_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        grad_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]

        # 3. Calcular a perda do gradiente (RMSE)
        loss_grad = tf.sqrt(tf.reduce_mean(tf.square(grad_true - grad_pred)))
        
        # 4. Combinar as duas perdas
        return self.alpha * loss_perfil + self.beta * loss_grad

#------------------------------------------------------------------------------------------------------------------
def semacoplamento(dataX, L = 1.):
    # L = distância entre o transmissor e o receptor
    ACp =-1 / (4. * np.pi * L**3)
    ACx = 1 / (2. * np.pi * L**3)
    # Criar um novo array vazio com as mesmas dimensões
    newdata = np.zeros_like(dataX)
    newdata[:,0] = dataX[:,0]
    newdata[:,1] = dataX[:,1] - ACp   #retirando-se o acoplamento mútuo de re{Hxx}
    newdata[:,2] = dataX[:,2]
    newdata[:,3] = dataX[:,3] - ACx   #retirando-se o acoplamento mútuo de re{Hzz}
    newdata[:,4] = dataX[:,4]
    return newdata

#------------------------------------------------------------------------------------------------------------------
def apenasnormaliza(dataX, train = True, sem_acoplamento = True):
    new_data = np.zeros_like(dataX)
    if sem_acoplamento:
        new_data = semacoplamento(dataX)
    else:
        new_data = dataX.copy()
    aux_data = np.concatenate((new_data[:,1].reshape(-1,1),new_data[:,2].reshape(-1,1),
                               new_data[:,3].reshape(-1,1),new_data[:,4].reshape(-1,1)), axis = 1)
    if train:
        print("'Re{Hxx}', 'Im{Hxx}', 'Re{Hzz}' e 'Im{Hzz}' normalizados sendo usadas")
        scaler = StandardScaler()   #atente-se que a tarefa de média nula e desvio-padrão 1 (StandardScaler) é somente sobre array 2D
        aux_data = scaler.fit_transform(aux_data)
        # Salva o scaler para reuso na validação
        joblib.dump(scaler, 'scaler.pkl')  # Ou pickle.dump(scaler, open('scaler.pkl', 'wb'))
    elif not train:
        # Carrega o scaler usado para a normalização do treino e teste
        scaler = joblib.load('scaler.pkl')  # Ou pickle.load(open('scaler.pkl', 'rb'))
        aux_data = scaler.transform(aux_data)

    x_final = np.concatenate((new_data[:,0].reshape(-1,1),aux_data),axis = 1)
    return x_final

#------------------------------------------------------------------------------------------------------------------
def Hxx_logHzz(dataX, normalizar = True, train = True, sem_acoplamento = True):
    
    # Criar um novo array vazio com as mesmas dimensões
    new_data = np.zeros_like(dataX)
    if sem_acoplamento:
        new_data = semacoplamento(dataX)
    else:
        new_data = dataX.copy()
        
    # Colunas 3 e 4: log(|H_zz|) e arg(H_zz)
    H_zz_real = new_data[:, 3]
    H_zz_imag = new_data[:, 4]
    H_zz_modulus = np.sqrt(H_zz_real**2 + H_zz_imag**2)  # |H_zz|
    H_zz_argument = np.arctan2(H_zz_imag, H_zz_real)     # arg(H_zz)
    new_data[:, 3] = np.log(H_zz_modulus)                # log(|H_zz|)
    new_data[:, 4] = H_zz_argument                       # arg(H_zz)
    
    aux_data = np.concatenate((new_data[:,1].reshape(-1,1),new_data[:,2].reshape(-1,1),
                               new_data[:,3].reshape(-1,1),new_data[:,4].reshape(-1,1)), axis = 1)
    if normalizar and train:
        print('Re[Hxx], Im[Hxx], ln|Hzz| e fase de Hzz sendo usadas')
        scaler = StandardScaler()   #atente-se que a tarefa de média nula e desvio-padrão 1 (StandardScaler) é somente sobre array 2D
        aux_data = scaler.fit_transform(aux_data)
        # Salva o scaler para reuso na validação
        joblib.dump(scaler, 'scaler.pkl')  # Ou pickle.dump(scaler, open('scaler.pkl', 'wb'))
    elif normalizar and not train:
        # Carrega o scaler usado para a normalização do treino e teste
        scaler = joblib.load('scaler.pkl')  # Ou pickle.load(open('scaler.pkl', 'rb'))
        aux_data = scaler.transform(aux_data)

    x_final = np.concatenate((new_data[:,0].reshape(-1,1),aux_data),axis = 1)

    return x_final

#------------------------------------------------------------------------------------------------------------------
def logHxx_logHzz(dataX, normalizar = True, train = True, sem_acoplamento = True):
    
    # Criar um novo array vazio com as mesmas dimensões
    new_data = np.zeros_like(dataX)
    if sem_acoplamento:
        new_data = semacoplamento(dataX)
    else:
        new_data = dataX.copy()
    
    # Colunas 1 e 2: log(|H_xx|) e arg(H_xx)
    H_xx_real = new_data[:, 1]
    H_xx_imag = new_data[:, 2]
    H_xx_modulus = np.sqrt(H_xx_real**2 + H_xx_imag**2)  # |H_xx|
    H_xx_argument = np.arctan2(H_xx_imag, H_xx_real)     # arg(H_xx)
    new_data[:, 1] = np.log(H_xx_modulus)                # log(|H_xx|)
    new_data[:, 2] = H_xx_argument                       # arg(H_xx)
    
    # Colunas 3 e 4: log(|H_zz|) e arg(H_zz)
    H_zz_real = new_data[:, 3]
    H_zz_imag = new_data[:, 4]
    H_zz_modulus = np.sqrt(H_zz_real**2 + H_zz_imag**2)  # |H_zz|
    H_zz_argument = np.arctan2(H_zz_imag, H_zz_real)     # arg(H_zz)
    new_data[:, 3] = np.log(H_zz_modulus)                # log(|H_zz|)
    new_data[:, 4] = H_zz_argument                       # arg(H_zz)
    
    aux_data = np.concatenate((new_data[:,1].reshape(-1,1),new_data[:,2].reshape(-1,1),
                               new_data[:,3].reshape(-1,1),new_data[:,4].reshape(-1,1)), axis = 1)
    if normalizar and train:
        print('ln|Hxx|, fase de Hxx, ln|Hzz| e fase de Hzz sendo usadas')
        scaler = StandardScaler()   #atente-se que a tarefa de média nula e desvio-padrão 1 (StandardScaler) é somente sobre array 2D
        aux_data = scaler.fit_transform(aux_data)
        # Salva o scaler para reuso na validação
        joblib.dump(scaler, 'scaler.pkl')  # Ou pickle.dump(scaler, open('scaler.pkl', 'wb'))
    elif normalizar and not train:
        # Carrega o scaler usado para a normalização do treino e teste
        scaler = joblib.load('scaler.pkl')  # Ou pickle.load(open('scaler.pkl', 'rb'))
        aux_data = scaler.transform(aux_data)

    x_final = np.concatenate((new_data[:,0].reshape(-1,1),aux_data),axis = 1)

    return x_final

#------------------------------------------------------------------------------------------------------------------
def ImHxxImHzz_razaoHxxHzz(dataX, normalizar = True, train = True, sem_acoplamento = True):
    if sem_acoplamento:
        newdata = semacoplamento(dataX)
    else:
        newdata = dataX.copy()
    H_xx_real = newdata[:, 1]
    H_xx_imag = newdata[:, 2]
    H_zz_real = newdata[:, 3]
    H_zz_imag = newdata[:, 4]
    new_data = np.zeros_like(newdata)
    new_data[:,0] = newdata[:, 0]
    new_data[:,1] = H_xx_imag
    new_data[:,2] = H_zz_imag
    new_data[:,3] = np.divide(H_xx_real, H_zz_real, where=(H_zz_real != 0))
    new_data[:,4] = np.divide(H_xx_imag, H_zz_imag, where=(H_zz_imag != 0))
    
    aux_data = np.concatenate((new_data[:,1].reshape(-1,1),new_data[:,2].reshape(-1,1),
                               new_data[:,3].reshape(-1,1),new_data[:,4].reshape(-1,1)), axis = 1)
    if normalizar and train:
        print("'Im{Hxx}', 'Im{Hzz}', 'Re{Hxx/Hzz}' e 'Im{Hxx/Hzz}' sendo usadas")
        scaler = StandardScaler()   #atente-se que a tarefa de média nula e desvio-padrão 1 (StandardScaler) é somente sobre array 2D
        aux_data = scaler.fit_transform(aux_data)
        # Salva o scaler para reuso na validação
        joblib.dump(scaler, 'scaler.pkl')  # Ou pickle.dump(scaler, open('scaler.pkl', 'wb'))
    elif normalizar and not train:
        # Carrega o scaler usado para a normalização do treino e teste
        scaler = joblib.load('scaler.pkl')  # Ou pickle.load(open('scaler.pkl', 'rb'))
        aux_data = scaler.transform(aux_data)

    x_final = np.concatenate((new_data[:,0].reshape(-1,1),aux_data),axis = 1)

    return x_final

#------------------------------------------------------------------------------------------------------------------
def ImHxxImHzz_lograzaoHxxHzz(dataX, normalizar = True, train = True, sem_acoplamento = True):
    # Criar um novo array vazio com as mesmas dimensões
    new_data = np.zeros_like(dataX)
    if sem_acoplamento:
        new_data = semacoplamento(dataX)
    else:
        new_data = dataX.copy()
    
    c_HxxHzz = []
    for i in new_data:
        HxxHzz = complex(i[1],i[2]) / complex(i[3],i[4])
        c_HxxHzz.append(HxxHzz)
    cHxxHzz = np.array(c_HxxHzz).reshape(-1,1)

    absHxxHzz = np.abs(cHxxHzz)
    fasHxxHzz = np.angle(cHxxHzz)
    # fasHxxHzz = (np.angle(cHxxHzz) + 2 * np.pi) % (2 * np.pi)
    # fasHxxHzz = np.unwrap(fasHxxHzz, period = 2 * np.pi, axis = -1)

    aux_data = np.concatenate((new_data[:,2].reshape(-1,1),new_data[:,4].reshape(-1,1),np.log(absHxxHzz),fasHxxHzz), axis = 1)

    if normalizar and train:
        print("'Im{Hxx}', 'Im{Hzz}', ln|Hxx/Hzz| e 'fase{Hxx/Hzz}' sendo usadas")
        scaler = StandardScaler()   #atente-se que a tarefa de média nula e desvio-padrão 1 (StandardScaler) é somente sobre array 2D
        aux_data = scaler.fit_transform(aux_data)
        # Salva o scaler para reuso na validação
        joblib.dump(scaler, 'scaler.pkl')  # Ou pickle.dump(scaler, open('scaler.pkl', 'wb'))
    elif normalizar and not train:
        # Carrega o scaler usado para a normalização do treino e teste
        scaler = joblib.load('scaler.pkl')  # Ou pickle.load(open('scaler.pkl', 'rb'))
        aux_data = scaler.transform(aux_data)

    x_final = np.concatenate((new_data[:,0].reshape(-1,1),aux_data),axis = 1)

    return x_final

#------------------------------------------------------------------------------------------------------------------
def ImHxxImHzz_FaseHxxFaseHzz(dataX, normalizar = True, train = True, sem_acoplamento = True):
    # Criar um novo array vazio com as mesmas dimensões
    new_data = np.zeros_like(dataX)
    if sem_acoplamento:
        new_data = semacoplamento(dataX)
    else:
        new_data = dataX.copy()
    
    # Colunas 1 e 2: log(|H_xx|) e arg(H_xx)
    H_xx_real = new_data[:, 1]
    H_xx_imag = new_data[:, 2]
    new_data[:, 1] = H_xx_imag
    new_data[:, 3] = np.arctan2(H_xx_imag, H_xx_real)     # arg(H_xx)
    
    # Colunas 3 e 4: log(|H_zz|) e arg(H_zz)
    H_zz_real = new_data[:, 3]
    H_zz_imag = new_data[:, 4]
    new_data[:, 2] = H_zz_imag
    new_data[:, 4] = np.arctan2(H_zz_imag, H_zz_real)     # arg(H_zz)
    
    aux_data = np.concatenate((new_data[:,1].reshape(-1,1),new_data[:,2].reshape(-1,1),
                               new_data[:,3].reshape(-1,1),new_data[:,4].reshape(-1,1)), axis = 1)
    if normalizar and train:
        print("'Im{Hxx}', 'Im{Hzz}', fase de Hxx e fase de Hzz sendo usadas")
        scaler = StandardScaler()   #atente-se que a tarefa de média nula e desvio-padrão 1 (StandardScaler) é somente sobre array 2D
        aux_data = scaler.fit_transform(aux_data)
        # Salva o scaler para reuso na validação
        joblib.dump(scaler, 'scaler.pkl')  # Ou pickle.dump(scaler, open('scaler.pkl', 'wb'))
    elif normalizar and not train:
        # Carrega o scaler usado para a normalização do treino e teste
        scaler = joblib.load('scaler.pkl')  # Ou pickle.load(open('scaler.pkl', 'rb'))
        aux_data = scaler.transform(aux_data)

    x_final = np.concatenate((new_data[:,0].reshape(-1,1),aux_data),axis = 1)

    return x_final

#------------------------------------------------------------------------------------------------------------------
def lr_schedule(epoch):
    lr_max = 1.e-3
    lr_min = 1.e-5
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / 150 * np.pi))

#------------------------------------------------------------------------------------------------------------------
class BoundedSmartReduceLR(Callback):
    def __init__(self, monitor = 'val_r2_metric', factor = 0.5, patience = 5,
                 min_lr = 1e-6, mode = 'max', min_metric = 0.97, max_metric = 0.99,
                 verbose = 1, log_lr = True):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.min_metric = min_metric
        self.max_metric = max_metric
        self.verbose = verbose
        self.log_lr = log_lr  # Controla se exibe o LR a cada época
        self.best_weights = None  # Armazenará TODOS os pesos do modelo (garantido)
        self.best_metric = -np.inf if mode == 'max' else np.inf
        self.wait = 0
        self.active = False
        self.best_epoch = 0  # Rastreia a época onde os melhores pesos foram salvos

    def on_epoch_begin(self, epoch, logs=None):
        if self.log_lr:
            current_lr = self._get_current_lr()
            print(f"Época {epoch + 1}: LR = {current_lr:.2e}")

    def _get_current_lr(self):
        """Método universal para obter o LR, compatível com AdamW e outros otimizadores."""
        if hasattr(self.model.optimizer, 'learning_rate'):
            return float(self.model.optimizer.learning_rate.numpy())
        else:
            return float(tf.keras.backend.get_value(self.model.optimizer.lr))

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor)
        if current_metric is None:
            if self.verbose > 0:
                print(f"Aviso: Métrica '{self.monitor}' não encontrada.")
            return

        # Desativa se max_metric for atingido
        if (self.mode == 'max' and current_metric >= self.max_metric) or \
           (self.mode == 'min' and current_metric <= self.max_metric):
            if self.verbose > 0 and self.active:
                print(f"\nCallback desativado: {self.monitor} = {current_metric:.4f} (>= {self.max_metric})")
            self.active = False
            return

        # Ativa se min_metric for atingido
        if not self.active:
            if (self.mode == 'max' and current_metric >= self.min_metric) or \
               (self.mode == 'min' and current_metric <= self.min_metric):
                self.active = True
                self.best_metric = current_metric
                self.best_weights = self.model.get_weights()  # Salva todos os pesos
                self.wait = 0
                if self.verbose > 0:
                    print(f"\nCallback ativado: {self.monitor} = {current_metric:.4f} (>= {self.min_metric})")

        # Atualiza melhores pesos se a métrica melhorou
        if self.active:
            if (self.mode == 'max' and current_metric > self.best_metric) or \
               (self.mode == 'min' and current_metric < self.best_metric):
                self.best_metric = current_metric
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch
                self.wait = 0

            # Reduz LR e restaura pesos se patience for esgotada
            if self.wait >= self.patience:
                self._reduce_lr_and_restore()
                self.wait = 0
            self.wait += 1

    def _reduce_lr_and_restore(self):
        # Reduz o LR (compatível com AdamW)
        current_lr = self._get_current_lr()
        new_lr = max(current_lr * self.factor, self.min_lr)

        if hasattr(self.model.optimizer, 'learning_rate'):
            self.model.optimizer.learning_rate.assign(new_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

        # Restaura os melhores pesos
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            if self.verbose > 0:
                print(f"\nPesos e viéses da época {self.best_epoch + 1} restaurados ({self.monitor} = {self.best_metric:.4f})")
                print(f"LR reduzido de {current_lr:.3e} para {new_lr:.3e} (fator: {self.factor}x). Pesos restaurados.")

#------------------------------------------------------------------------------------------------------------------
class LRNoise(Callback):
    def __init__(self, noise_factor=0.1, min_lr=1e-6, max_lr=1e-3, 
                 monitor='val_loss', mode='min', target_metric=0.2, patience=5, 
                 start_epoch=0, verbose=1):
        super().__init__()
        self.noise_factor = noise_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.monitor = monitor
        self.mode = mode  # 'min' para loss, 'max' para R²
        self.target_metric = target_metric  # Substitui min_metric
        self.patience = patience
        self.start_epoch = start_epoch
        self.verbose = verbose
        self.wait = 0
        self.active = False
        self.best_metric = np.inf if mode == 'min' else -np.inf
        self.best_weights = None
        self.monitor_op = np.less if mode == 'min' else np.greater  # Operador dinâmico

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose > 0 and hasattr(self, 'log_lr') and self.log_lr:
            current_lr = self._get_current_lr()
            print(f"Época {epoch + 1}: LR = {current_lr:.2e}")

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor)
        if current_metric is None:
            return

        # Ativação condicional universal
        if not self.active and self.monitor_op(current_metric, self.target_metric) and epoch >= self.start_epoch:
            self.active = True
            self.best_metric = current_metric
            self.best_weights = [layer.get_weights() for layer in self.model.layers]
            if self.verbose > 0:
                print(f"\nLRNoise ativado: {self.monitor} = {current_metric:.4f} ({'<=' if self.mode == 'min' else '>='} {self.target_metric})")

        # Lógica principal universal
        if self.active:
            if self.monitor_op(current_metric, self.best_metric):
                self.best_metric = current_metric
                self.best_weights = [layer.get_weights() for layer in self.model.layers]
                self.wait = 0
            else:
                self.wait += 1

            if self.wait >= self.patience:
                self._apply_noise()
                self.wait = 0

    def _apply_noise(self):
        current_lr = self._get_current_lr()
        noise = np.random.uniform(1 - self.noise_factor, 1 + self.noise_factor)
        new_lr = np.clip(current_lr * noise, self.min_lr, self.max_lr)
        
        if hasattr(self.model.optimizer, 'learning_rate'):
            self.model.optimizer.learning_rate.assign(new_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        
        if self.verbose > 0:
            print(f" - LR com ruído: {current_lr:.2e} → {new_lr:.2e}")

    def _get_current_lr(self):
        if hasattr(self.model.optimizer, 'learning_rate'):
            return float(self.model.optimizer.learning_rate.numpy())
        return float(tf.keras.backend.get_value(self.model.optimizer.lr))

#------------------------------------------------------------------------------------------------------------------
class EarlyStoppingWithEpoch(EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_epoch = 0  # Armazena a época com a melhor métrica
        self.best_weights = None  # Armazena os melhores pesos
        
        # Inicialização robusta
        self.monitor_op = np.greater if self.mode == 'max' else np.less
        self.best = -np.inf if self.mode == 'max' else np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose > 0:
                print(f"\nAviso: Métrica '{self.monitor}' não encontrada.")
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    if self.verbose > 0:
                        mode_desc = 'maior' if self.mode == 'max' else 'menor'
                        print(f"\nMelhores pesos restaurados (Época {self.best_epoch + 1}, {self.monitor} {mode_desc} = {self.best:.4f})")
                self.model.stop_training = True


#§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
start_time = time.perf_counter()
# Leitura do arquivo de saída, binário:
# Identifica o caminho em que está este script:
mypath = os.path.dirname(os.path.realpath(__file__)) + '/'  #a slash serve para contenação do nome do arquivo de saída
myout = mypath + 'Inv0Dip30mil.dat'
# myout = mypath + 'Inv0Dip100milB.dat'
#------------------------------------------------------------------------------------------------------------------
# Definindo o tipo de dados (dtype) para corresponder ao formato do arquivo binário
# Neste caso, a primeira coluna é inteira (int32) e as demais são reais de precisão dupla (float64)
# Em cada linha o conteúdo do arquivo binário é:
# Med, zobs, resh, resv, Re{Hxx}, Im{Hxx}, Re{Hxy}, Im{Hxy}, Re{Hxz}, Im{Hxz}, 
#                        Re{Hyx}, Im{Hyx}, Re{Hyy}, Im{Hyy}, Re{Hyz}, Im{Hyz},
#                        Re{Hzx}, Im{Hzx}, Re{Hzy}, Im{Hzy}, Re{Hzz}, Im{Hzz}
# O arquivo foi construído por 3 loops encaixados. O mais externo compreende à variação dos ângulos de inclinação.
# O segundo é sobre o número de frequências, enquanto o mais interno é sobre as medidas.
#------------------------------------------------------------------------------------------------------------------
dtyp = np.dtype([('col1', np.int32)] + [('col{}'.format(i), np.float64) for i in range(2, 23)])
mydat = np.fromfile(myout, dtype=dtyp)
# Convertendo o binário para um array numpy estruturado como lista
myarr = np.array(mydat.tolist())

nrow = myarr.shape[0]   #número de linhas do arquivo
ncol = myarr.shape[1]   #número de colunas do arquivo
print('-------------------------------------------------------')
print('Número de linhas e colunas do dataset de treino e teste')
print(nrow,ncol)
#§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
info = mypath + 'infoInv0Dip30mil.out'
# info = mypath + 'infoInv0Dip100milB.out'
# Ler o arquivo
dados = ler_arquivo_com_int_float(info) #aqui dados é uma lista de listas
nt, nf, nm = dados[0]
# nm = 2 * nm
theta = dados[1]    #array de ângulos de inclinação theta
freqs = dados[2]    #array de frequências

nmeds = dados[3]    #array de números de medidas para cada inclinação theta
print('-------------------------------------------------------')
# print('N° de inclinações, n° de frequências, n° de modelos, inclinação, frequência e n° de medidas')
# print(nt, nf, nm, theta[0], freqs[0], nmeds[0])
#------------------------------------------------------------------------------------------------------------------
nmtrain = int(0.8 * nm)
nmteste = int(0.2 * nm)
#------------------------------------------------------------------------------------------------------------------
# myarr = np.vstack((myarr1,myarr2))
#------------------------------------------------------------------------------------------------------------------
# Embaralhamento em blocos. Ou seja, a partir dos dados ocorre o embalhamento somente em blocos com tamanho das medidas.
# Isso é necessário se os dados foram criados por diversas distribuições lognormal para as resistividades.
myseed = 42
myarr = shuffleinblocks(myarr,nmeds[0],myseed)
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
x_full = myarr[:, [1, 4, 5, 20, 21]]  # z, Re{H_xx}, Im{H_xx}, Re{H_zz}, Im{H_zz}
y_full = np.log(myarr[:, [2, 3]])     # ln(resh) e  ln(resv)
#------------------------------------------------------------------------------------------------------------------
# x_full = apenasnormaliza(x_full)
x_full = Hxx_logHzz(x_full)
#x_full = logHxx_logHzz(x_full)
# x_full = ImHxxImHzz_razaoHxxHzz(x_full)
# x_full = ImHxxImHzz_lograzaoHxxHzz(x_full)
# x_full = ImHxxImHzz_FaseHxxFaseHzz(x_full)
print('-------------------------------------------------------')
#------------------------------------------------------------------------------------------------------------------
# Reshape para (n_modelos, n_medidas, n_features):
nmodels = int(nrow / nmeds[0])
x_3d = x_full.reshape(nmodels, nmeds[0], 5)  # Shape (12000, 601, 5) para usar z, Re{H_xx}, Im{H_xx}, Re{H_zz} e Im{H_zz} normalizados
y_3d = y_full.reshape(nmodels, nmeds[0], 2)  # Shape (12000, 601, 2) para os alvos resh e resv

x_train, x_test = x_3d[:nmtrain], x_3d[nmtrain:]
y_train, y_test = y_3d[:nmtrain], y_3d[nmtrain:]
#------------------------------------------------------------------------------------------------------------------
# Carregamento dos arquivos de validação:
myarq = mypath + 'validacao.dat'  #nome do arquivo, com seu caminho completo
my_dat = np.fromfile(myarq, dtype=dtyp)
# Convertendo o binário para um array numpy estruturado como lista
myval = np.array(my_dat.tolist())

infv = mypath + 'infovalidacao.out'
# Ler o arquivo
datval = ler_arquivo_com_int_float(infv) #aqui dados é uma lista de listas
nt, nf, nm = datval[0]
# nm = 2 * nm
t = datval[1]    #array de ângulos de inclinação theta
f = datval[2]    #array de frequências
nmed = datval[3]    #array de números de medidas para cada inclinação theta
# print(nt, nf, nm, t[0], f[0], nmed[0])

xvfull = myval[:, [1, 4, 5, 20, 21]]  # z, Re{H_xx}, Im{H_xx}, Re{H_zz}, Im{H_zz}
yvfull = np.log(myval[:, [2, 3]])     # ln(resh) e  ln(resv)

# xvfull = apenasnormaliza(xvfull, train = False)
xvfull = Hxx_logHzz(xvfull, train = False)
#xvfull = logHxx_logHzz(xvfull, train = False)
# xvfull = ImHxxImHzz_razaoHxxHzz(xvfull, train = False)
# xvfull = ImHxxImHzz_lograzaoHxxHzz(xvfull, train = False)
# xvfull = ImHxxImHzz_FaseHxxFaseHzz(xvfull, train = False)
xv3D = xvfull.reshape(nm, nmed[0], 5)  # Shape (12000, 601, 5) para usar z, Re{H_xx}, Im{H_xx}, Re{H_zz} e Im{H_zz} normalizados
yv3D = yvfull.reshape(nm, nmed[0], 2)  # Shape (12000, 601, 2) para os alvos resh e resv

x_val1 = xv3D[:1]
y_val1 = yv3D[:1]
x_val2 = xv3D[1:2]
y_val2 = yv3D[1:2]
x_val3 = xv3D[2:3]
y_val3 = yv3D[2:3]
x_val4 = xv3D[3:4]
y_val4 = yv3D[3:4]
x_val5 = xv3D[4:5]
y_val5 = yv3D[4:5]
# x_val6 = xv3D[5:6]
# y_val6 = yv3D[5:6]
#------------------------------------------------------------------------------------------------------------------
sequence_length = nmeds[0]
num_features = 5
model = models.Sequential([
    layers.Input(shape = (sequence_length, num_features)), # Camada de Entrada
    #-------------------------------------------------------------------------------------------------
    # --- ENCODER 1 ---
    layers.Conv1D(filters = 256, kernel_size = 6, activation = 'gelu', kernel_initializer = 'he_normal', padding = 'same'),
    layers.BatchNormalization(),
    layers.Dropout(0.1875),

    # --- RECORRENTE 1 ---
    layers.Bidirectional(layers.LSTM(256, return_sequences = True), merge_mode = 'ave'),
    #layers.Dropout(0.1), # Regularização da saída da LSTM 1

    # --------------------
    layers.Conv1D(filters = 256, kernel_size = 6, strides = 3, activation = 'gelu', kernel_initializer = 'he_normal', padding = 'same'), # 120 -> 24.
    layers.BatchNormalization(),
    layers.Dropout(0.1875),

    # --- RECORRENTE 2 ---
    layers.Bidirectional(layers.LSTM(256, return_sequences = True), merge_mode = 'ave'), # Mais unidades LSTM
    #layers.Dropout(0.1), # Regularização da saída da LSTM 2

    layers.Conv1DTranspose(256, kernel_size = 6, strides = 3, activation = 'gelu', kernel_initializer = 'he_normal', padding = 'same'),  # Volta para 600 pontos
    layers.BatchNormalization(),
    layers.Dropout(0.1875),

    layers.Bidirectional(layers.LSTM(256, return_sequences = True), merge_mode = 'ave'),
    #--------------------------------------------------------------------------------------------------------------------------
    # --- CAMADAS FINAIS ---
    layers.Dense(64,activation = 'gelu', kernel_initializer = 'he_normal', activity_regularizer = L2(5.e-6)),   #, activity_regularizer = L2(1.e-6)),
    layers.Dropout(0.09375),
    layers.Dense(2, activation = 'linear') # Saída para log(Rh) e log(Rv)
])

model.compile(
    # optimizer = tf.keras.optimizers.Adam(learning_rate = 5.e-4, decay = 1.e-4), #clipnorm = 1.5),
    # optimizer = 'adam',
    optimizer = tf.keras.optimizers.AdamW(learning_rate = 1.e-3, weight_decay = 1.e-4), #, clipnorm = 1.5),
    #optimizer = 'adamw',
    loss = rmse_loss,
    #loss = hybrid_loss_instance,
    metrics = [r2_metric, 'mae'])
    #metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse"), 'mae'])
# Treino
# --- Configuração dos Callbacks ---
callbacks_list = [
    # Estágio 1: Reduz LR para 0.5x quando R² >= 0.97 (só até R² < 0.99)
    #BoundedSmartReduceLR(
    #                    #monitor = 'val_r2_metric',
    #                    #factor = 0.75,
    #                    #min_metric = 0.99,
    #                    #max_metric = 0.995,  # Desativa-se ao atingir 0.99
    #    
    #                    monitor = 'val_loss',
    #                    factor = 0.5,
    #                    min_metric = 0.225,
    #                    max_metric = 0.05,  # Desativa-se ao atingir
    #                    patience = 5,
    #                    verbose = 1,
    #                    log_lr = False
    #                    ),
    
    # Estágio 2: Reduz LR para 0.25x quando R² >= 0.99
    #BoundedSmartReduceLR(
    #monitor = 'val_r2_metric',
    #factor = 0.5,
    #min_metric = 0.995,  # Só ativa aqui!
    #max_metric = 1.0,

    #    #monitor = 'val_loss',
    #    #factor = 0.25,
    #    #min_metric = 0.15,  # Só ativa aqui!
    #    #max_metric = 0.0,
    #patience = 6,       # Paciência menor para oscilações
    #verbose = 1,
    #log_lr = True
    #),
    
    # Ruído
    #LRNoise(
    #monitor = 'val_r2_metric',
    #mode = 'max',
    #target_metric = 0.997,  # Ativa quando R² >= 0.992
    #noise_factor = 0.05,
    #patience = 3
    #),
    #LRNoise(
            #monitor = 'val_loss',
            #mode = 'min',
            #target_metric = 0.098,  # Ativa quando loss <= 0.2
            #noise_factor = 0.05,
            #patience = 3
            #),

    #reduce_lr = 
    ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.5,     # Reduz para 1/2 da taxa atual
                              patience = 5,     # Espera 5 épocas sem melhora
                              mode = 'min',
                              min_lr = 5e-7,    # Exemplo de taxa de aprendizado mínima
                              verbose = 1),      # Para exibir mensagens quando a taxa é reduzida
    
    #early_stopping = 
    EarlyStopping(monitor = 'val_loss',
                            patience = 15,      # Exemplo: 20 épocas sem melhora na val_loss
                            mode = 'min',
                            restore_best_weights = True,
                            verbose = 1)

]
# Lista de callbacks a serem passados para model.fit
#callbacks_list = [reduce_lr, early_stopping]#, LearningRateScheduler(lr_schedule)]
#callbacks_list = [SmartReduceLR1(monitor = 'val_r2_metric', min_metric = 0.97, factor = 0.5, patience = 5, mode = 'max', verbose = 1),
#                  SmartReduceLR2(monitor = 'val_r2_metric', min_metric = 0.99, factor = 0.25, patience = 5, mode = 'max', verbose = 1)]
history = model.fit(x_train, y_train, epochs = 90, callbacks = callbacks_list, validation_data = (x_test, y_test), batch_size = 30)
#------------------------------------------------------------------------------------------------------------------
# Avaliação do R² com scikit-learn
print('------------------------------------------------------------------------------------------------')
y1pred_scaled = model.predict(x_val1)
print("R² sobre o modelo 1 de validação:", r2_score(y_val1.reshape(-1, 2), y1pred_scaled.reshape(-1, 2)))
y1pred = np.exp(y1pred_scaled)  #usado para a plotagem
print('------------------------------------------------------------------------------------------------')
y2pred_scaled = model.predict(x_val2)
print("R² sobre o modelo 2 de validação:", r2_score(y_val2.reshape(-1, 2), y2pred_scaled.reshape(-1, 2)))
y2pred = np.exp(y2pred_scaled)  #usado para a plotagem
print('------------------------------------------------------------------------------------------------')
y3pred_scaled = model.predict(x_val3)
print("R² sobre o modelo 3 de validação:", r2_score(y_val3.reshape(-1, 2), y3pred_scaled.reshape(-1, 2)))
y3pred = np.exp(y3pred_scaled)  #usado para a plotagem
print('------------------------------------------------------------------------------------------------')
y4pred_scaled = model.predict(x_val4)
print("R² sobre o modelo 4 de validação:", r2_score(y_val4.reshape(-1, 2), y4pred_scaled.reshape(-1, 2)))
y4pred = np.exp(y4pred_scaled)  #usado para a plotagem
print('------------------------------------------------------------------------------------------------')
y5pred_scaled = model.predict(x_val5)
print("R² sobre o modelo 5 de validação:", r2_score(y_val5.reshape(-1, 2), y5pred_scaled.reshape(-1, 2)))
y5pred = np.exp(y5pred_scaled)  #usado para a plotagem
print('------------------------------------------------------------------------------------------------')
end_time = time.perf_counter()
print('------------------------------------------------------------------------------------------------')
print(f"Tempo de execução: {(end_time - start_time)/3600:.6f} horas")

# Habilita o suporte a LaTeX
plt.rcParams.update({
    "text.usetex": True,  # Habilita o uso de LaTeX para renderizar texto
    "font.family": "serif",  # Usa uma fonte serifada (compatível com LaTeX)
    "font.serif": ["Computer Modern Roman"],  # Fonte padrão do LaTeX
    # Configurando o tamanho da fonte globalmente
    "font.size": 12,                          # Tamanho padrão para todos os textos
    "axes.titlesize": 20,                     # Tamanho do título
    "axes.labelsize": 16,                     # Tamanho dos rótulos dos eixos
    "xtick.labelsize": 14,                    # Tamanho dos ticks do eixo x
    "ytick.labelsize": 14,                    # Tamanho dos ticks do eixo y
    "legend.fontsize": 14                     # Tamanho da fonte da legenda
})

# Extrai os dados do histórico de treino
r2 = history.history['r2_metric']  # Nome da sua métrica customizada
val_r2 = history.history['val_r2_metric']
mae = history.history['mae']
val_mae = history.history['val_mae']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Criando a figura e os subplots do histórico:
fig0, his = plt.subplots(1, 3, figsize=(18, 6)) #, constrained_layout = True)  # 1 linha e 3 colunas de subplots
fig0.suptitle("Desempenho nas métricas")  # Título geral da figura

# Subplot 1: R² Score
his[0].plot(r2, label = r'Treino $R^2$')
his[0].plot(val_r2, label = r'Teste $R^2$')
his[0].set_title(r'$R^2$ Score por Época')
his[0].set_xlabel('Época')
his[0].set_ylabel(r'$R^2$')
his[0].legend()

# Subplot 2: MAE
his[1].plot(mae, label = 'Treino MAE')
his[1].plot(val_mae, label = 'Teste MAE')
his[1].set_title('MAE por Época')
his[1].set_xlabel('Época')
his[1].set_ylabel(r'MAE ($\Omega\cdot\ m$)')
his[1].legend()

# # Subplot 3: Loss
his[2].plot(loss, label = 'Treino Loss (MSE)')
his[2].plot(val_loss, label = 'Teste Loss (MSE)')
his[2].set_title('Loss por Época')
his[2].set_xlabel('Época')
his[2].set_ylabel('MSE')
his[2].legend()

#  Ajusta o layout para evitar sobreposição
fig0.tight_layout()

#------------------------------------------------------------------------------------------------------------------
z = x_full[0:nmeds[0],0]
# Criando as figuras dos modelos e os seus subplots
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6)) #, constrained_layout = True)  # 1 linha e 2 colunas de subplots
# Plota resh (resistividade horizontal)
ax1[0].plot(np.log10(np.exp(y_val1[0,:,0])),z, 'b-', label='Verdadeira')
ax1[0].plot(np.log10(y1pred[0,:,0]),z, 'r--', label='Predita')
ax1[0].invert_yaxis()  # Inverte o eixo y (profundidade aumenta para baixo)
ax1[0].set_xlabel(r'$log_{10}[\rho_h]$ ($\Omega\cdot m$)')
ax1[0].set_ylabel(r'z ($m$)')
ax1[0].set_title(r'Perfil de Resistividade Horizontal')
ax1[0].legend()
# Plota resv (resistividade vertical)
ax1[1].plot(np.log10(np.exp(y_val1[0,:,1])), z, 'g-', label='Verdadeira')
ax1[1].plot(np.log10(y1pred[0,:,1]), z, 'm--', label='Predita')
ax1[1].invert_yaxis()  # Inverte o eixo y (profundidade aumenta para baixo)
ax1[1].set_xlabel(r'$log_{10}[\rho_v]$ ($\Omega\cdot m$)')
ax1[1].set_ylabel(r'z ($m$)')
ax1[1].set_title(r'Perfil de Resistividade Vertical')
ax1[1].legend()
# Adicionando um título principal com valores numéricos em LaTeX
titulo_principal = (
    f'1DCNN + LSTM sobre o modelo anisotrópico Oklahoma de 3 camadas'
)
fig1.suptitle(titulo_principal, fontsize=16, y=0.98)
fig1.tight_layout()

#------------------------------------------------------------------------------------------------------------------
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6)) #, constrained_layout = True)  # 1 linha e 2 colunas de subplots
# Plota resh (resistividade horizontal)
ax2[0].plot(np.log10(np.exp(y_val2[0,:,0])),z, 'b-', label='Verdadeira')
ax2[0].plot(np.log10(y2pred[0,:,0]),z, 'r--', label='Predita')
ax2[0].invert_yaxis()  # Inverte o eixo y (profundidade aumenta para baixo)
ax2[0].set_xlabel(r'$log_{10}[\rho_h]$ ($\Omega\cdot m$)')
ax2[0].set_ylabel(r'z ($m$)')
ax2[0].set_title(r'Perfil de Resistividade Horizontal')
ax2[0].legend()
# Plota resv (resistividade vertical)
ax2[1].plot(np.log10(np.exp(y_val2[0,:,1])), z, 'g-', label='Verdadeira')
ax2[1].plot(np.log10(y2pred[0,:,1]), z, 'm--', label='Predita')
ax2[1].invert_yaxis()  # Inverte o eixo y (profundidade aumenta para baixo)
ax2[1].set_xlabel(r'$log_{10}[\rho_v]$ ($\Omega\cdot m$)')
ax2[1].set_ylabel(r'z ($m$)')
ax2[1].set_title(r'Perfil de Resistividade Vertical')
ax2[1].legend()
# Adicionando um título principal com valores numéricos em LaTeX
titulo_principal = (
    f'1DCNN + LSTM sobre o modelo de Oklahoma de 5 camadas'
)
fig2.suptitle(titulo_principal, fontsize=16, y=0.98)
fig2.tight_layout()
#------------------------------------------------------------------------------------------------------------------
fig3, ax3 = plt.subplots(1, 2, figsize=(12, 6)) #, constrained_layout = True)  # 1 linha e 2 colunas de subplots
# Plota resh (resistividade horizontal)
ax3[0].plot(np.log10(np.exp(y_val3[0,:,0])),z, 'b-', label='Verdadeira')
ax3[0].plot(np.log10(y3pred[0,:,0]),z, 'r--', label='Predita')
ax3[0].invert_yaxis()  # Inverte o eixo y (profundidade aumenta para baixo)
ax3[0].set_xlabel(r'$log_{10}[\rho_h]$ ($\Omega\cdot m$)')
ax3[0].set_ylabel(r'z ($m$)')
ax3[0].set_title(r'Perfil de Resistividade Horizontal')
ax3[0].legend()
# Plota resv (resistividade vertical)
ax3[1].plot(np.log10(np.exp(y_val3[0,:,1])), z, 'g-', label='Verdadeira')
ax3[1].plot(np.log10(y3pred[0,:,1]), z, 'm--', label='Predita')
ax3[1].invert_yaxis()  # Inverte o eixo y (profundidade aumenta para baixo)
ax3[1].set_xlabel(r'$log_{10}[\rho_v]$ ($\Omega\cdot m$)')
ax3[1].set_ylabel(r'z ($m$)')
ax3[1].set_title(r'Perfil de Resistividade Vertical')
ax3[1].legend()
# Adicionando um título principal com valores numéricos em LaTeX
titulo_principal = (
    f'1DCNN + LSTM sobre o modelo isotrópico Devine de 8 camadas'
)
fig3.suptitle(titulo_principal, fontsize=16, y=0.98)
fig3.tight_layout()
#------------------------------------------------------------------------------------------------------------------
fig4, ax4 = plt.subplots(1, 2, figsize=(12, 6)) #, constrained_layout = True)  # 1 linha e 2 colunas de subplots
# Plota resh (resistividade horizontal)
ax4[0].plot(np.log10(np.exp(y_val4[0,:,0])),z, 'b-', label='Verdadeira')
ax4[0].plot(np.log10(y4pred[0,:,0]),z, 'r--', label='Predita')
ax4[0].invert_yaxis()  # Inverte o eixo y (profundidade aumenta para baixo)
ax4[0].set_xlabel(r'$log_{10}[\rho_h]$ ($\Omega\cdot m$)')
ax4[0].set_ylabel(r'z ($m$)')
ax4[0].set_title(r'Perfil de Resistividade Horizontal')
ax4[0].legend()
# Plota resv (resistividade vertical)
ax4[1].plot(np.log10(np.exp(y_val4[0,:,1])), z, 'g-', label='Verdadeira')
ax4[1].plot(np.log10(y4pred[0,:,1]), z, 'm--', label='Predita')
ax4[1].invert_yaxis()  # Inverte o eixo y (profundidade aumenta para baixo)
ax4[1].set_xlabel(r'$log_{10}[\rho_v]$ ($\Omega\cdot m$)')
ax4[1].set_ylabel(r'z ($m$)')
ax4[1].set_title(r'Perfil de Resistividade Vertical')
ax4[1].legend()
# Adicionando um título principal com valores numéricos em LaTeX
titulo_principal = (
    f'1DCNN + LSTM sobre o modelo isotrópico Oklahoma de 15 camadas'
)
fig4.suptitle(titulo_principal, fontsize=16, y=0.98)
fig4.tight_layout()
#------------------------------------------------------------------------------------------------------------------
fig5, ax5 = plt.subplots(1, 2, figsize=(12, 6)) #, constrained_layout = True)  # 1 linha e 2 colunas de subplots
# Plota resh (resistividade horizontal)
ax5[0].plot(np.log10(np.exp(y_val5[0,:,0])),z, 'b-', label='Verdadeira')
ax5[0].plot(np.log10(y5pred[0,:,0]),z, 'r--', label='Predita')
ax5[0].invert_yaxis()  # Inverte o eixo y (profundidade aumenta para baixo)
ax5[0].set_xlabel(r'$log_{10}[\rho_h]$ ($\Omega\cdot m$)')
ax5[0].set_ylabel(r'z ($m$)')
ax5[0].set_title(r'Perfil de Resistividade Horizontal')
ax5[0].legend()
# Plota resv (resistividade vertical)
ax5[1].plot(np.log10(np.exp(y_val5[0,:,1])), z, 'g-', label='Verdadeira')
ax5[1].plot(np.log10(y5pred[0,:,1]), z, 'm--', label='Predita')
ax5[1].invert_yaxis()  # Inverte o eixo y (profundidade aumenta para baixo)
ax5[1].set_xlabel(r'$log_{10}[\rho_v]$ ($\Omega\cdot m$)')
ax5[1].set_ylabel(r'z ($m$)')
ax5[1].set_title(r'Perfil de Resistividade Vertical')
ax5[1].legend()
# Adicionando um título principal com valores numéricos em LaTeX
titulo_principal = (
    f'1DCNN + LSTM sobre o modelo anisotrópico Oklahoma de 28 camadas'
)
fig5.suptitle(titulo_principal, fontsize=16, y=0.98)
fig5.tight_layout()

plt.show()