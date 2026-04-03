# Catálogo de 26 Loss Functions — Pipeline v2.0

## Referencia rapida para geracao de C41 e configuracao de C10

---

## 1. Visao Geral

| Categoria | Qtd | Versao | Descricao |
|:----------|:---:|:------:|:----------|
| Genericas | 13 | v5.0.1+ | MSE, MAE, Huber, RMSE, etc. |
| Geofisicas | 4 | v5.0.3+ | log_scale_aware + variantes adaptativas (gangorra) |
| Geosteering (v5.0.7) | 2 | v5.0.7 | probabilistic_nll + look_ahead_weighted |
| Avançadas (v5.0.15+) | 7 | v5.0.15+ | DILATE, Sobolev, Spectral, Cross-Gradient, Encoder-Decoder, Multi-Task, morales_physics_hybrid |

**Total: 26 losses** — todas configuráveis via PipelineConfig, implementadas em `losses/catalog.py` (LossFactory).

---

## 2. Losses Genericas (13)

| # | LOSS_TYPE | Descricao | Keras Built-in | Quando Usar |
|:-:|:----------|:----------|:--------------:|:------------|
| 1 | `mse` | Mean Squared Error | Sim | Baseline; penaliza erros grandes |
| 2 | `rmse` | Root Mean Squared Error | Custom | Interpretavel na unidade original |
| 3 | `mae` | Mean Absolute Error | Sim | Robustez a outliers moderada |
| 4 | `mbe` | Mean Bias Error | Custom | Detectar bias sistematico |
| 5 | `rse` | Residual Sum of Squares | Custom | Compara com modelo nulo |
| 6 | `rae` | Relative Absolute Error | Custom | Erro relativo ao modelo nulo |
| 7 | `mape` | Mean Abs Percentage Error | Sim | Erro percentual; evitar se y~0 |
| 8 | `msle` | Mean Squared Log Error | Sim | Penaliza sub-estimacao em log |
| 9 | `rmsle` | Root Mean Squared Log Error | Custom | MSLE interpretavel |
| 10 | `nrmse` | Normalized RMSE | Custom | RMSE normalizado pelo range |
| 11 | `rrmse` | Relative RMSE | Custom | RMSE relativo a media |
| 12 | `huber` | Huber Loss (delta=1.0) | Sim | Hibrido MSE/MAE; robusto a outliers |
| 13 | `log_cosh` | Log-Cosh Loss | Sim | Similar a Huber, duplamente diferenciavel |

> **Nota:** Genericas operam em dominio log10 (TARGET_SCALING="log10"). Nao possuem termos
> geofisicos especificos (interface, oscilacao, subestimacao).

---

## 3. Losses Geofisicas (4)

| # | LOSS_TYPE | Base | Termos | FLAGS Principais |
|:-:|:----------|:-----|:-------|:-----------------|
| 14 | `log_scale_aware` | RMSE | InterfaceError + OscillationPenalty + UnderestimationPenalty | alpha=0.2, beta=0.1, gamma=0.05 |
| 15 | `adaptive_log_scale` | RMSE | Mesmos + gangorra beta | beta_min=0.1, beta_max=0.5, max_noise_expected |
| 16 | `robust_log_scale` | Huber | Interface + Oscillation + GlobalSmoothness + Underestimation | alpha=0.15, beta=0.1, gamma=0.15, delta_smooth=0.05 |
| 17 | `adaptive_robust_log_scale` | Huber | Mesmos + gangorra (noise_factor reduz alpha, beta, gamma) | Mesmos + max_noise_expected |

### Formula Geral (log_scale_aware):

```
L = w_mse * RMSE + alpha * InterfaceError + beta * OscillationPenalty + gamma * UnderestimationPenalty
w_mse = 1 - alpha - beta - gamma
```

### Mecanismo Gangorra (adaptive_*):

```
noise_ratio = clip(current_noise_level / max_noise_expected, 0, 1)
beta_eff = beta_min + (beta_max - beta_min) * noise_ratio

Ruido = 0   -> beta_eff = beta_min  (confia nos dados, fitting agressivo)
Ruido = max -> beta_eff = beta_max  (confia na fisica, suavidade prioritaria)
```

> **Nota:** `current_noise_level` e um `tf.Variable` escalar atualizado por
> `UpdateNoiseLevelCallback` a cada epoca (curriculum noise).

---

## 4. Losses Geosteering v5.0.7 (2)

| # | LOSS_TYPE | Formula | FLAGS (C10) | Pre-condicoes |
|:-:|:----------|:--------|:------------|:--------------|
| 18 | `probabilistic_nll` | L = log(sigma) + (y - mu)^2 / (2*sigma^2) | USE_PROBABILISTIC_LOSS | OUTPUT_CHANNELS >= 4 (mu_h, mu_v, sigma_h, sigma_v) |
| 19 | `look_ahead_weighted` | L = sum(w(d) * (y - y_hat)^2), w(d) = exp(-d / lambda) | USE_LOOK_AHEAD_LOSS, LOOK_AHEAD_LOSS_WEIGHT=0.3, LOOK_AHEAD_DECAY=10.0 | INFERENCE_MODE="realtime" |

- **probabilistic_nll:** NLL gaussiana para estimativa de incerteza. Saida (mu, sigma) por canal.
- **look_ahead_weighted:** Penaliza erros em pontos futuros com decaimento exponencial. Forca
  antecipacao de transicoes geologicas no geosteering.

---

## 5. Losses Geosteering Avancado v5.0.15 (6)

### #20 — DILATE (Le Guen & Thome, NeurIPS 2019)

```
L = alpha * softDTW(y_ds, y_hat_ds) / N_ds + (1-alpha) * MSE(dy/dz, dy_hat/dz)
```

| Aspecto | Detalhe |
|:--------|:--------|
| FLAGS | USE_DILATE_LOSS, DILATE_ALPHA=0.5, DILATE_GAMMA_SDTW=0.01, DILATE_DOWNSAMPLE_FACTOR=10 |
| Pre-condicoes | N_MEDIDAS=600, OUTPUT_CHANNELS >= 2 |
| Usar quando | Perfis com shifts espaciais; boundaries deslocadas entre y e y_hat |
| Evitar quando | N_MEDIDAS muito grande sem downsampling (O(N^2) no softDTW) |
| Impl. | soft-DTW em TF puro via DP (`_compute_softdtw`); processa canais rho_h e rho_v |

### #21 — Encoder-Decoder (Araya-Polo et al., 2018)

```
L = (1-lambda) * MSE(y, y_hat) + lambda * MSE(F(y_hat), F(y))
```

| Aspecto | Detalhe |
|:--------|:--------|
| FLAGS | USE_ENCODER_DECODER_LOSS, ENCODER_DECODER_WEIGHT=0.5 |
| Pre-condicoes | forward_model neural pre-treinado (surrogate do Fortran), pesos congelados |
| Usar quando | Surrogate disponivel; forca consistencia fisica (F(y_hat) ~ F(y)) |
| Evitar quando | Sem surrogate treinado (nao ha F(.) disponivel) |
| Impl. | F(.) e um Keras Model congelado que mapeia resistividade -> resposta EM |

### #22 — Multi-Task Learned (Kendall, Gal & Cipolla, CVPR 2018)

```
L = sum_k [ exp(-2*s_k)/2 * L_k + s_k ],  s_k = log(sigma_k) = tf.Variable
```

| Aspecto | Detalhe |
|:--------|:--------|
| FLAGS | USE_MULTITASK_LEARNED_LOSS, MULTITASK_INITIAL_LOG_SIGMA=0.0 |
| Pre-condicoes | log_sigmas em trainable_variables do optimizer (custom loop ou compile) |
| Usar quando | Multiplas sub-losses a balancear automaticamente (homoscedastic uncertainty) |
| Evitar quando | Loss unica (overhead desnecessario dos parametros s_k) |
| Impl. | `.trainable_variables` exposto na funcao retornada; sigma_k auto-ajustados |

### #23 — Sobolev H1 (Czarnecki et al., NeurIPS 2017)

```
L = MSE(y, y_hat) + lambda * MSE(dy/dz, dy_hat/dz)
```

| Aspecto | Detalhe |
|:--------|:--------|
| FLAGS | USE_SOBOLEV_LOSS, SOBOLEV_LAMBDA_GRAD=0.1 |
| Pre-condicoes | Nenhuma especifica |
| Usar quando | Preservar transicoes abruptas de resistividade (layer boundaries) |
| Evitar quando | lambda > 0.5 (degrada convergencia); perfis muito suaves |
| Impl. | Gradiente espacial via diferencas finitas: dy = y[:,1:,:] - y[:,:-1,:] |

### #24 — Cross-Gradient (Gallardo & Meju, GRL 2003)

```
L = MSE(y, y_hat) + lambda * MSE(tau_true, tau_pred)
tau(z) = (d_rho_h/dz) * (d_rho_v/dz)
```

| Aspecto | Detalhe |
|:--------|:--------|
| FLAGS | USE_CROSS_GRADIENT_LOSS, CROSS_GRADIENT_LAMBDA=0.1 |
| Pre-condicoes | OUTPUT_CHANNELS >= 2 (rho_h e rho_v) |
| Usar quando | Formacoes anisotropicas; forcar co-localizacao de transicoes rho_h <-> rho_v |
| Evitar quando | Formacoes isotropicas (rho_h ~ rho_v -> tau ~ 0, termo nao-informativo) |
| Impl. | tau > 0 = transicao concordante; tau ~ 0 = discordante ou estavel |

### #25 — Spectral (Jiang et al., ICCV 2021)

```
L = MSE(y, y_hat) + lambda * MSE(log1p(|FFT(y)|), log1p(|FFT(y_hat)|))
```

| Aspecto | Detalhe |
|:--------|:--------|
| FLAGS | USE_SPECTRAL_LOSS, SPECTRAL_LAMBDA=0.5 |
| Pre-condicoes | N_MEDIDAS idealmente potencia de 2 (FFT eficiente); N=600 OK |
| Usar quando | Oscilacoes espurias em altas resistividades; suavidade excessiva |
| Evitar quando | Perfis muito curtos (espectro pouco informativo) |
| Impl. | tf.signal.rfft ao longo do eixo z; amplitude em log-scale (log1p) |

---

## 6. Referencia: create_* Signatures

```python
# Geofisicas (novas_loss.py)
log_scale_aware_loss(y_true, y_pred, alpha=0.2, beta=0.1, gamma=0.05)        # funcao direta
robust_log_scale_loss(y_true, y_pred, alpha=0.15, beta=0.1, gamma=0.15,
                      delta_smooth=0.05)                                      # funcao direta
create_adaptive_logloss(alpha=0.2, beta_min=0.1, beta_max=0.4,
                        gamma=0.05, max_noise_expected=0.1)                   # -> callable
create_adaptive_robustloss(alpha=0.15, beta=0.1, gamma=0.15,
                           delta_smooth=0.05, max_noise_expected=0.1)         # -> callable

# v5.0.15 (novas_loss.py)
create_dilate_loss(alpha=0.5, gamma_sdtw=0.01, downsample_factor=10)          # -> callable
create_encoder_decoder_loss(forward_model, reconstruction_weight=0.5)         # -> callable
create_multitask_learned_loss(loss_fns, initial_log_sigma=0.0)                # -> callable (.trainable_variables)
create_sobolev_h1_loss(lambda_grad=0.1)                                       # -> callable
create_cross_gradient_loss(lambda_cross=0.1)                                  # -> callable
create_spectral_loss(lambda_spectral=0.5)                                     # -> callable
```

---

## 7. Fluxo de Decisao — Qual Loss Usar?

```
Treinamento padrao (offline, sem ruido)?
  -> log_scale_aware (default, 4 termos geofisicos)

Treinamento com ruido on-the-fly (curriculum noise)?
  -> adaptive_log_scale (gangorra: beta varia com noise_level)

Necessita robustez a outliers?
  -> robust_log_scale ou adaptive_robust_log_scale (base Huber)

Preservar transicoes abruptas (layer boundaries)?
  -> + sobolev_h1 (complementar, lambda=0.1)

Acoplamento rho_h <-> rho_v (anisotropia)?
  -> + cross_gradient (co-localizacao de transicoes)

Combater oscilacoes espurias em altas resistividades?
  -> + spectral (matching de amplitude FFT)

Shifts espaciais entre predicao e ground truth?
  -> + dilate (softDTW shape + derivative matching)

Multiplas losses simultaneas a balancear?
  -> multitask_learned (auto-balanceamento via sigma_k aprendidos)

Surrogate neural do simulador Fortran disponivel?
  -> + encoder_decoder (consistencia fisica F(y_hat) ~ F(y))

Geosteering com incerteza?
  -> probabilistic_nll (NLL gaussiana, saida mu+sigma)

Geosteering com look-ahead?
  -> look_ahead_weighted (penaliza erros futuros, decaimento exp.)

DTB como target adicional (P5 Picasso)?
  -> USE_DTB_LOSS=True + DTB_LOSS_WEIGHT=0.2
```

> **Combinacoes recomendadas:**
> - Offline padrao: `log_scale_aware`
> - Offline + ruido: `adaptive_log_scale` + `sobolev_h1`
> - Anisotropico: `adaptive_log_scale` + `cross_gradient` + `sobolev_h1`
> - Geosteering: `adaptive_log_scale` + `probabilistic_nll` + `look_ahead_weighted`
> - Geosteering avancado: `multitask_learned` wrapping sub-losses acima

---

## 8. Constantes Criticas (ERRATA v5.0.15)

| Constante | Valor CORRETO | Contexto |
|:----------|:------------:|:---------|
| DILATE_ALPHA | 0.5 | Equilibrio shape/temporal |
| DILATE_GAMMA_SDTW | 0.01 | Suavizacao soft-DTW |
| DILATE_DOWNSAMPLE_FACTOR | 10 | N=600 -> 60 (O(3600) ops) |
| SOBOLEV_LAMBDA_GRAD | 0.1 | Peso gradiente (10%) |
| CROSS_GRADIENT_LAMBDA | 0.1 | Acoplamento rho_h <-> rho_v |
| SPECTRAL_LAMBDA | 0.5 | Peso espectral (50%) |
| ENCODER_DECODER_WEIGHT | 0.5 | Reconstrucao (50%) |
| MULTITASK_INITIAL_LOG_SIGMA | 0.0 | sigma=1 inicial |
| DTB_LOSS_WEIGHT | 0.2 | Peso DTB na loss total |
| LOOK_AHEAD_LOSS_WEIGHT | 0.3 | Peso look-ahead |
| LOOK_AHEAD_DECAY | 10.0 | Decaimento exponencial |

---

*Catalogo de 25 Loss Functions — Pipeline v5.0.15*
