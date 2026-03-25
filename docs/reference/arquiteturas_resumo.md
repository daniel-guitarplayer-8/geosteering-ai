# Catálogo de 26 Arquiteturas — Pipeline v5.0.15

## Referência rápida para geração de células C27–C37

---

## 1. Tabela de Arquiteturas

| # | Nome | Tipo | Dual-Mode | Causal | Tier | Uso Recomendado |
|:-:|:-----|:----:|:---------:|:------:|:----:|:----------------|
| 1 | **ResNet_18** ★ | CNN | ✅ | Adaptável | 1 (Alta) | **DEFAULT — baseline** |
| 2 | ResNet_34 | CNN | ✅ | Adaptável | 1 | Modelos mais profundos |
| 3 | CNN_1D | CNN | ✅ | Adaptável | 1 | Simples e rápido |
| 4 | TCN | CNN | ✅ | **Nativo** | 1 | Dilated causal conv |
| 5 | TCN_Advanced | CNN | ✅ | Adaptável | 2 | TCN + gating |
| 6 | LSTM | RNN | ✅ | **Nativo** | 3 | Sequências curtas |
| 7 | BiLSTM | RNN | ✅ | **Incompatível** | 2 | Apenas offline |
| 8 | CNN_LSTM | Híbrido | ✅ | Adaptável | 2 | Features locais + contexto |
| 9 | CNN_BiLSTM_ED | Híbrido | ✅ | **Incompatível** | 1 | Encoder-Decoder bidirecional |
| 10 | UNet_Inversion | U-Net | ✅ | **Incompatível** | 2 | Downsample + upsample |
| 11 | Attention_UNet | U-Net | ✅ | **Incompatível** | 1 | U-Net + atenção |
| 12 | UNet_ResNet18 | U-Net | ✅ | **Incompatível** | 1 | U-Net com backbone ResNet |
| 13 | UNet_ResNet34 | U-Net | ✅ | **Incompatível** | 1 | U-Net com backbone ResNet |
| 14 | Transformer | Transformer | ✅ | Adaptável | 2 | Atenção global |
| 15 | TFT | Transformer | ✅ | Adaptável | 2 | Temporal Fusion |
| 16 | N-BEATS | Decomposição | ✅ | Adaptável | 2 | Basis expansion |
| 17 | N-HiTS | Decomposição | ✅ | Adaptável | 2 | Hierarchical N-BEATS |
| 18 | FNO | Operador | ✅ | Adaptável* | 3 | Fourier Neural Operator |
| 19 | DeepONet | Operador | ✅ | **Incompatível** | 3 | Deep Operator Network |
| 20 | Geophysical_Attention | Atenção | ✅ | Adaptável | 3 | Atenção geofísica |
| 21 | DNN | Dense | ✅ | Adaptável | 3 | Baseline simples |
| 22 | **WaveNet** | CNN | ✅ | **Nativo** | Geo | Inversão causal alta qualidade |
| 23 | **Causal_Transformer** | Transformer | ✅ | **Nativo** | Geo | GPT-style causal attention |
| 24 | **Informer/PatchTST** | Transformer | ✅ | Via máscara | Geo | Sparse attention O(NlogN) |
| 25 | **Mamba/S4** | SSM | ✅ | **Nativo** | Geo | **Menor latência O(1)** |
| 26 | **Encoder_Forecaster** | Seq2Seq | ✅ | **Nativo** | Geo | **Look-ahead explícito** |

*FNO: FFT global viola causalidade estritamente, mas adaptável com restrição espectral*

## 2. Compatibilidade Causal

| Categoria | Arquiteturas | Contagem |
|:----------|:-------------|:--------:|
| **Nativas causais** | WaveNet, Causal_Transformer, TCN, Mamba_S4, LSTM, Encoder_Forecaster | 6 |
| **Adaptáveis** | ResNet_18/34, CNN_1D, CNN_LSTM, TCN_Advanced, TFT, Informer, N-HiTS, N-BEATS, DNN, Geophysical_Attention, Transformer | 12* |
| **Incompatíveis** (offline only) | BiLSTM, CNN_BiLSTM_ED, UNet_*, Attention_UNet, DeepONet | 6 |

## 3. Regra de Preservação Temporal

TODA arquitetura DEVE preservar N_MEDIDAS no output:
```python
Input:  (batch, None, N_FEATURES)      # None para multi-ângulo
Output: (batch, N_MEDIDAS, OUTPUT_CHANNELS)  # SEMPRE preserva N

# Conv1D: strides=1, padding="same" (ou "causal" em realtime)
# RNN: return_sequences=True
# Dense: TimeDistributed(Dense(OUTPUT_CHANNELS, activation='linear'))
# U-Net: downsample + upsample retorna ao tamanho original
```

## 4. Model Factory (C37)

```python
model = build_model(
    model_type=MODEL_TYPE,        # "ResNet_18" (default)
    input_shape=(None, N_FEATURES),  # None para multi-ângulo
    output_channels=OUTPUT_CHANNELS,  # 2 (padrão) ou 4/6
    use_causal=USE_CAUSAL_MODE,
    **ARCH_PARAMS[MODEL_TYPE]
)
```

---

*Catálogo de Arquiteturas — Pipeline v5.0.15*
