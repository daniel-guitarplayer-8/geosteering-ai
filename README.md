# Geosteering AI

Pipeline de Inversão Geofísica 1D com Deep Learning para Geosteering.

Reproduz, com fidelidade física, a inversão eletromagnética em tempo real
através de arquiteturas de Deep Learning. Suporta componentes EM + geosinais
e/ou Feature Views como features para inversão de resistividade em cenários
de inferência offline (acausal) e realtime causal para geosteering em
ambientes ruidosos (on-the-fly).

## Instalação

```bash
# Desenvolvimento local
pip install -e ".[dev]"

# Google Colab (via GitHub)
!pip install git+https://github.com/daniel-leal/geosteering-ai.git@v2.0.0

# Google Colab (via Drive)
!pip install -e /content/drive/MyDrive/Geosteering_AI
```

## Uso Rápido

```python
from geosteering_ai import PipelineConfig
from geosteering_ai.data import DataPipeline
from geosteering_ai.models import ModelRegistry
from geosteering_ai.training import TrainingLoop

# Configuração via preset
config = PipelineConfig.robusto()

# Pipeline de dados
pipeline = DataPipeline(config)
data = pipeline.prepare("/path/to/dataset")

# Modelo
model = ModelRegistry().build(config)

# Treinamento
trainer = TrainingLoop(config, model, pipeline, data)
history = trainer.run()
```

## Presets Disponíveis

| Preset | Descrição |
|:-------|:---------|
| `PipelineConfig.baseline()` | P1: sem noise, sem GS, debugging |
| `PipelineConfig.robusto()` | E-Robusto: noise 8%, curriculum, LR 1e-4 |
| `PipelineConfig.nstage(n=2)` | N-Stage: clean → noise progressivo |
| `PipelineConfig.geosinais_p4()` | P4: geosinais on-the-fly + noise |
| `PipelineConfig.dtb_p5()` | P5: detecção de fronteiras geológicas (DTB) |
| `PipelineConfig.realtime()` | Geosteering: modo causal, sliding window |

## Arquitetura

- **48 arquiteturas** em 9 famílias (CNN, TCN, RNN, Híbrido, U-Net, Transformer, Decomposição, Avançado, Geosteering)
- **26 funções de perda** (13 genéricas + 4 geofísicas + 2 geosteering + 7 avançadas)
- **8 cenários PINN** (oracle, surrogate, maxwell, smoothness, skin_depth, continuity, variational, self_adaptive)
- **34 tipos de ruído** (9 CORE + 6 LWD + 12 EXTENDED + 5 originais + 2 geosteering) + curriculum 3-phase
- **7 Feature Views** (identity, raw, H1_logH2, logH1_logH2, 3× fase/razão, second_order)
- **5 famílias Geosinais** (USD, UAD, UHR, UHA, U3DF)
- **246 campos** PipelineConfig (ponto único de verdade)
- **UQ:** MC Dropout + Ensemble + INN (perturbação de entrada)
- **Dual-mode:** offline (acausal) e realtime (causal)
- **On-the-fly:** noise → FV → GS → scale (fisicamente correto)
- **Integração científica:** Consensus MCP Server (Semantic Scholar + ArXiv)

Documentação completa: [docs/ARCHITECTURE_v2.md](docs/ARCHITECTURE_v2.md)

## Estrutura do Pacote

```
geosteering_ai/
├── config.py              ← PipelineConfig dataclass (246 campos)
├── data/                  ← Loading, splitting, FV(7), GS(5), scaling(8), pipeline, DTB, surrogate
├── noise/                 ← On-the-fly noise (34 tipos), curriculum 3-phase
├── models/                ← 48 arquiteturas (9 famílias) + ModelRegistry
├── losses/                ← 26 losses + LossFactory + PINNs
├── training/              ← TrainingLoop, callbacks, N-Stage, Optuna HPO
├── inference/             ← InferencePipeline, realtime, export, UQ
├── evaluation/            ← Métricas, comparação, relatórios
├── visualization/         ← Plots, Picasso, EDA, geosteering
└── utils/                 ← Logger, timer, validação, formatação
```

## Framework

TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).

## Autor

Daniel Leal Souza, José Jadson, Celso Rafael, Valdelírio da Silva 

## Licença

MIT
