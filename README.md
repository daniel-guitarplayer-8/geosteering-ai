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

- **44 arquiteturas** (39 standard + 5 geosteering nativas causais)
- **26 funções de perda** (13 genéricas + 4 geofísicas + 9 avançadas)
- **3 cenários PINN** (oracle, surrogate, maxwell) + TIV constraint
- **7 Feature Views** (identity, H1_logH2, logH1_logH2, 3× fase/razão, second_order)
- **5 famílias Geosinais** (USD, UAD, UHR, UHA, U3DF)
- **Dual-mode:** offline (acausal) e realtime (causal)
- **On-the-fly:** noise → FV → GS → scale (fisicamente correto)
- **Integração científica:** Consensus MCP Server (Semantic Scholar + ArXiv)

Documentação completa: [docs/ARCHITECTURE_v2.md](docs/ARCHITECTURE_v2.md)

## Estrutura do Pacote

```
geosteering_ai/
├── config.py              ← PipelineConfig dataclass (150+ campos)
├── data/                  ← Loading, splitting, FV, GS, scaling, pipeline
├── noise/                 ← On-the-fly noise (15 tipos), curriculum 3-phase
├── models/                ← 44 arquiteturas + ModelRegistry
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

Daniel Leal

## Licença

MIT
