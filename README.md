# Geosteering AI

Pipeline de Inversao Geofisica 1D com Deep Learning para Geosteering.

Reproduz, com fidelidade fisica, a inversao eletromagnetica em tempo real
atraves de arquiteturas de Deep Learning. Suporta componentes EM + geosinais
e/ou Feature Views como features para inversao de resistividade em cenarios
de inferencia offline (acausal) e realtime causal para geosteering em
ambientes ruidosos (on-the-fly).

## Instalacao

```bash
# Desenvolvimento local
pip install -e ".[dev]"

# Google Colab (via GitHub)
!pip install git+https://github.com/daniel-leal/geosteering-ai.git@v2.0.0

# Google Colab (via Drive)
!pip install -e /content/drive/MyDrive/Geosteering_AI
```

## Uso Rapido

```python
from geosteering_ai import PipelineConfig
from geosteering_ai.data import DataPipeline
from geosteering_ai.models import ModelRegistry
from geosteering_ai.training import TrainingLoop

# Configuracao via preset
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

## Presets Disponiveis

| Preset | Descricao |
|:-------|:---------|
| `PipelineConfig.baseline()` | P1: sem noise, sem GS, debugging |
| `PipelineConfig.robusto()` | E-Robusto: noise 8%, curriculum, LR 1e-4 |
| `PipelineConfig.nstage(n=2)` | N-Stage: clean → noise progressivo |
| `PipelineConfig.geosinais_p4()` | P4: geosinais on-the-fly + noise |
| `PipelineConfig.realtime()` | Geosteering: modo causal, sliding window |

## Arquitetura

- **44 arquiteturas** (39 standard + 5 geosteering nativas causais)
- **26 funcoes de perda** (13 genericas + 4 geofisicas + 9 avancadas)
- **Dual-mode:** offline (acausal) e realtime (causal)
- **On-the-fly:** noise → FV → GS → scale (fisicamente correto)

Documentacao completa: [docs/ARCHITECTURE_v2.md](docs/ARCHITECTURE_v2.md)

## Framework

TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).

## Autor

Daniel Leal

## Licenca

MIT
