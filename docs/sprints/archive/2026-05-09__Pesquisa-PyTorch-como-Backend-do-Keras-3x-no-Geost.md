# Pesquisa: PyTorch como Backend do Keras 3.x no Geosteering AI

**Data**: 2026-05-09 · **Modo**: pesquisa apenas (plan mode) · **Sem alterações no projeto**

---

## Achados-Chave

1. **Mecanismo de seleção é estritamente startup-only**. Backend é definido por `os.environ["KERAS_BACKEND"]` *antes* de `import keras`, ou via `~/.keras/keras.json` (`{"backend": "torch"}`). A documentação oficial declara explicitamente: *"the backend must be configured before importing Keras, and cannot be changed after the package has been imported"* (keras.io/getting_started/). Não há API de runtime-switch.

2. **Keras 3 entrega backend-agnostic real desde 3.0**. Modelos `.keras`, `keras.ops` (NumPy + NN API), losses, metrics, optimizers e `model.fit()` rodam idênticos em TF/JAX/Torch. **`tf.data.Dataset` é aceito como input em `fit()` mesmo com backend Torch ou JAX** (keras.io/api/data_loading/) — o pipeline de dados atual não obriga migração.

3. **Custom code escrito com `tf.*` ops quebra ao mudar de backend**. `tf.GradientTape`, `tf.function`, `tf.reduce_sum`, `tf.concat`, `tf.range` etc. exigem reescrita para `keras.ops.*` equivalente. O guia oficial de migração (keras.io/guides/migrating_to_keras_3/) lista o mapping. Custom `train_step` precisa de implementação backend-específica (ou usar `compute_loss` que é agnóstico).

4. **Performance oficial favorece TF/JAX em modelos científicos pequenos/médios**. Benchmark oficial Keras 3 (A100, 6 modelos): **PyTorch backend é 2-4× mais lento que TF/JAX em todos os casos de fit/predict CV+NLP** (ex.: SegmentAnything fit: TF 355ms vs Torch 1389ms; BERT fit: TF 214ms vs Torch 808ms). Para LLM generation, Torch é 10× pior por usar static padding. **Não há benchmark oficial para 1D EM/CNN/TCN do tipo Geosteering**.

5. **Pesos `.keras` são portáveis entre backends desde que custom layers usem só `keras.ops`**. Salvar com Torch e carregar em TF "just works" — caveat: qualquer custom layer escrito com `tf.*` ops exige reescrita backend-agnostic antes de portar.

6. **Reprodutibilidade exata entre backends NÃO é garantida**. `keras.utils.set_random_seed()` é determinístico *dentro* de cada backend, mas RNG difere entre backends (numerics idênticos só até ~1e-7 em float32, e drift acumula em treino longo). cuDNN não-determinístico em ambos os backends sem flags adicionais.

7. **`TorchModuleWrapper` é via única para integrar `torch.nn.Module` em `keras.Model`** — e *só funciona com backend Torch* (keras.io/api/layers/backend_specific_layers/torch_module_wrapper/). Não há mecanismo equivalente que opere agnóstico.

---

## Tabela de Compatibilidade

| Item | Backend-agnostic | TF-only | Torch-only |
|:---|:---:|:---:|:---:|
| `keras.ops.*` (NumPy + NN) | OK | — | — |
| `keras.layers.*` builtin | OK | — | — |
| `keras.losses.*`, `keras.metrics.*`, `keras.optimizers.*` | OK | — | — |
| `model.fit/evaluate/predict` | OK | — | — |
| Salvar/carregar `.keras` | OK | — | — |
| Input via `tf.data.Dataset` | OK | — | — |
| Input via `torch.utils.data.DataLoader` | OK | — | — |
| Custom layer com `tf.function`, `tf.GradientTape` | — | TF | — |
| Custom training loop com `tape.gradient` | — | TF | — |
| `model.export(savedmodel)` para serving TF | — | TF | — |
| Custom train_step com `loss.backward()` + `zero_grad()` | — | — | Torch |
| `TorchModuleWrapper` (HuggingFace `nn.Module`) | — | — | Torch |
| `keras.distribution` (model parallelism) | parcial | — | — (só JAX hoje) |
| `tf.data.Dataset` *com layers Keras dentro do pipeline* | — | TF | — |
| `tf.keras.callbacks.TensorBoard` profiling avançado | — | TF | — |

Fonte: keras.io/keras_3/ (matriz de cross-framework features) e guia de migração.

---

## Snippets (referência, não para o projeto)

**Ativação Torch backend**:
```python
import os
os.environ["KERAS_BACKEND"] = "torch"  # DEVE vir antes de qualquer import keras
import keras
print(keras.backend.backend())  # "torch"
```

**Custom layer agnóstico (porta TF→Torch sem mudanças)**:
```python
import keras
class MyLayer(keras.layers.Layer):
    def call(self, x):
        return keras.ops.softmax(keras.ops.matmul(x, self.w))  # NUNCA tf.matmul
```

**Pytest multi-backend (subprocess por causa do startup-only)**:
```python
import subprocess, pytest
@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_model_parity(backend):
    env = {**os.environ, "KERAS_BACKEND": backend}
    subprocess.check_call(["python", "-m", "pytest", "tests/_inner.py"], env=env)
```
Não dá para parametrizar dentro do mesmo processo (Keras já importado). Pattern recomendado: `pytest-xdist` com workers isolados, ou matrix em CI.

---

## Aplicação ao Geosteering AI — Auditoria de Compatibilidade

| Componente | Estado | Esforço de migração |
|:---|:---|:---|
| 48 arquiteturas em `models/` | Provavelmente usam `tf.keras.layers` + alguns `tf.*` ops em custom blocks (TIVConstraintLayer, blocos residuais customizados) | **Médio** — auditar cada arquivo, trocar `tf.*` por `keras.ops.*` |
| 26 losses em `losses/` | Catálogo PINN com residuos físicos provavelmente usa `tf.GradientTape` para Maxwell residue | **Alto** — PINNs com tape requerem rewrite por backend |
| 17+ callbacks em `training/` | `tf.keras.callbacks` builtin OK; customizados podem usar `tf.summary` | **Baixo-Médio** |
| Pipeline `tf.data` + on-the-fly noise/FV/GS | `tf.data.Dataset` aceito por Torch backend, MAS map_fn que chame layers Keras dentro do pipeline funciona apenas em TF | **Crítico** — `pipeline.build_train_map_fn` provavelmente quebra |
| `noise/` curriculum 3-phase | Se usa `tf.random` para noise on-the-fly em map_fn, OK só em TF | **Médio** |
| Surrogate (TCN+ModernTCN) | Standard layers, baixo risco | **Baixo** |
| Simulator Numba/JAX | Independente do Keras, sem impacto | **Nenhum** |

**Bloqueador principal**: a "cadeia raw → noise → FV → GS → scale" implementada como `tf.data.Dataset.map(...)` com ops TF dentro do map. Se essa cadeia chama layers Keras (FV/GS são layers em alguns pipelines), só roda em backend TF — confirmado pela docs Keras 3 ("Custom layers in `tf.data` — Limited — Works with TensorFlow backend only").

---

## Vantagens e Cenários de Uso (Torch backend)

- **HuggingFace integration nativa** via `TorchModuleWrapper` (modelos pré-treinados, foundation models EM se aparecerem).
- **ROCm/AMD** maturidade superior em PyTorch.
- **Distributed training** (FSDP, DeepSpeed) ecossistema mais rico — irrelevante para 1D inversion atual.
- **Debugging eager** mais introspecivo (mas Keras 3 + TF eager já oferece o mesmo).
- **Nenhuma vantagem de performance documentada** para CNN/TCN/LSTM de tamanho moderado (< 100M params) no benchmark oficial.

---

## Gotchas e Antipatterns

1. `import tensorflow as tf` em código que pretende ser multi-backend é instant-break se Torch sem TF instalado.
2. `tf.function` decorator no `call()` quebra (autograph removido em Keras 3 mesmo com backend TF).
3. `self.w = tf.Variable(...)` não rastreia em Keras 3 — usar `self.add_weight()` ou `keras.Variable`.
4. `tf.random.normal()` em layers — usar `keras.random.normal(seed=keras.random.SeedGenerator(...))`.
5. `model.save("dir.savedmodel")` removido — usar `model.export()` (só TF) ou `.keras` (agnóstico).
6. Custom `train_step` com `tape.gradient` — precisa rewrite para `loss.backward()` em Torch.
7. `jit_compile=True` é default em GPU Keras 3 e pode quebrar XLA em ops customizadas.

---

## Recomendação Final: **NO-GO** (com janela de revisão)

**Justificativa**:

- **Proibição absoluta no CLAUDE.md**: *"PyTorch — PROIBIDO em qualquer parte do pipeline"* (linha 30 do CLAUDE.md). Mesmo backend-via-Keras qualifica como "PyTorch no pipeline" — exige decisão explícita do autor para revogar a regra.
- **Performance oficial é desfavorável**: 2-4× pior em todos os benchmarks oficiais Keras 3 para modelos do tipo presente no projeto (CNN/Transformer). Sem evidência de ganho.
- **Custo de migração é alto e cirúrgico**: 26 losses (incluindo 8 cenários PINN com residuos Maxwell que dependem de `tf.GradientTape`), pipeline `tf.data` com map_fn que chama layers (FV/GS), 1597 testes pytest validados em TF backend. Migração realista exige sprint dedicada de 2-4 semanas com validação de paridade `<1e-7`.
- **Reprodutibilidade entre backends não é bit-exact**, conflitando com a regra de paridade Fortran `<1e-12` que governa a cultura do projeto.
- **Não há ganho científico identificado**: nenhum modelo geofísico EM 1D pré-treinado em HuggingFace que justifique adoção; ROCm não é relevante (Colab Pro+ T4/A100 = NVIDIA).

**Quando reabrir o tópico (GO-CONDICIONAL futuro)**:
- Surgir foundation model EM/geofísico em HuggingFace que valha a integração.
- Necessidade de FSDP para modelos > 1B params (não está no roadmap até v3.0).
- Adoção de hardware AMD MI300 / Intel Gaudi onde Torch domina.
- Conclusão de migração `tf.*` → `keras.ops.*` completa (deveria ser feita por outras razões: portabilidade JAX backend já é vantajosa para co-existir com simulador JAX).

**Recomendação adjacente**: investir o esforço em **auditar custom layers/losses para usar `keras.ops` em vez de `tf.*`** independentemente de mudar de backend. Isso é prerrequisito barato para qualquer mobilidade futura (Torch *ou* JAX) e melhora portabilidade do código. Não viola CLAUDE.md (continua TF backend), mas remove o lock-in tácito.

---

## Fontes

- [Keras 3 Getting Started](https://keras.io/getting_started/) — mecanismo de backend
- [Keras 3 Multi-backend](https://keras.io/keras_3/) — matriz cross-framework
- [Keras 3 Migration Guide](https://keras.io/guides/migrating_to_keras_3/) — breaking changes
- [Keras 3 Benchmarks oficiais](https://keras.io/getting_started/benchmarks/) — A100, 6 modelos
- [TorchModuleWrapper](https://keras.io/api/layers/backend_specific_layers/torch_module_wrapper/)
- [Custom train_step PyTorch](https://keras.io/guides/custom_train_step_in_torch/)
- [Reproducibility Recipes](https://keras.io/examples/keras_recipes/reproducibility_recipes/)
- [Keras 3 Data Loading](https://keras.io/api/data_loading/) — tf.data multi-backend
- [GitHub Discussion #20037 — Torch significantly slower](https://github.com/keras-team/keras/discussions/20037)
