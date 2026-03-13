# Crystal Mancer — Autoresearch Program

## Objective
Systematically scale Crystal Mancer from a 0.5M-param proof-of-concept to a 50M-param SOTA crystal generation model. Each stage must PASS validation before progressing.

## Philosophy
**Small steps, scaleable.** Every architecture change runs at the smallest config first. If it improves val_loss → scale up. If not → revert and try next.

---

## Stage 1: Baseline (0.5M params) — Must Pass First
**Goal**: Establish a working baseline with measurable val_loss.

```json
{
    "stage": 1,
    "model": "model_v2",
    "config": "tiny",
    "hidden_dim": 64,
    "num_interaction_layers": 2,
    "num_attention_heads": 2,
    "edge_dim": 32,
    "num_rbf": 20,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 50,
    "max_training_time_seconds": 300
}
```

**Pass criteria**: val_loss < 10.0 (just proves the pipeline works).

### Sweep within Stage 1
Try these variations (one at a time, keep best):
1. Learning rate: [5e-4, 1e-3, 2e-3, 5e-3]
2. Dropout: [0.0, 0.05, 0.1, 0.2]
3. Aggregation: ["add", "mean"]
4. Activation: ["SiLU", "GELU"]

---

## Stage 2: Small (1.3M params)
**Goal**: Real learning, validate that multi-head attention helps.

```json
{
    "stage": 2,
    "model": "model_v2",
    "config": "small",
    "hidden_dim": 128,
    "num_interaction_layers": 4,
    "num_attention_heads": 4,
    "edge_dim": 64,
    "num_rbf": 40,
    "learning_rate": "<best from stage 1>",
    "batch_size": 32,
    "num_epochs": 100,
    "max_training_time_seconds": 600
}
```

**Pass criteria**: val_loss < 5.0 AND val_loss improves over Stage 1.

### Sweep within Stage 2
1. num_interaction_layers: [3, 4, 5, 6]
2. num_attention_heads: [2, 4, 8]
3. With/without conditioning: [true, false]
4. With/without element embeddings: [true, false]
5. Noise schedule: ["cosine", "linear"] (if running diffusion)

---

## Stage 3: Medium-Small (3M params)
**Goal**: Scale width, validate that more capacity helps.

```json
{
    "stage": 3,
    "hidden_dim": 192,
    "num_interaction_layers": 6,
    "num_attention_heads": 8,
    "edge_dim": 96,
    "num_rbf": 50,
    "batch_size": 16,
    "num_epochs": 100,
    "max_training_time_seconds": 1200
}
```

**Pass criteria**: val_loss improves over Stage 2.

### Ablations at Stage 3
1. With/without Hamiltonian loss: compare val_loss
2. With/without physics loss (charge neutrality + Goldschmidt): compare
3. With/without global features: compare
4. RBF type: Bessel vs Gaussian smearing

---

## Stage 4: Medium (8.5M params) — Colab T4
**Goal**: Serious training with proper dataset (10K+ structures).

```json
{
    "stage": 4,
    "hidden_dim": 256,
    "num_interaction_layers": 8,
    "num_attention_heads": 8,
    "edge_dim": 128,
    "num_rbf": 64,
    "batch_size": 16,
    "num_epochs": 200,
    "hardware": "colab_t4"
}
```

**Pass criteria**: Property prediction MAE < 50mV for overpotential.

### Sweep within Stage 4
1. Conditioning method: ["cross_attention", "addition", "film"]
2. Pooling: ["mean", "add", "attention"]
3. Scheduler: ["cosine_annealing", "reduce_on_plateau", "warmup_cosine"]

---

## Stage 5: Large (25M params) — Colab A100
**Goal**: Scale to competitive size. Add diffusion score training.

```json
{
    "stage": 5,
    "hidden_dim": 384,
    "num_interaction_layers": 10,
    "num_attention_heads": 12,
    "edge_dim": 192,
    "num_rbf": 96,
    "batch_size": 8,
    "num_epochs": 500,
    "hardware": "colab_a100",
    "training_mode": "diffusion_score_matching"
}
```

**Pass criteria**: Generated structures pass CHGNet stability check >50%.

---

## Stage 6: Production (50M params) — MatterGen Scale
**Goal**: SOTA crystal generation with performance conditioning.

```json
{
    "stage": 6,
    "hidden_dim": 512,
    "num_interaction_layers": 12,
    "num_attention_heads": 16,
    "edge_dim": 256,
    "num_rbf": 128,
    "batch_size": 4,
    "num_epochs": 1000,
    "hardware": "colab_a100",
    "training_mode": "diffusion + rl_finetuning"
}
```

**Pass criteria**: Generated crystals validated via DFT.

---

## Fixed Components (do NOT modify)
- `crystalmancer/graph/featurizer.py` — Atom and bond features
- `crystalmancer/graph/graph_builder.py` — CIF-to-graph conversion
- `crystalmancer/graph/dataset.py` — Dataset loading

## Editable Components
- `crystalmancer/model/model_v2.py` — Architecture (ModelConfig)
- `crystalmancer/model/physics_loss.py` — Physics constraint losses
- `crystalmancer/model/train.py` — Training loop + optimizer
- `train_config.json` — Hyperparameters

## How to Run One Experiment
```bash
cd /Users/roshan/Documents/Code/crystalMancer
export OPENROUTER_API_KEY='sk-or-v1-...'
export MP_API_KEY='V9oxuqCRq4MvdyDAzrE2FYLlvmw29wjz'

python -c "
from crystalmancer.model.model_v2 import CrystalMancerV2, ModelConfig
from crystalmancer.graph.dataset import CrystalDataset
from crystalmancer.model.train import Trainer, TrainConfig
from torch_geometric.loader import DataLoader

# Pick stage config
config = ModelConfig.small()  # Stage 2

# Load data
dataset = CrystalDataset()
train_idx, val_idx, _ = dataset.get_splits()
train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=32, shuffle=True)
val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=32)

# Train
model = CrystalMancerV2(config)
train_config = TrainConfig(num_epochs=50, max_training_time_seconds=300)
trainer = Trainer(train_config)
result = trainer.train(model, train_loader, val_loader)
print(f'RESULT: val_loss={result[\"best_val_loss\"]:.6f} params={sum(p.numel() for p in model.parameters()):,}')
"
```

## Success Metric
```
val_loss = masked_mean(MSE for each available target)
```
Lower is better. Report val_loss AND parameter count for every experiment.

## Rules
1. **Never skip stages** — each must pass before advancing
2. **One change at a time** — isolate what helps
3. **Always report params** — efficiency matters (val_loss per million params)
4. **Save checkpoints** — best model at each stage goes to `checkpoints/`
5. **Time budget per experiment**: Stage 1-2: 5 min, Stage 3: 20 min, Stage 4+: unlimited
6. **Use MPS on Mac** for Stages 1-3, GPU on Colab for Stages 4+
