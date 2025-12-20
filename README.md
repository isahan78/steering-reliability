# Steering Reliability Under Distribution Shift

A research pipeline for measuring how linear steering directions generalize across distribution-shifted prompts in small transformer language models.

## Overview

This project implements a rigorous testing framework for **steering vector reliability**, measuring:
- **Generalization**: Do directions learned on one prompt distribution transfer to another?
- **Side effects**: Do interventions degrade performance on benign queries?
- **Layer sensitivity**: Which layers provide stable and effective steering?

**Key features:**
- Contrastive forced-prefix direction construction
- Two intervention types: additive steering + projection ablation
- Transparent rule-based metrics (no black-box judges)
- GPU-optimized with TransformerLens hooks
- Full reproducibility with deterministic seeds

---

## Quick Start

### Installation

```bash
# Clone and navigate
cd steering-reliability

# Install package (recommended: use a virtual environment)
pip install -e .

# Or install dependencies directly
pip install torch transformer-lens transformers datasets pandas numpy matplotlib seaborn pyyaml tqdm pyarrow scikit-learn
```

### Run Complete Pipeline (CPU/GPU)

```bash
# 1. Generate prompt datasets
python scripts/make_prompts.py --out_dir data/prompts --n_train 150 --n_test 150 --n_benign 200

# 2. Run full pipeline (baseline + sweep + plots)
python scripts/run_all.py --config configs/default.yaml
```

**Expected runtime:**
- CPU: ~2-4 hours (pythia-160m)
- GPU: ~20-40 minutes (pythia-160m)

**Output:**
- `artifacts/runs/run_001/baseline_results.parquet`
- `artifacts/runs/run_001/sweep_results.parquet`
- `artifacts/figures/*.png` (generalization gap, tradeoff curve, heatmaps)
- `artifacts/tables/summary.csv`

---

## Project Structure

```
steering-reliability/
├── configs/
│   └── default.yaml           # Experiment configuration
├── data/
│   └── prompts/               # JSONL prompt files
├── src/steering_reliability/
│   ├── config.py              # Configuration management
│   ├── model.py               # Model loading with auto GPU detection
│   ├── data.py                # JSONL data loaders
│   ├── generation.py          # Batched text generation
│   ├── directions/
│   │   └── build_direction.py # Contrastive direction construction
│   ├── interventions/
│   │   └── steer.py           # TransformerLens steering hooks
│   ├── metrics/
│   │   ├── refusal.py         # Rule-based refusal detection
│   │   ├── helpfulness.py     # Helpfulness scoring
│   │   └── confounds.py       # Length/entropy metrics
│   ├── experiments/
│   │   ├── run_baseline.py    # Baseline (no intervention)
│   │   └── run_sweep.py       # Layer × alpha × intervention sweep
│   └── analysis/
│       ├── aggregate.py       # Results aggregation
│       └── plots.py           # Visualization functions
├── scripts/
│   ├── make_prompts.py        # Generate prompt datasets
│   ├── run_all.py             # One-shot pipeline runner
│   └── make_plots.py          # Generate plots from results
├── notebooks/                  # Jupyter notebooks (coming next)
└── artifacts/                  # Generated outputs
```

---

## Usage

### 1. Generate Prompts

```bash
python scripts/make_prompts.py \
    --out_dir data/prompts \
    --n_train 150 \
    --n_test 150 \
    --n_benign 200 \
    --seed 0
```

Creates three JSONL files with distribution shift between `harm_train` (direct phrasing) and `harm_test` (indirect/hypothetical phrasing).

### 2. Run Experiments

**Option A: Full Pipeline**
```bash
python scripts/run_all.py --config configs/default.yaml
```

**Option B: Step-by-Step**
```bash
# Baseline
python -m steering_reliability.experiments.run_baseline --config configs/default.yaml

# Sweep
python -m steering_reliability.experiments.run_sweep --config configs/default.yaml

# Plots
python scripts/make_plots.py --runs_dir artifacts/runs/run_001 --out_dir artifacts/figures
```

### 3. Configuration

Edit `configs/default.yaml` to customize:
- **Model**: Switch between `pythia-160m`, `gpt2`, etc.
- **Layers**: Test different layer combinations
- **Alphas**: Adjust steering strengths
- **Prefixes**: Modify refusal/compliance prefixes
- **Device**: `auto`, `cuda`, `cpu`, or `mps`

Example quick test config:
```yaml
experiment:
  layers: [8]           # Test single layer
  alphas: [0, 1, 2]     # Fewer alpha values
  interventions: [add]  # Just additive steering

data:
  max_prompts_per_split: 50  # Smaller dataset
```

---

## Key Concepts

### Steering Direction Construction

**Contrastive Forced-Prefix Method:**

For each prompt `p`:
1. Create two sequences:
   - `p + " I can't help with that."` (refusal)
   - `p + " Sure, here's how:"` (compliance)
2. Run teacher-forcing to get residual stream activations at specified layer
3. Compute difference: `d = resid_refusal - resid_compliance`
4. Average across prompts: `v = mean(d)`
5. Normalize: `v = v / ||v||`

### Interventions

**Additive Steering:**
```
resid = resid + alpha * v
```

**Projection Ablation:**
```
proj = (resid · v) * v
resid = resid - alpha * proj
```

### Metrics

**Refusal Detection (rule-based):**
- Markers: "I can't", "sorry", "not able", etc.
- Score in [0, 1] where 1 = strong refusal

**Helpfulness (benign queries):**
- Length, structure (bullets/steps), absence of refusal
- Score in [0, 1] where 1 = helpful

**Confounds:**
- Character/token length
- Token entropy
- Average log probability

---

## Outputs

### Plots

1. **Generalization Gap**: `refusal_rate(harm_train) - refusal_rate(harm_test)` vs alpha
2. **Tradeoff Curve**: refusal_rate(harm_test) vs helpfulness_drop(benign)
3. **Heatmaps**: Layer × alpha grids for refusal/helpfulness rates

### Tables

- `summary.csv`: Aggregated metrics by (split, layer, alpha, intervention_type)
- Refusal rates, helpfulness rates, completion lengths

---

## Notebooks (GPU-Optimized)

Three Jupyter notebooks for interactive experimentation with GPU acceleration:

### 01_smoke_test.py
**Quick validation (10 minutes)**
- Tests pipeline on 20 prompts per split
- Builds one direction (layer 8)
- Compares baseline vs one steering config
- Perfect for verifying setup works

**How to use:**
```bash
# Local Jupyter
jupyter notebook notebooks/01_smoke_test.py

# Or copy cells into Google Colab
# (cells are marked with # %% for easy copying)
```

### 02_full_sweep.py
**Complete experiment (1-2 hours on GPU)**
- Full dataset (150/150/200 prompts)
- All layers × alphas × interventions
- Automatic GPU detection and memory monitoring
- Optional save to Google Drive

**Runtime estimates:**
- GPU (T4/V100): ~1-2 hours
- CPU: ~4-6 hours

### 03_analysis.py
**Generate plots and analysis (< 5 minutes)**
- Loads results from sweep
- Creates all publication-ready plots
- Generates summary report
- Shows qualitative examples

**Features:**
- Inline plot rendering
- Interactive comparisons
- Exportable markdown report
- Best configuration recommendations

---

## Research Acceptance Criteria

Per the PRD, results should demonstrate:
- ✅ At least one **generalization failure** OR **side effect tradeoff**
- ✅ At least **two sanity checks** (template robustness, entropy confounds, seed robustness)
- ✅ At least one **layer comparison** with non-trivial differences

---

## Citation

```bibtex
@software{steering_reliability_2025,
  title={Steering Reliability Under Distribution Shift},
  author={MATS Research},
  year={2025},
  url={https://github.com/yourusername/steering-reliability}
}
```

---

## License

MIT

---

## Troubleshooting

**CUDA out of memory:**
- Reduce `batch_size` in `configs/default.yaml`
- Use smaller model (gpt2 instead of pythia-160m)
- Set `max_prompts_per_split: 50`

**Slow on CPU:**
- Enable fast mode: `--n_train 60 --n_test 60 --n_benign 60`
- Test single layer: `layers: [8]`

**Import errors:**
- Ensure you ran `pip install -e .` from project root
- Check Python version >= 3.9
