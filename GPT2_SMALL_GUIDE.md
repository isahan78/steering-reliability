# GPT-2 Small Quick Start Guide

**Use GPT-2 Small for fast iteration and development!**

---

## Why GPT-2 Small?

| Model | Parameters | Layers | Speed | Use Case |
|-------|------------|--------|-------|----------|
| **gpt2** (small) | 124M | 12 | **3-4x faster** | Development & iteration |
| gpt2-medium | 355M | 24 | Baseline | Final validation |

**Strategy:** Develop on small → Validate on medium

---

## Quick Iteration Configs

### 1. Smoke Test (2-3 minutes)
```bash
!python scripts/run_all.py --config configs/gpt2_small_smoke.yaml
```
- 20 prompts per split
- 2 layers, 3 alphas
- **Goal:** Verify everything works

### 2. Layer Test (10-12 minutes)
```bash
!python scripts/run_all.py --config configs/gpt2_small_layer_test.yaml
```
- 50 prompts per split
- 4 layers (4, 6, 8, 10)
- **Goal:** Find best layer

### 3. Alpha Sweep (20-25 minutes)
```bash
# First, edit config to use best layer from step 2
!python scripts/run_all.py --config configs/gpt2_small_alpha_sweep.yaml
```
- 100 prompts per split
- 5 alphas, 2 interventions
- **Goal:** Find optimal steering strength

### 4. Full Sweep (30-40 minutes)
```bash
!python scripts/run_all.py --config configs/gpt2_small_full.yaml
```
- All prompts (150/150/200)
- Complete sweep
- **Goal:** Development results

---

## Timeline Comparison

### GPT-2 Small (Recommended for iteration)
- ✅ Smoke: 2-3 min
- ✅ Layer: 10-12 min
- ✅ Alpha: 20-25 min
- ✅ Full: 30-40 min
- **Total:** ~1 hour with analysis breaks

### GPT-2 Medium (For final validation)
- Smoke: 5-7 min
- Layer: 30-40 min
- Alpha: 45-60 min
- Full: 90-120 min
- **Total:** ~3 hours

---

## Complete Colab Workflow

### Initial Setup (Run Once)

```python
# 1. Clone repo
!git clone https://github.com/isahan78/steering-reliability.git
%cd steering-reliability

# 2. Install dependencies
!pip uninstall -y numpy pandas datasets transformer-lens transformers pyarrow scikit-learn -q
!pip install --no-cache-dir numpy pandas torch transformer-lens transformers datasets matplotlib seaborn pyyaml tqdm pyarrow scikit-learn

import sys
sys.path.insert(0, '/content/steering-reliability/src')

# 3. Verify imports
from steering_reliability.model import load_model
print("✅ Ready!")
```

### Run Experiments

```python
# Level 1: Quick sanity check
!python scripts/run_all.py --config configs/gpt2_small_smoke.yaml

# Level 2: Find best layer
!python scripts/run_all.py --config configs/gpt2_small_layer_test.yaml

# Level 3: Tune alpha (after editing config with best layer)
!python scripts/run_all.py --config configs/gpt2_small_alpha_sweep.yaml

# Level 4: Full development results
!python scripts/run_all.py --config configs/gpt2_small_full.yaml
```

### Quick Analysis After Each Level

```python
import pandas as pd

# Load results
summary = pd.read_csv('artifacts/tables/summary.csv')

# Find best layer (after Level 2)
harm_test = summary[
    (summary['split'] == 'harm_test') &
    (summary['alpha'] == 4)
].sort_values('is_refusal_mean', ascending=False)

print("Best layers:")
print(harm_test[['layer', 'is_refusal_mean', 'is_helpful_mean']].head(3))
```

### Download Results

```python
!zip -r gpt2_small_results.zip artifacts/
from google.colab import files
files.download('gpt2_small_results.zip')
```

---

## When to Switch to GPT-2 Medium

**After GPT-2 Small experiments, you'll know:**
1. Which layers work best
2. Optimal alpha values
3. Whether additive or ablation is better

**Then run ONE targeted experiment on GPT-2 Medium:**

```yaml
# configs/medium_targeted.yaml
model:
  name: gpt2-medium

experiment:
  layers: [12, 16]  # Best layers from small (scaled: 6→12, 8→16)
  interventions: [add]  # Best intervention from small
  alphas: [0, 2, 4]  # Best alphas from small
```

This takes ~30 minutes instead of 2 hours!

---

## Layer Mapping: Small → Medium

GPT-2 Small has 12 layers, Medium has 24 (exactly 2x).

Scale your findings:

| Small (12 layers) | Medium (24 layers) | Position |
|-------------------|--------------------| ---------|
| Layer 4 | Layer 8 | Early |
| Layer 6 | Layer 12 | Early-Mid |
| Layer 8 | Layer 16 | Mid-Late |
| Layer 10 | Layer 20 | Late |

---

## Best Practices

1. **Always start with GPT-2 Small** for new experiments
2. **Iterate quickly** - each experiment is 2-25 minutes
3. **Analyze after each level** - don't run all 4 blindly
4. **Download results** after each level
5. **Only run GPT-2 Medium** when you know what configs to test

---

## Example Session

```
09:00 - Start Colab, run smoke test (3 min)
09:03 - Review results, looks good ✓
09:05 - Run layer test (12 min)
09:17 - Analyze: Layer 6 and 8 are best
09:20 - Edit alpha_sweep.yaml to use layer 6
09:22 - Run alpha sweep (25 min)
09:47 - Analyze: Alpha=4 is optimal, additive > ablation
09:50 - Run full sweep to confirm (40 min)
10:30 - Download all results

10:35 - Create targeted medium config with best params
10:37 - Run medium experiment (30 min)
11:07 - Done! Publication results ready
```

**Total time:** 2 hours with full analysis vs 5+ hours running medium blindly

---

Ready to start? Run the smoke test:

```bash
!python scripts/run_all.py --config configs/gpt2_small_smoke.yaml
```
