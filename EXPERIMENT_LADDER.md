# Experiment Iteration Ladder

Progressive experimentation strategy for efficient research on Google Colab.

---

## Philosophy

Instead of running one massive 2-hour experiment, build up iteratively:
1. **Fast feedback loops** (5-15 min experiments)
2. **Learn from each iteration** (which layer works best?)
3. **Focus compute** on promising configs
4. **Reduce risk** of Colab disconnects

---

## The Ladder

### Level 1: Smoke Test (5 minutes)

**Goal:** Verify pipeline works end-to-end

```bash
!python scripts/run_all.py --config configs/smoke.yaml
```

**Config:**
- 1 layer (layer 12 - middle of model)
- 2 alphas (0, 2)
- 1 intervention (add)
- 20 prompts per split
- **Total:** ~180 completions

**What to check:**
- ✅ Pipeline completes without errors
- ✅ Baseline refusal rate ~10-30% on harm_test
- ✅ Alpha=2 increases refusal rate
- ✅ Plots generated

**Output:** `artifacts/runs/smoke_test/`

---

### Level 2: Layer Comparison (15 minutes)

**Goal:** Find which layer provides best steering effect

```bash
!python scripts/run_all.py --config configs/layer_test.yaml
```

**Config:**
- 3 layers (8, 12, 16) - early, middle, late
- 3 alphas (0, 2, 4)
- 1 intervention (add)
- 50 prompts per split
- **Total:** ~900 completions

**What to analyze:**
- Which layer gives highest refusal on harm_test?
- Which layer has lowest side effects on benign?
- Is there a generalization gap (harm_train vs harm_test)?

**Decision:** Pick best layer for Level 3

**Output:** `artifacts/runs/layer_comparison/`

---

### Level 3: Alpha Sweep (30 minutes)

**Goal:** Fine-tune steering strength on best layer

```bash
# FIRST: Edit configs/alpha_sweep.yaml
# Change: layers: [12]  →  layers: [YOUR_BEST_LAYER]

!python scripts/run_all.py --config configs/alpha_sweep.yaml
```

**Config:**
- 1 layer (best from Level 2)
- 5 alphas (0, 1, 2, 4, 8)
- 2 interventions (add, ablate)
- 100 prompts per split
- **Total:** ~3,000 completions

**What to analyze:**
- Optimal alpha (highest refusal, lowest side effects)
- Additive vs ablation: which works better?
- Tradeoff curve: refusal vs helpfulness

**Decision:** Identify best (layer, alpha, intervention) config

**Output:** `artifacts/runs/alpha_sweep/`

---

### Level 4: Full Experiment (1-2 hours)

**Goal:** Publication-ready results with complete sweep

```bash
!python scripts/run_all.py --config configs/full.yaml
```

**Config:**
- 3 layers (8, 12, 16)
- 5 alphas (0, 1, 2, 4, 8)
- 2 interventions (add, ablate)
- All prompts (150/150/200)
- **Total:** ~12,000 completions

**What to analyze:**
- Confirm findings from Levels 2-3 on full dataset
- Generate all publication plots
- Run statistical significance tests
- Write up results

**Output:** `artifacts/runs/full_gpt2_medium/`

---

## Colab Workflow

### Initial Run (Level 1)

```python
# Cell 1: Clone and setup
!git clone https://github.com/isahan78/steering-reliability.git
%cd steering-reliability

# Cell 2: Install dependencies
!pip uninstall -y numpy pandas datasets transformer-lens transformers pyarrow scikit-learn -q
!pip install --no-cache-dir numpy pandas torch transformer-lens transformers datasets matplotlib seaborn pyyaml tqdm pyarrow scikit-learn
import sys
sys.path.insert(0, '/content/steering-reliability/src')

# Cell 3: Verify imports
from steering_reliability.model import load_model
print("✅ Imports work!")

# Cell 4: Run smoke test
!python scripts/run_all.py --config configs/smoke.yaml

# Cell 5: Check results
!ls -lh artifacts/runs/smoke_test/
!head artifacts/tables/summary.csv
```

### Iteration Cycle

After each level:

1. **Download results:**
```python
!zip -r results_level_X.zip artifacts/
from google.colab import files
files.download('results_level_X.zip')
```

2. **Analyze locally** (or in Colab):
```python
import pandas as pd
summary = pd.read_csv('artifacts/tables/summary.csv')

# For layer comparison:
summary[summary['split'] == 'harm_test'].groupby('layer')['is_refusal_mean'].max()

# For alpha sweep:
summary[summary['split'] == 'harm_test'].groupby('alpha')['is_refusal_mean'].plot()
```

3. **Update config for next level** based on findings

4. **Run next level**

---

## Quick Analysis Commands

### After Layer Test:

```python
import pandas as pd
summary = pd.read_csv('artifacts/tables/summary.csv')

# Find best layer (highest refusal on harm_test, alpha=4)
best = summary[
    (summary['split'] == 'harm_test') &
    (summary['alpha'] == 4)
].sort_values('is_refusal_mean', ascending=False)

print("Best layer:")
print(best[['layer', 'is_refusal_mean', 'is_helpful_mean']].head(1))
```

### After Alpha Sweep:

```python
import pandas as pd
summary = pd.read_csv('artifacts/tables/summary.csv')

# Plot tradeoff curve
harm_test = summary[summary['split'] == 'harm_test']
benign = summary[summary['split'] == 'benign']

merged = harm_test.merge(
    benign,
    on=['layer', 'alpha', 'intervention_type'],
    suffixes=('_harm', '_benign')
)

import matplotlib.pyplot as plt
plt.scatter(
    1 - merged['is_helpful_mean_benign'],  # Helpfulness drop
    merged['is_refusal_mean_harm']         # Refusal rate
)
plt.xlabel('Helpfulness Drop (Benign)')
plt.ylabel('Refusal Rate (Harm Test)')
plt.title('Tradeoff Curve')
plt.show()
```

---

## Expected Timeline

| Level | Config | Completions | A100 Time | Total Time |
|-------|--------|-------------|-----------|------------|
| 1. Smoke | smoke.yaml | ~180 | 5 min | 5 min |
| 2. Layer | layer_test.yaml | ~900 | 15 min | 20 min |
| 3. Alpha | alpha_sweep.yaml | ~3,000 | 30 min | 50 min |
| 4. Full | full.yaml | ~12,000 | 90 min | 140 min |

**Total research time:** ~2.5 hours (but with 3 intermediate checkpoints for analysis!)

---

## Tips

1. **Always download results** after each level before moving on
2. **Commit configs** to Git after updating them
3. **Keep notes** on what you learn from each iteration
4. **Don't skip levels** - each builds on the previous
5. **If Colab disconnects**, you only lose current level (not entire experiment)

---

## Next Steps

Start with:
```bash
!python scripts/run_all.py --config configs/smoke.yaml
```

Then work your way up the ladder!
