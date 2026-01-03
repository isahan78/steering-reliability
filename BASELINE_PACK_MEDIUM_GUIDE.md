# Running Baseline Pack on GPT-2 Medium

Quick guide for running the baseline pack experiment on GPT-2 Medium (355M parameters).

## What Changed from GPT-2 Small

### 1. Model
- **GPT-2 Small**: 124M params, 12 layers
- **GPT-2 Medium**: 355M params, 24 layers

### 2. Configuration
- **Layer**: 16 (middle layer of 24 total)
- **Batch size**: 4 (reduced from 8 due to larger model)
- **Config file**: `configs/gpt2_medium_baseline_pack.yaml`
- **Output dir**: `artifacts/baselines_medium/`

### 3. Bug Fix Applied
Fixed BOS token extraction bug in `generation.py` that was causing completions to include `<|endoftext|>` prefix and repeated prompts.

## Running in Google Colab

### Option 1: Use New Notebook (Recommended)

1. **Upload notebook**: `colab_baseline_pack_medium.ipynb`
2. **Runtime**: T4 GPU (40-60 min) or A100 (25-35 min)
3. **Run all cells**

The notebook automatically:
- Loads GPT-2 Medium
- Uses layer 16
- Runs all baselines
- Generates figures
- Downloads results

### Option 2: Modify Existing Notebook

In `colab_baseline_pack.ipynb`, change the run command:

```python
!PYTHONPATH=/content/steering-reliability/src python scripts/run_baseline_pack.py \
  --config configs/gpt2_medium_baseline_pack.yaml \
  --layer 16 \
  --alphas 0 1 4 8 \
  --n_harm_test 50 \
  --n_benign 50 \
  --n_random 10 \
  --seed 0 \
  --include_benign_contrast \
  --output_dir artifacts/baselines_medium
```

## Running Locally

```bash
# Make sure you're in the repo root
cd steering-reliability

# Run baseline pack for GPT-2 Medium
python scripts/run_baseline_pack.py \
  --config configs/gpt2_medium_baseline_pack.yaml \
  --layer 16 \
  --alphas 0 1 4 8 \
  --n_harm_test 50 \
  --n_benign 50 \
  --n_random 10 \
  --seed 0 \
  --include_benign_contrast \
  --output_dir artifacts/baselines_medium

# Generate figures
python -m steering_reliability.analysis.plot_baseline_pack \
  --in_parquet artifacts/baselines_medium/baseline_pack_results.parquet \
  --out_dir artifacts/baselines_medium/figures
```

## Expected Results

### If Direction is Specific (Good Result)

At α=8.0, harm_test:
- **Learned**: 85-95% refusal
- **Random**: 30-50% refusal
- **Gap**: >30% (strong evidence)

### If Direction is Not Specific (Concerning)

- **Learned**: ~50% refusal
- **Random**: ~50% refusal
- **Gap**: <10% (weak/no evidence)

## Output Files

```
artifacts/baselines_medium/
├── learned_layer16.pt                  # Learned direction
├── learned_layer16.json                # Metadata
├── shuffled_layer16.pt                 # Shuffled direction
├── shuffled_layer16.json
├── benign_contrast_layer16.pt          # Benign contrast direction
├── benign_contrast_layer16.json
├── baseline_pack_results.parquet       # Full results (5200 rows)
├── baseline_pack_summary.csv           # Aggregated summary
└── figures/
    ├── direction_specificity.png       # Main figure for paper
    ├── benign_preservation.png         # Helpfulness check
    └── baseline_pack_table.csv         # Summary table for paper
```

## Comparing GPT-2 Small vs Medium

If you've run both experiments:

```python
import pandas as pd

# Load both results
df_small = pd.read_parquet('artifacts/baselines/baseline_pack_results.parquet')
df_medium = pd.read_parquet('artifacts/baselines_medium/baseline_pack_results.parquet')

# Compare at α=8, harm_test
for model_name, df in [('Small', df_small), ('Medium', df_medium)]:
    harm_8 = df[(df['alpha'] == 8) & (df['split'] == 'harm_test')]

    learned = harm_8[harm_8['direction_type'] == 'learned']['is_refusal'].mean()
    random = harm_8[harm_8['direction_type'] == 'random']['is_refusal'].mean()
    gap = learned - random

    print(f"{model_name}: Learned={learned:.1%}, Random={random:.1%}, Gap={gap:+.1%}")
```

## Troubleshooting

### Out of Memory Error

Reduce batch size in config:
```yaml
generation:
  batch_size: 2  # or even 1
```

### Results Look Wrong

Check sample completions:
```python
df = pd.read_parquet('artifacts/baselines_medium/baseline_pack_results.parquet')
sample = df[(df['alpha'] == 0) & (df['split'] == 'benign')].iloc[0]
print(f"Prompt: {sample['prompt']}")
print(f"Completion: {sample['completion']}")
```

Completions should NOT start with `<|endoftext|>` (bug was fixed).

### Runtime Too Long

Reduce prompt count:
```bash
--n_harm_test 30 --n_benign 30 --n_random 5
```

This cuts runtime by ~40%.

## Next Steps

1. **Run experiment** (40-60 min on T4)
2. **Analyze results** (check gap > 30%)
3. **Add to MATS application**:
   - Insert `direction_specificity.png`
   - Insert `baseline_pack_table.csv`
   - Add 4-6 sentence description
4. **Compare models** (if you ran both Small and Medium)

---

**Questions?** Check `BASELINE_PACK_README.md` for full details.
