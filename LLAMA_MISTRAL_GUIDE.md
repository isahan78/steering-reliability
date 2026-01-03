# Running Baseline Pack on Llama-2 & Mistral

Complete guide for running baseline pack experiments on modern 7B models.

## Quick Comparison

| Feature | GPT-2 Small | GPT-2 Medium | Mistral 7B | Llama-2 7B |
|---------|-------------|--------------|------------|------------|
| **Size** | 124M | 355M | 7B | 7B |
| **Layers** | 12 | 24 | 32 | 32 |
| **Layer Tested** | 10 | 16 | 16 | 16 |
| **Batch Size** | 8 | 4 | 2 | 2 |
| **VRAM Required** | ~2GB | ~4GB | ~15GB | ~15GB |
| **T4 Runtime** | 20-30 min | 40-60 min | 60-90 min | 60-90 min |
| **A100 Runtime** | 15-20 min | 25-35 min | 35-50 min | 35-50 min |
| **HF Token?** | ❌ No | ❌ No | ❌ No | ✅ **Yes** |
| **License** | MIT | MIT | Apache 2.0 | Meta License |

---

## ⭐ Recommended: Mistral 7B

**Why Mistral?**
- ✅ No HuggingFace authentication required
- ✅ Fully open-access (Apache 2.0)
- ✅ State-of-the-art performance
- ✅ Better instruction following than Llama-2
- ✅ Easier to set up in Colab

**Start here:** `colab_baseline_pack_mistral.ipynb`

---

## Running Mistral 7B

### Colab (Easiest)

1. **Upload notebook**: `colab_baseline_pack_mistral.ipynb`
2. **Set runtime**: T4 GPU (free) or A100 (Colab Pro)
3. **Run all cells**
4. **Wait**: ~60-90 min (T4) or ~35-50 min (A100)

### Local

```bash
cd steering-reliability
git pull  # Get latest configs

python scripts/run_baseline_pack.py \
  --config configs/mistral_7b_baseline_pack.yaml \
  --layer 16 \
  --alphas 0 1 4 8 \
  --n_harm_test 50 \
  --n_benign 50 \
  --n_random 10 \
  --seed 0 \
  --include_benign_contrast \
  --output_dir artifacts/baselines_mistral

# Generate figures
python -m steering_reliability.analysis.plot_baseline_pack \
  --in_parquet artifacts/baselines_mistral/baseline_pack_results.parquet \
  --out_dir artifacts/baselines_mistral/figures
```

---

## Running Llama-2 7B

### Prerequisites (IMPORTANT)

1. **Accept license**: Go to https://huggingface.co/meta-llama/Llama-2-7b-hf
   - Click "Agree and access repository"
   - Wait for approval (~5 minutes)

2. **Get HF token**: Go to https://huggingface.co/settings/tokens
   - Create new token (read access)
   - Copy token

### Colab

1. **Upload notebook**: `colab_baseline_pack_llama2.ipynb`
2. **Run authentication cell**: Paste your HF token
3. **Run all cells**
4. **Wait**: ~60-90 min (T4) or ~35-50 min (A100)

### Local

```bash
# Set HF token
export HF_TOKEN="your_token_here"
huggingface-cli login --token $HF_TOKEN

# Run experiment
python scripts/run_baseline_pack.py \
  --config configs/llama2_7b_baseline_pack.yaml \
  --layer 16 \
  --alphas 0 1 4 8 \
  --n_harm_test 50 \
  --n_benign 50 \
  --n_random 10 \
  --seed 0 \
  --include_benign_contrast \
  --output_dir artifacts/baselines_llama2
```

---

## Configuration Files

All configs are in `configs/`:

- **Mistral 7B**: `mistral_7b_baseline_pack.yaml`
- **Llama-2 7B**: `llama2_7b_baseline_pack.yaml`
- **GPT-2 Medium**: `gpt2_medium_baseline_pack.yaml`
- **GPT-2 Small**: `gpt2_small_alpha_sweep.yaml`

### Key Settings for 7B Models

```yaml
model:
  name: mistralai/Mistral-7B-v0.1  # or meta-llama/Llama-2-7b-hf
  device: auto
  dtype: float32  # Use bfloat16 if OOM

generation:
  batch_size: 2  # Reduced for 7B models
  max_new_tokens: 80

direction:
  refusal_prefix: " I cannot help with that."  # Updated for chat models
  compliance_prefix: " Sure, here's how:"
```

---

## Expected Results

### Mistral 7B (Hypothesis)

Mistral has stronger safety training, so we expect:

**At α=8.0, harm_test:**
- **Learned**: 90-98% refusal (strong safety)
- **Random**: 60-80% refusal (already safe baseline)
- **Gap**: 10-30% (moderate specificity)

**Interpretation:**
- Smaller gap is OK - Mistral already refuses many prompts
- Focus on: Does learned >> random?
- Check benign preservation (should be >95%)

### Llama-2 7B (Hypothesis)

Llama-2 7B has moderate safety tuning:

**At α=8.0, harm_test:**
- **Learned**: 85-95% refusal
- **Random**: 40-60% refusal
- **Gap**: 25-45% (good specificity)

**Interpretation:**
- Should show stronger gap than Mistral
- Llama-2 is more "steerable" than Mistral

---

## Troubleshooting

### Out of Memory (OOM)

**Solution 1: Reduce batch size**
```yaml
generation:
  batch_size: 1  # In config file
```

**Solution 2: Reduce prompts**
```bash
--n_harm_test 30 --n_benign 30 --n_random 5
```

**Solution 3: Use bfloat16** (if supported)
```yaml
model:
  dtype: bfloat16
```

**Solution 4: Use A100** (Colab Pro)
- 40GB VRAM vs 16GB on T4
- 2-3x faster

### Llama-2 Auth Error

**Error**: `401 Client Error: Unauthorized`

**Solution:**
1. Accept license: https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Wait 5-10 minutes for approval
3. Create new token: https://huggingface.co/settings/tokens
4. Re-run authentication cell

### Slow Download

First download takes time:
- Mistral 7B: ~14GB
- Llama-2 7B: ~13GB

Use A100 for faster download speeds.

### Results Look Strange

Check sample completions:
```python
df = pd.read_parquet('artifacts/baselines_mistral/baseline_pack_results.parquet')
sample = df[(df['alpha'] == 0) & (df['split'] == 'benign')].iloc[0]
print(f"Prompt: {sample['prompt']}")
print(f"Completion: {sample['completion']}")
```

Completions should:
- ✅ Start with the actual continuation (not prompt)
- ✅ NOT start with `<|endoftext|>` or `<s>`
- ✅ Be coherent and on-topic

---

## Model Comparison

After running experiments on multiple models:

```python
import pandas as pd

models = {
    'Mistral 7B': 'artifacts/baselines_mistral/baseline_pack_results.parquet',
    'Llama-2 7B': 'artifacts/baselines_llama2/baseline_pack_results.parquet',
    'GPT-2 Medium': 'artifacts/baselines_medium/baseline_pack_results.parquet'
}

print("MODEL COMPARISON (α=8, harm_test)")
print("="*80)

for model_name, path in models.items():
    try:
        df = pd.read_parquet(path)
        harm_8 = df[(df['alpha'] == 8) & (df['split'] == 'harm_test')]

        learned = harm_8[harm_8['direction_type'] == 'learned']['is_refusal'].mean()
        random = harm_8[harm_8['direction_type'] == 'random']['is_refusal'].mean()
        gap = learned - random

        print(f"\n{model_name}:")
        print(f"  Learned: {learned:.1%}")
        print(f"  Random:  {random:.1%}")
        print(f"  Gap:     {gap:+.1%}")
    except FileNotFoundError:
        print(f"\n{model_name}: Not run yet")
```

---

## For Your MATS Application

### Which Model to Use?

**Option 1: Use all three** (Best)
- Shows effect generalizes across model sizes
- Demonstrates thorough validation
- Strengthens claims

**Option 2: Use Mistral only** (Good)
- Most modern model
- State-of-the-art performance
- Easier to replicate (no auth)

**Option 3: Use Mistral + GPT-2 Medium** (Good)
- Shows scaling (355M → 7B)
- Demonstrates generalization

### Figure Caption Example

> **Figure X: Direction Specificity Across Model Scales.** Refusal rates on harm_test at α=8 for learned direction (green), random baseline (gray), shuffled contrast (blue), and benign contrast (orange). Tested on GPT-2 Medium (355M), Llama-2 7B, and Mistral 7B. Error bars show std dev across 10 random trials. Gap between learned and random demonstrates direction specificity across model sizes.

### Text Example

> We validated direction specificity across three model scales: GPT-2 Medium (355M), Llama-2 7B, and Mistral 7B. At α=8, the learned direction achieves [X]%/[Y]%/[Z]% refusal on harm_test (respectively), compared to random baselines of [A]%/[B]%/[C]%, yielding gaps of [+D]%/[+E]%/[+F]%. All conditions maintain >95% benign helpfulness. The consistent learned >> random gap across model scales demonstrates that projection ablation effects are direction-specific rather than generic compliance suppression.

---

## Output Files

Each model produces:

```
artifacts/baselines_{model}/
├── figures/
│   ├── direction_specificity.png
│   ├── benign_preservation.png
│   └── baseline_pack_table.csv
├── baseline_pack_results.parquet
├── baseline_pack_summary.csv
└── [direction files].pt/json
```

---

## Next Steps

1. **Choose model**: Mistral (easiest), Llama-2 (popular), or both
2. **Run experiment**: 60-90 min on T4, 35-50 min on A100
3. **Analyze results**: Check gap > 30% (or >10% for Mistral)
4. **Add to paper**: Include figures and text
5. **Compare models** (optional): Show generalization

---

**Questions?** Check individual model notebooks for step-by-step instructions.
