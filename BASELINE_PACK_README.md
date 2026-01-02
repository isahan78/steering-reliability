# Baseline Pack: Testing Direction Specificity

This module implements minimal baselines to validate that the strong projection ablation effects are direction-specific, not artifacts of ablating any random direction.

## Overview

The baseline pack tests 3 critical hypotheses:

1. **Random Direction Baseline**: Is ablation effective because the learned direction is meaningful, or because removing any projection disrupts harmful compliance?

2. **Shuffled Contrast Baseline**: Does the contrastive pairing (refusal vs compliance on the same prompt) matter, or would any averaged difference work?

3. **Benign Contrast Baseline** (optional): Is the effect specific to harmful compliance/refusal geometry, or would any contrast direction produce refusal?

## Quick Start

### 1. Run Baseline Pack

```bash
python scripts/run_baseline_pack.py \
  --layer 10 \
  --alphas 0 1 4 8 \
  --n_harm_test 50 \
  --n_benign 50 \
  --n_random 10 \
  --seed 0 \
  --include_benign_contrast \
  --output_dir artifacts/baselines
```

**Runtime:** ~20-30 minutes on A100 GPU

This will:
- Load GPT-2 Small
- Build/load learned direction (from harm_train)
- Build shuffled contrast direction
- Generate 10 random directions
- (Optional) Build benign contrast direction
- Test all directions on 50 harm_test + 50 benign prompts
- Save results to `artifacts/baselines/`

### 2. Generate Figures

```bash
python -m steering_reliability.analysis.plot_baseline_pack \
  --in_parquet artifacts/baselines/baseline_pack_results.parquet \
  --out_dir artifacts/baselines/figures
```

This generates:
- `direction_specificity.png` - Refusal rates on harm_test by direction type
- `benign_preservation.png` - Helpfulness on benign by direction type
- `baseline_pack_table.csv` - Summary table with all metrics

## Expected Results

**If the learned direction is truly capturing something meaningful:**

1. **Learned direction >> Random directions** on harm_test refusal
   - Learned: ~95-98% refusal at α=8
   - Random (mean): <50% refusal at α=8

2. **Learned direction > Shuffled direction**
   - Shuffled should be weaker, showing that per-prompt alignment matters

3. **Benign contrast ≠ Learned** on harm_test
   - Benign contrast should not produce high refusal on harmful queries

4. **Benign helpfulness preserved**
   - All directions maintain >95% helpfulness on benign queries

## Interpretation

### Success Criteria

✅ **Pass:** Learned ablation achieves substantially higher harm_test refusal than all baselines, while maintaining benign helpfulness

❌ **Fail:** Random ablation matches learned performance → Effect is not direction-specific

### What This Tests

**Random baseline:**
- Tests whether ablation is a "generic compliance suppressor" (bad)
- Or whether it works via specific learned features (good)

**Shuffled baseline:**
- Tests whether contrastive pairing matters
- If shuffled ≈ learned, the "contrast" story is undermined

**Benign baseline:**
- Tests domain specificity
- Benign contrast should capture style/structure differences, not harmful content

## Implementation Details

### Direction Types

1. **Learned** (`direction_type='learned'`)
   ```python
   v = mean(resid_refusal(p_i) - resid_compliance(p_i))
   ```
   - Aligned contrastive pairing
   - Built on harm_train prompts

2. **Shuffled** (`direction_type='shuffled'`)
   ```python
   v = mean(resid_refusal(p_i) - resid_compliance(p_perm(i)))
   ```
   - Misaligned prompts (fixed permutation)
   - Same suffixes, breaks per-prompt alignment

3. **Random** (`direction_type='random'`, `random_trial=0..9`)
   ```python
   v ~ N(0, I), normalized
   ```
   - Sampled from standard normal
   - 10 independent trials for robustness

4. **Benign Contrast** (`direction_type='benign_contrast'`, optional)
   ```python
   v = mean(resid_A(benign_i) - resid_B(benign_i))
   ```
   - A: " Here's a helpful answer:"
   - B: " Let me think step by step:"
   - Tests whether any contrast → refusal

### Output Schema

Results saved to `baseline_pack_results.parquet`:

```python
{
    'direction_type': str,  # learned, shuffled, random, benign_contrast
    'random_trial': int | None,  # 0-9 for random, None otherwise
    'layer': int,  # 10
    'alpha': float,  # 0, 1, 4, 8
    'intervention': str,  # 'ablate'
    'split': str,  # 'harm_test' or 'benign'
    'prompt': str,
    'completion': str,
    'refusal_score': float,
    'is_refusal': bool,
    'helpfulness_score': float | None,
    'is_helpful': bool | None,
    'char_length': int,
    'timestamp': str,
    'seed': int,
    'temperature': float,
    'top_p': float,
    'max_new_tokens': int
}
```

## Usage in Colab

```python
# Clone repo and install dependencies (as usual)
!git clone https://github.com/isahan78/steering-reliability.git
%cd steering-reliability

# Run baseline pack (faster with smaller dataset)
!python scripts/run_baseline_pack.py \
  --layer 10 \
  --alphas 1 4 8 \
  --n_harm_test 30 \
  --n_benign 30 \
  --n_random 5 \
  --seed 0

# Generate figures
!python -m steering_reliability.analysis.plot_baseline_pack \
  --in_parquet artifacts/baselines/baseline_pack_results.parquet

# Download results
!zip -r baseline_pack.zip artifacts/baselines/
from google.colab import files
files.download('baseline_pack.zip')
```

## Adding to MATS Application

### Text to Include (4-6 sentences)

> To test whether the ablation effect is direction-specific, we implemented three baselines: (1) **random direction ablation** (n=10 trials, sampled from N(0,I)) tests whether removing any projection disrupts compliance, (2) **shuffled contrast** tests whether per-prompt alignment matters by breaking the refusal/compliance pairing, and (3) **benign contrast** (optional) tests whether contrast directions generically induce refusal. The learned direction achieves [XX]% refusal on harm_test at α=8, compared to random mean of [YY]±[ZZ]% and shuffled [WW]%, while maintaining [PP]% benign helpfulness across all conditions. This substantial gap rules out the hypothesis that ablation is a generic "compliance suppressor" and supports the claim that the learned direction captures a specific harmful-compliance feature. The benign contrast baseline shows [QQ]% refusal, confirming the effect is domain-specific rather than a property of any contrastive direction.

### Figures to Add

1. **Figure: Direction Specificity**
   - Barplot showing refusal rates at α={1,4,8}
   - Learned (green) >> Random (gray, with error bars) >> Shuffled (blue)
   - Insert in "Methods" or "Results" section

2. **Table: Baseline Summary**
   - Rows: direction types
   - Columns: harm_refusal_α1, harm_refusal_α4, harm_refusal_α8, benign_helpful_α8
   - Insert in "Results" section

## Files Created

```
steering-reliability/
├── src/steering_reliability/
│   ├── directions/
│   │   ├── random_direction.py       # Random direction generator
│   │   ├── benign_contrast.py        # Benign contrast builder
│   │   └── build_direction.py        # Updated with shuffled mode
│   └── analysis/
│       └── plot_baseline_pack.py     # Plotting & table generation
├── scripts/
│   └── run_baseline_pack.py          # Main experiment runner
└── artifacts/baselines/               # Results directory
    ├── learned_layer10.pt            # Learned direction
    ├── shuffled_layer10.pt           # Shuffled direction
    ├── benign_contrast_layer10.pt    # Benign direction (optional)
    ├── baseline_pack_results.parquet # Full results
    ├── baseline_pack_summary.csv     # Aggregated summary
    └── figures/
        ├── direction_specificity.png
        ├── benign_preservation.png
        └── baseline_pack_table.csv
```

## Troubleshooting

### "No module named 'steering_reliability'"
```bash
# Make sure you're in the repo root
cd steering-reliability
# Add src to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### "CUDA out of memory"
Reduce batch size or number of prompts:
```bash
python scripts/run_baseline_pack.py --n_harm_test 30 --n_benign 30
```

### Results show random ≈ learned
This would be a **negative result** indicating:
- The ablation effect is not direction-specific
- The current mechanistic story needs revision
- Consider investigating whether ablation acts as generic compliance suppressor in this model

## Next Steps

1. **Run baseline pack**: `python scripts/run_baseline_pack.py`
2. **Generate figures**: `python -m steering_reliability.analysis.plot_baseline_pack ...`
3. **Analyze results**: Check if learned >> baselines
4. **Update application**: Add baseline results to Methods/Results sections
5. **Include figures**: Insert direction_specificity.png and table
6. **Interpret**: Write 4-6 sentences explaining what baselines show

---

**Questions?** Check the code comments in `scripts/run_baseline_pack.py` and analysis modules.
