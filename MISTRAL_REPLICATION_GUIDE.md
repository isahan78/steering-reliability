# Mistral 7B: Complete Paper Replication Guide

This guide walks you through replicating ALL experiments from your GPT-2 Small paper using Mistral 7B.

## 📊 What Gets Replicated

### From Your Paper → To Mistral

| Experiment | GPT-2 Small Setup | Mistral 7B Setup |
|------------|-------------------|------------------|
| **Layer Sweep** | Layers 4,6,8,10 (33%-83%) | Layers 8,16,22,27 (25%-84%) |
| | Additive, α={0,2,4} | Additive, α={0,2,4} |
| | 50 prompts/split | 100 prompts/split |
| **Intervention** | Layer 10 (best) | Layer 27 (best) |
| | Add vs Ablate | Add vs Ablate |
| | α={1,2,4,8} | α={1,2,4,8} |
| | 100 prompts/split | 100 prompts/split |
| **Dist. Shift** | harm_train → harm_test | harm_train → harm_test |
| | Ablation, α=8 | Ablation, α=8 |

---

## 🚀 Quick Start (3 Hours Total)

### **Option 1: Full Automated Colab** ⭐ **Recommended**

1. Open Colab: https://colab.research.google.com/
2. Upload: `colab_mistral_full_replication.ipynb`
3. Runtime → Change runtime → T4 GPU (or A100 for 2x speed)
4. Run all cells
5. Wait 3-4 hours (T4) or 2-2.5 hours (A100)
6. Download `mistral_full_results.zip`

**That's it!** All experiments run automatically.

### **Option 2: Manual Step-by-Step**

```bash
cd steering-reliability
git pull

# Experiment 1: Layer Sweep (~90 min)
python scripts/run_all.py \
  --config configs/mistral_layer_sweep.yaml \
  --skip-baseline

# Experiment 2: Intervention Comparison (~90 min)
python scripts/run_all.py \
  --config configs/mistral_intervention_comparison.yaml \
  --skip-baseline
```

---

## 📋 Expected Results

### Experiment 1: Layer Sensitivity

**Your GPT-2 Results:**
- Layer 10 (83% depth): 84% refusal at α=4
- Late layers perform best

**Expected Mistral Results:**
- Layer 27 (84% depth): 90-95% refusal at α=4
- Similar pattern: late > mid > early
- **Why higher?** Mistral has stronger safety training

**Graph:** `layer_comparison.png` (4 lines, one per layer)

### Experiment 2: Intervention Comparison

**Your GPT-2 Results (α=8):**
- Additive: 83% refusal, 100% helpfulness
- Ablation: 98% refusal, 100% helpfulness
- Gap: +15%

**Expected Mistral Results (α=8):**
- Additive: 85-90% refusal
- Ablation: 95-98% refusal
- Gap: +10-13%
- **Why smaller gap?** Mistral baseline is already safer

**Graphs:**
- `intervention_comparison.png` (2 lines: additive vs ablation)
- `tradeoff_curve.png` (scatter: safety vs helpfulness)

**Table:** `intervention_comparison_table.csv`

| Alpha | Additive Refusal | Ablation Refusal | Ablation Helpfulness |
|-------|------------------|------------------|----------------------|
| 1 | ~XX% | ~XX% | ~XX% |
| 2 | ~XX% | ~XX% | ~XX% |
| 4 | ~XX% | ~XX% | ~XX% |
| 8 | ~XX% | ~XX% | ~XX% |

### Experiment 3: Distribution Shift

**Your GPT-2 Results:**
- harm_train: 98% refusal
- harm_test: 98% refusal
- Gap: <2%

**Expected Mistral Results:**
- harm_train: 95-98% refusal
- harm_test: 95-98% refusal
- Gap: <5%

---

## 🎯 Interpreting Results

### Success Criteria

✅ **Strong Replication:**
- Late layers (27) >> early layers (8)
- Ablation >> Additive by >10%
- Generalization gap <5%
- Benign helpfulness >95%

⚠️ **Acceptable Variation:**
- Smaller gaps OK (Mistral is safer baseline)
- Absolute numbers differ (Mistral 7B vs GPT-2 Small)
- Pattern consistency matters more than exact numbers

❌ **Concerning:**
- Random performance across layers
- Additive = Ablation
- Large generalization gap (>15%)
- Benign helpfulness <90%

### Why Results Might Differ

1. **Model Size:** 7B vs 124M parameters
2. **Architecture:** Mistral optimizations vs vanilla GPT-2
3. **Training:** Mistral has stronger RLHF/safety tuning
4. **Prompt Distribution:** Your harmful prompts may not match Mistral's training

**Key insight:** The *mechanistic pattern* should replicate (ablation > additive, late layers best), even if absolute numbers differ.

---

## 📊 Updating Your Paper

### Adding Mistral Results

**Option 1: Replace GPT-2 with Mistral**
- Update all numbers to Mistral
- Add note: "We use Mistral 7B, a modern 7B parameter model, for computational efficiency and stronger baseline performance."

**Option 2: Show Both (Recommended)**
- Keep GPT-2 as main results
- Add Mistral as "replication on modern architecture"
- Strengthens claims about generalization

### Text Template

> **Replication on Mistral 7B.** To validate our findings generalize beyond GPT-2, we replicated all experiments on Mistral 7B (7B parameters, 32 layers). At layer 27 (equivalent depth to layer 10 in GPT-2), ablation achieves [X]% refusal vs [Y]% for additive steering (Δ=[Z]%), maintaining [W]% benign helpfulness. The pattern replicates: late-layer control (layer 27 > layer 8), ablation superiority (gap=[Z]%), and minimal distribution shift (harm_train → harm_test gap <[G]%). These results demonstrate the mechanistic insights generalize from 124M to 7B scale models.

### Figure Captions

**Figure 1 (Layer Sweep):**
> Layer sensitivity analysis on Mistral 7B. Refusal rates on harm_test at steering strengths α={0,2,4} across layers 8, 16, 22, 27 (early to late). Late layer (27) achieves highest refusal, consistent with GPT-2 findings.

**Figure 2 (Intervention Comparison):**
> Additive vs ablation steering on Mistral 7B (layer 27). Ablation (green) achieves [X]% refusal at α=8 compared to [Y]% for additive (blue), demonstrating mechanistic asymmetry replicates at 7B scale.

**Figure 3 (Tradeoff Curve):**
> Safety-helpfulness tradeoff on Mistral 7B. Each point represents (refusal_rate, helpfulness) at different α. Ablation (squares) reaches top-right corner (high safety, high helpfulness) while additive (circles) shows worse tradeoff.

---

## 🔧 Troubleshooting

### Out of Memory

**Symptoms:** CUDA OOM error during model loading or generation

**Solutions:**
1. Reduce batch size: Edit config, change `batch_size: 2` → `batch_size: 1`
2. Use A100: Colab Pro gives 40GB VRAM vs 16GB on T4
3. Reduce prompts: Change `max_prompts_per_split: 100` → `50`
4. Use bfloat16: Change `dtype: float32` → `dtype: bfloat16` (if supported)

### Results Look Wrong

**Check 1: Completions quality**
```python
df = pd.read_parquet('artifacts/runs/mistral_layer_sweep/all_results.parquet')
sample = df[(df['alpha'] == 0) & (df['split'] == 'benign')].iloc[0]
print(f"Prompt: {sample['prompt']}")
print(f"Completion: {sample['completion']}")
```

Completions should:
- ✅ Be coherent and on-topic
- ✅ NOT start with `<|endoftext|>` or prompt repetition
- ✅ Be 50-300 characters

**Check 2: Baseline (α=0)**
- Should have ~50-70% refusal on harmful queries (Mistral's safety training)
- Should have ~95-100% helpfulness on benign

**Check 3: Direction norms**
Look in metadata files - norms should be ~0.5-2.0 after normalization.

### Experiments Take Too Long

**Quick test** (30 min instead of 3 hours):
```yaml
# In config files, change:
max_prompts_per_split: 30  # Instead of 100
layers: [16, 27]  # Just 2 layers instead of 4
alphas: [0, 4, 8]  # Skip intermediate alphas
```

This gives you enough data to see if patterns replicate.

---

## 📁 Output Files

After running, you'll have:

```
artifacts/runs/
├── mistral_layer_sweep/
│   ├── all_results.parquet          # Full data (all layers, splits, alphas)
│   ├── figures/
│   │   └── layer_comparison.png     # Main figure for paper
│   └── directions/
│       ├── layer_8/v.pt
│       ├── layer_16/v.pt
│       ├── layer_22/v.pt
│       └── layer_27/v.pt
│
└── mistral_intervention_comparison/
    ├── all_results.parquet          # Full data (both interventions)
    ├── intervention_comparison.png  # Additive vs ablation plot
    ├── tradeoff_curve.png           # Safety-helpfulness scatter
    ├── intervention_comparison_table.csv  # Paper table
    └── directions/
        └── layer_27/v.pt
```

---

## ⏱️ Runtime Breakdown

### T4 GPU (Free Colab)
- Layer sweep: ~90 minutes
  - 4 layers × 3 alphas × 3 splits = 36 conditions
  - ~2.5 min per condition
- Intervention comparison: ~90 minutes
  - 2 interventions × 5 alphas × 3 splits = 30 conditions
  - ~3 min per condition
- **Total: ~3-3.5 hours**

### A100 GPU (Colab Pro)
- Layer sweep: ~50 minutes
- Intervention comparison: ~50 minutes
- **Total: ~2-2.5 hours**

**Speedup tips:**
- Use A100 (2x faster)
- Reduce prompts to 50 per split (1.5x faster)
- Skip α=2 (1.3x faster)

---

## 🎓 For Your MATS Application

### Key Points to Emphasize

1. **Generalization Across Scales**
   - "Findings replicate from GPT-2 Small (124M) to Mistral 7B (7B), spanning 56× model size increase"

2. **Mechanistic Consistency**
   - "Late-layer behavioral control and ablation superiority hold across architectures"

3. **Modern Baseline**
   - "Mistral 7B represents state-of-the-art open models, demonstrating practical relevance"

4. **Stronger Test**
   - "Mistral's stronger safety training makes it a more challenging testbed for steering reliability"

### What to Include

**Must have:**
- Intervention comparison table (α=8 row)
- One figure (intervention_comparison.png or tradeoff_curve.png)
- 2-3 sentences in results section

**Nice to have:**
- Layer sweep figure
- Full comparison table (all alphas)
- Side-by-side GPT-2 vs Mistral results

**Not necessary:**
- Raw data files
- All intermediate alphas
- Sample completions (unless specifically interesting)

---

## 🎯 Next Steps

1. **Run experiments** (3-4 hours on T4)
2. **Check results quality** (completions, baselines, patterns)
3. **Generate figures** (automatic in Colab notebook)
4. **Update paper** (add Mistral section, update claims)
5. **Commit results** (optional: add figures to repo)

---

**Questions?** Check the Colab notebook comments or `LLAMA_MISTRAL_GUIDE.md` for more details.

**Ready to run!** Upload `colab_mistral_full_replication.ipynb` to Colab and hit "Run all" 🚀
