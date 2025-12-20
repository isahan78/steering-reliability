# Google Colab Workflow Guide

Complete guide for running steering reliability experiments on Google Colab.

---

## 🚀 Quick Start

### 1. Push Your Code to GitHub

```bash
# In your local repository
git init
git add .
git commit -m "Initial commit: steering reliability project"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/steering-reliability.git
git push -u origin main
```

### 2. Open Colab Notebook

**Option A: Upload the notebook**
1. Go to [Google Colab](https://colab.research.google.com)
2. File → Upload notebook
3. Upload `colab_full_experiment.ipynb`

**Option B: Open from GitHub** (after pushing)
1. Go to Colab
2. File → Open notebook → GitHub tab
3. Enter your repo URL
4. Select `colab_full_experiment.ipynb`

### 3. Enable GPU

1. Runtime → Change runtime type
2. Hardware accelerator → **GPU** (T4)
3. Save

### 4. Run Experiment

- Click Runtime → **Run all**
- Or run cells one by one with Shift+Enter
- Wait ~2-3 hours for completion

### 5. Download Results

The notebook will create a ZIP file at the end. Download it and extract locally:

```bash
# On your local machine
cd steering-reliability
unzip ~/Downloads/steering_reliability_results.zip

# Commit results to Git
git add artifacts/
git commit -m "Experiment results: gpt2-medium full sweep"
git push
```

---

## 📋 Complete Workflow

### Initial Setup (Once)

```bash
# 1. Create GitHub repo (if you haven't)
cd steering-reliability
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/steering-reliability.git
git push -u origin main
```

### Running Experiments (Repeatable)

```bash
# 1. LOCAL: Make changes to config/code
vim configs/default.yaml  # Edit experiment parameters
git add configs/default.yaml
git commit -m "Update config: test layer 20"
git push

# 2. COLAB: Run experiment
#    - Open colab_full_experiment.ipynb in Colab
#    - Runtime → Restart and run all
#    - Wait for completion (~2-3 hours)
#    - Download results ZIP

# 3. LOCAL: Commit results
unzip ~/Downloads/steering_reliability_results.zip
git add artifacts/
git commit -m "Results: layer 20 experiment"
git push
```

---

## 🔧 Configuration Tips

### Running Different Experiments

Edit `configs/default.yaml` before pushing to GitHub:

**Quick test** (30 min):
```yaml
experiment:
  layers: [12]  # Single layer
  alphas: [0, 2, 4]  # Fewer alphas

data:
  max_prompts_per_split: 50  # Smaller dataset
```

**Full experiment** (2-3 hours):
```yaml
experiment:
  layers: [8, 12, 16]  # Multiple layers
  alphas: [0, 1, 2, 4, 8]  # Full alpha sweep

data:
  max_prompts_per_split: null  # All prompts
```

**Different model**:
```yaml
model:
  name: gpt2-large  # Larger model (774M params)
  # or: EleutherAI/pythia-410m
  # or: EleutherAI/pythia-1b
```

---

## 💾 Google Drive Integration (Optional)

To automatically save results to Drive:

### In Colab Notebook:

```python
# 1. Mount Drive (run this cell)
from google.colab import drive
drive.mount('/content/drive')

# 2. After experiment, copy results
!cp -r artifacts/ /content/drive/MyDrive/steering_reliability_results/
```

### On Local Machine:

1. Install Google Drive for Desktop
2. Results appear in `~/Google Drive/steering_reliability_results/`
3. Copy to your Git repo and commit

---

## 🎯 Best Practices

### Experiment Tracking

Keep a log of experiments:

```bash
# experiments_log.md
## Experiment 1: Baseline GPT2-Medium
- Date: 2025-01-20
- Config: layers [8,12,16], alphas [0,1,2,4,8]
- Results: artifacts/runs/full_gpt2_medium/
- Key finding: Layer 12 shows strongest effect

## Experiment 2: Strong Alphas
- Date: 2025-01-21
- Config: alphas [0,5,10,15,20]
- Results: artifacts/runs/strong_alphas/
- Key finding: Alpha > 10 causes degradation
```

### Git Organization

```
steering-reliability/
├── configs/
│   ├── default.yaml          # Current config
│   ├── experiment_001.yaml   # Archived configs
│   └── experiment_002.yaml
├── artifacts/
│   └── runs/
│       ├── full_gpt2_medium/     # Experiment 1
│       ├── strong_alphas/        # Experiment 2
│       └── layer_comparison/     # Experiment 3
└── experiments_log.md        # Track all runs
```

### Commit Messages

Use descriptive commit messages:

```bash
# Good
git commit -m "Experiment: Layer comparison [8,12,16,20] on gpt2-medium"

# Bad
git commit -m "results"
```

---

## 🐛 Troubleshooting

### Out of Memory on Colab

If you get OOM errors:

```yaml
# Reduce batch size in configs/default.yaml
generation:
  batch_size: 2  # Instead of 4
```

### Git Clone Authentication

For private repos:

```bash
# In Colab cell:
!git clone https://YOUR_GITHUB_TOKEN@github.com/YOUR_USERNAME/steering-reliability.git
```

Get token: GitHub → Settings → Developer settings → Personal access tokens

### Download Fails

If ZIP download fails (large file):

Use Google Drive method instead:
1. Mount Drive in Colab
2. Copy artifacts to Drive
3. Access from Drive on local machine

---

## ⏱️ Runtime Estimates

On Colab T4 GPU:

| Configuration | Completions | Time |
|---------------|-------------|------|
| Quick test (50 prompts/split) | ~1,800 | 30 min |
| Medium (100 prompts/split) | ~6,000 | 1 hour |
| Full (150/150/200 prompts) | ~12,500 | 2-3 hours |

On Colab A100 GPU (Colab Pro):
- ~50% faster than T4
- Full experiment: 1-1.5 hours

---

## 📊 Monitoring Progress

While Colab is running, you can monitor:

```python
# In a new cell, run periodically:
!ls -lh artifacts/runs/full_gpt2_medium/

# Check how many results files generated:
!find artifacts/ -name "*.parquet" -exec ls -lh {} \;
```

Or watch the output cell for progress bars.

---

## 🔄 Iterative Experimentation

Typical research cycle:

```
1. LOCAL: Edit config → commit → push
2. COLAB: Pull changes → run experiment → download
3. LOCAL: Unzip → analyze → commit results → push
4. Repeat with new config
```

Example:

```bash
# Iteration 1: Test layers
vim configs/default.yaml  # Set layers: [8,12,16]
git commit -am "Config: test 3 layers"
git push
# → Run in Colab → Download results

# Iteration 2: Focus on best layer
vim configs/default.yaml  # Set layers: [12], more alphas
git commit -am "Config: focus layer 12, alphas [0,2,4,8,16]"
git push
# → Run in Colab → Download results

# Iteration 3: Try larger model
vim configs/default.yaml  # Set model: gpt2-large
git commit -am "Config: scale to gpt2-large"
git push
# → Run in Colab → Download results
```

---

## ✨ Advanced: Colab Pro

If you need faster runs, consider [Colab Pro](https://colab.research.google.com/signup) ($10/month):

- **A100 GPU**: ~2x faster than T4
- **Longer runtime**: Up to 24 hours vs 12 hours
- **More memory**: 40GB RAM vs 12GB

---

## 📝 Summary

**Your workflow:**
1. ✅ Code lives in GitHub (version controlled)
2. ✅ Experiments run on Colab (free GPU)
3. ✅ Results download to local (commit to Git)
4. ✅ Iterate: change config → push → run on Colab → download → commit

**Benefits:**
- 🚀 Fast GPU experimentation
- 💻 Laptop stays free
- 🔄 Full Git version control
- 💰 Free compute (or cheap with Colab Pro)

Ready to run your first Colab experiment!
