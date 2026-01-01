# MATS Application Submission Checklist

Your application is almost complete! Follow these steps to finalize and submit.

---

## ✅ What's Done

- [x] Complete experimental pipeline
- [x] Level 2 & 3 experiments run successfully
- [x] Strong results (98% refusal, 0% side effects)
- [x] Executive summary written
- [x] Full application document created
- [x] Code repository on GitHub
- [x] Reproducible Colab notebook

---

## 📝 Steps to Submit

### 1. Generate Missing Graphs in Colab

You need 3 key graphs for the executive summary. Run this in Colab:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

summary = pd.read_csv('artifacts/tables/summary.csv')

# GRAPH 1: Layer Comparison
plt.figure(figsize=(10, 6))
harm_test = summary[summary['split'] == 'harm_test']
for layer in [4, 6, 8, 10]:
    data = harm_test[harm_test['layer'] == layer].sort_values('alpha')
    plt.plot(data['alpha'], data['is_refusal_mean'],
             marker='o', linewidth=2, markersize=8, label=f'Layer {int(layer)}')

plt.xlabel('Alpha (Steering Strength)', fontsize=12)
plt.ylabel('Refusal Rate (harm_test)', fontsize=12)
plt.title('Layer Comparison: Refusal Rate vs Steering Strength', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('layer_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# GRAPH 2: Intervention Comparison
plt.figure(figsize=(10, 6))
layer10 = harm_test[harm_test['layer'] == 10]

for intervention in ['add', 'ablate']:
    data = layer10[layer10['intervention_type'] == intervention].sort_values('alpha')
    plt.plot(data['alpha'], data['is_refusal_mean'],
             marker='o', linewidth=2, markersize=8,
             label=f'{intervention.capitalize()}')

plt.xlabel('Alpha (Steering Strength)', fontsize=12)
plt.ylabel('Refusal Rate (harm_test)', fontsize=12)
plt.title('Intervention Comparison: Additive vs Ablation (Layer 10)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('intervention_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# GRAPH 3: Tradeoff Curve
benign = summary[summary['split'] == 'benign']
harm = summary[summary['split'] == 'harm_test']

merged = harm.merge(benign, on=['layer', 'alpha', 'intervention_type'],
                    suffixes=('_harm', '_benign'))
merged = merged[merged['layer'] == 10]

plt.figure(figsize=(10, 6))
for intervention in ['add', 'ablate']:
    data = merged[merged['intervention_type'] == intervention]
    plt.scatter(1 - data['is_helpful_mean_benign'],
                data['is_refusal_mean_harm'],
                s=150, alpha=0.7, label=intervention.capitalize())

    for _, row in data.iterrows():
        plt.annotate(f"α={row['alpha']}",
                    (1 - row['is_helpful_mean_benign'], row['is_refusal_mean_harm']),
                    fontsize=9, xytext=(5,5), textcoords='offset points')

plt.xlabel('Helpfulness Drop on Benign Queries', fontsize=12)
plt.ylabel('Refusal Rate on Harmful Queries', fontsize=12)
plt.title('Safety-Helpfulness Tradeoff: Additive vs Ablation', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tradeoff_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ All 3 graphs saved!")

# Download them
from google.colab import files
files.download('layer_comparison.png')
files.download('intervention_comparison.png')
files.download('tradeoff_curve.png')
```

**Download these 3 PNG files to your computer.**

---

### 2. Create Google Doc

#### Option A: Import Markdown (Easiest)

1. Go to https://docs.google.com
2. File → New → Blank document
3. Open `MATS_APPLICATION.md` in a text editor
4. Copy ALL content
5. Paste into Google Doc
6. Format will need cleanup (see below)

#### Option B: Use Pandoc (If Installed)

```bash
cd /Users/IsahanKhan/steering-reliability
pandoc MATS_APPLICATION.md -o MATS_APPLICATION.docx
```

Then upload DOCX to Google Drive and open in Google Docs.

---

### 3. Format the Google Doc

After importing:

1. **Title:** Make "Steering Reliability Under Distribution Shift" a large heading

2. **Section headings:**
   - "EXECUTIVE SUMMARY" → Heading 1
   - "What Problem Am I Trying to Solve?" → Heading 2
   - "Key Experiments" → Heading 2
   - Etc.

3. **Insert graphs:**
   - Find `[Insert layer_comparison.png]` → Delete this text
   - Insert → Image → Upload your PNG
   - Repeat for all 3 graphs in Executive Summary
   - Also insert into Full Results section

4. **Format tables:**
   - Select table text
   - Insert → Table → Convert text to table
   - Or just ensure tables are readable

5. **Format code blocks:**
   - Use monospace font (Courier New)
   - Light gray background
   - Or just ensure code is readable

6. **Page breaks:**
   - Add page break after Executive Summary
   - Add before each major section

---

### 4. Final Checklist Before Submission

- [ ] All 3 graphs inserted in Executive Summary
- [ ] Tables are formatted and readable
- [ ] Code snippets are monospace/readable
- [ ] No placeholder text (like `[Your Name]`, `[Your Email]`)
- [ ] Sharing set to "Anyone with the link can view"
- [ ] Word count ~4000-6000 (including appendix)
- [ ] Executive summary is 500-600 words
- [ ] All section headings are formatted

---

### 5. Fill in Placeholders

Find and replace these in the Google Doc:

- `[Your Name]` → Your actual name
- `[Your Email]` → Your actual email
- Double-check all numbers match your actual results
- Verify GitHub link works: https://github.com/isahan78/steering-reliability

---

### 6. Get Sharing Link

1. In Google Doc, click **Share** (top right)
2. Change to "Anyone with the link can view"
3. Click **Copy link**
4. This is your submission link!

---

### 7. Optional: Add Additional Graphs

If you want to include all the graphs from your experiments:

In Colab, find your generated graphs:
```bash
!ls artifacts/figures/
```

Should show:
- `generalization_gap.png`
- `heatmap_refusal_harm_test.png`
- `heatmap_helpfulness_benign.png`

Download and insert these in the "Full Results" section.

---

## 📊 What Your Final Doc Should Look Like

```
Page 1: Title + Executive Summary start
Page 2-3: Executive Summary (with 3 graphs)
Page 4-7: Detailed Methods
Page 8-12: Full Results (with additional graphs/tables)
Page 13-15: Discussion
Page 16-18: Appendices

Total: ~15-20 pages with graphs
```

---

## ⏱️ Time Tracking

**For the application form, state:**

**Total time:** 20 hours

**Breakdown:**
- Environment setup & debugging: 3h
- Prompt dataset creation: 1h
- Initial experiments: 1h
- Layer comparison: 3h
- Intervention comparison: 4h
- Analysis: 3h
- Documentation: 2h
- Figures: 1h
- Executive summary: 2h

**Not counted:** Paper reading, GPU setup, waiting for experiments

---

## 🚀 Ready to Submit!

Once you've completed steps 1-6:

1. Copy the Google Docs link
2. Fill out MATS application form
3. Paste the link in the "Research project write-up" field
4. Double-check sharing permissions
5. Submit!

---

## 📧 Questions?

If MATS reviewers have questions, they can:
- Read the Google Doc (main submission)
- Check the GitHub repo (code + results)
- Run the Colab notebook (full reproduction)

Everything is self-contained and reproducible!

---

**Good luck with your application! 🎉**

Your results are strong (98% refusal, 0% side effects, clean mechanistic insight). You've built a solid research pipeline in 20 hours.
