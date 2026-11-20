# Reverse-Engineering the IOI Circuit in GPT-2 Small

**Activation patching, path patching and a transfer study of the Indirect Object Identification circuit**

![Model](https://img.shields.io/badge/model-GPT--2%20small-informational)
![Status](https://img.shields.io/badge/status-research%20code-orange)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bhargavrrao-web/IOI_Circuit_GPT2/blob/main/IOI_Circuit_GPT2_Colab.ipynb)

Authors: **Bhargav R Rao**, **Pushpak V Reddy**, **Karthikeyan Periyasami** · RV University, Bangalore

---

## Overview

GPT-2 small completes *"When Mary and John went to the store, John gave a drink to ___"* with **Mary**, but nobody programmed that rule. This project uses **causal interventions** (not attention maps) to find which of the model's 144 attention heads do the work, and tests how far that circuit carries over to similar tasks.

1. **Reproduce** the 26-head IOI circuit of Wang et al. (ICLR 2023) with an independent, unit-tested implementation of activation patching and path patching.
2. **Validate** it with sufficiency, completeness and minimality tests (mean-ablation).
3. **Transfer**: test it on seven new entity-tracking task families (animals, objects, colours, professions, fruits, vehicles, countries).
4. **Explore** what predicts transfer, and whether a head ranking learned on IOI guides head pruning on other tasks.

> **Positioning.** This is a reproduction plus a statistical extension. Earlier work already showed that IOI circuit components are reused on another task (Merullo et al., ICLR 2024) and that such circuits are stable across training and scale (Tigges et al., NeurIPS 2024). This project does not claim to be the first to show circuit reuse.

## Key results

GPT-2 small, 100 prompts per task, random seed 0, free Google Colab GPU.

| Question | Result |
|---|---|
| Are key head classes recovered? | IOI accuracy 100% and mean logit difference 3.64 (paper: 99.3%, 3.56). Negative Name Mover and S-Inhibition heads recovered exactly; 13 of 15 heads found are in the published circuit. |
| Is the published circuit sufficient on IOI? | Alone it keeps **0.84** (95% CI 0.76–0.93) of the full-model performance; random 26-head sets give negative fractions. |
| Does it transfer to new tasks? | It keeps **0.28–0.63** on six new families. Name Mover heads attend more strongly to the target entity than to the repeated one, and S-Inhibition heads are re-discovered on every family. |
| Is it complete? | **No.** Greedy removal gives an incompleteness of up to 110.9% of the logit difference (original paper: about 87%). |
| What predicts transfer? | Neither semantic distance to names nor task difficulty (inconclusive, n = 7). |
| Does an IOI ranking help pruning? | 80% of the full-model mean logit difference is reached with 20 heads on IOI and 26–60 on the transfer families (smallest tested budget). |

Vehicles is excluded from conclusions because GPT-2 small is near chance on it (56% accuracy); Fruits (67%) is weak.

## Quick start

### Google Colab (recommended)

1. Open the notebook with the badge above.
2. **Runtime → Change runtime type → GPU.**
3. **Runtime → Run all.** Allow the Google Drive pop-up when asked.

Progress is checkpointed to Google Drive (`ioi_project/`). If Colab disconnects, run all again: finished sweeps are loaded (`[resumed]`) instead of recomputed. Expect roughly 10–20 minutes on a GPU (estimate; it depends on the GPU and on `N_PROMPTS`).

### Local machine

```bash
pip install -r requirements.txt
jupyter notebook IOI_Circuit_GPT2_Colab.ipynb
```

A GPU is not required (GPT-2 small has 124M parameters) but the patching sweeps run hundreds of forward passes, so on CPU expect an hour or more; lower `N_PROMPTS` in the setup cell to speed it up. TransformerLens is pinned to 2.16.1 because the notebook was written against that API.

## Repository layout

```text
.
├── IOI_Circuit_GPT2_Colab.ipynb          # clean notebook (no outputs): all experiments, tables and figures
├── IOI_Circuit_GPT2_Colab_v4_executed.ipynb  # the same analysis, saved with the outputs of the run behind the reported numbers
├── results/                              # CSV tables and figures written by the run
├── data/
│   ├── prompts_all_tasks.csv             # the exact 100 clean/corrupted prompts per task used in the run
│   ├── word_lists_used.csv               # the words that were kept (single GPT-2 tokens)
│   ├── export_prompts_cell.py            # Colab cell that regenerates the two files above
│   ├── IOI_prompt_dataset.xlsx           # templates, word lists and example prompts
│   ├── Wang_IOI_original_dataset.xlsx    # original IOI lists/templates + samples (see "Data")
│   ├── wang_p_IOI.csv, wang_p_ABC.csv
│   └── generate_wang_ioi.py              # regenerates the original dataset with the authors' code
├── LICENSE
├── requirements.txt
└── README.md
```

**Which notebook to use.** `IOI_Circuit_GPT2_Colab.ipynb` is the clean notebook to run (the Colab badge opens it). The numbers in the manuscript come from one execution on Google Colab (free GPU, 100 prompts per task, seed 0), saved with its outputs as `IOI_Circuit_GPT2_Colab_v4_executed.ipynb`. The two notebooks run the same analysis; they differ only in a corrected printed verdict label in the final "evidence summary" cell (in the executed notebook the label, for example "WEAK transfer (1/4 criteria)", is wrong because boolean values were added as logical OR). No table, figure or reported number is affected.

## What the notebook does

| Part | Content |
|---|---|
| 1 | Load GPT-2 small, build the IOI prompt dataset, check that the model solves the task |
| 2 | Activation patching over all 144 heads (denoising and noising) |
| 3 | Path patching: direct effect on the logits, then trace back to Name Movers → S-Inhibition → earlier heads |
| 4 | Assemble the circuit by head role and compare with Wang et al. |
| 5 | Transfer test: the same analysis on seven new task families |
| 6 | Circuit validation: sufficiency, completeness, minimality (also applied to the new tasks) |
| 7 | Head behaviour: do Name Movers still attend to the correct entity? |
| 8 | Re-discovery of the circuit on each new task, bootstrap intervals, permutation tests |
| 9 | Evidence summary and exploratory analysis of what predicts transfer |
| 10 | Circuit-guided head-pruning curves |

## Methods in brief

- **Metric.** Logit difference = logit(correct name) − logit(repeated name) at the last position. Above 0 means the model prefers the correct name.
- **Clean and corrupted prompts.** The corrupted ("ABC") version replaces the three entities with three different, unrelated entities, so there is no clue. Entities are single tokens, so both prompts have equal length.
- **Activation patching.** One head's output is copied between a clean and a corrupted run, one head at a time, in two directions (denoising: clean into corrupted; noising: corrupted into clean).
- **Path patching.** Isolates one connection (sender head → receiver): all other heads are frozen to their clean values, MLPs and layer norms are recomputed, and the receiver's input is patched into a clean run.
- **Sufficiency.** Keep only a head set and replace every other head by its mean activation on the corrupted prompts. The retained fraction is F(circuit) / F(full model). Compared with 20 random sets of 26 heads.
- **Statistics.** Bootstrap 95% confidence intervals over prompts (2000 resamples) and permutation tests for rank correlations.

## Data

There is **no training data**: GPT-2 small is frozen. The prompts are probe sentences with a known answer, generated at run time from templates and word lists with a fixed seed.

- **IOI.** The 15 templates (in both entity orders) come from Wang et al., Appendix E. The names, places and objects are **my own word lists, not the original ones** (only 13 of my 40 names appear in the original 99-name list; see the `Comparison` sheet in `Wang_IOI_original_dataset.xlsx`). The IOI prompts are therefore "Wang's templates with my words".
- **Seven new families.** Templates and word lists were written by the author; there is no external benchmark behind them. Professions, Fruits and Vehicles reuse the Animals templates, and Countries reuses the IOI templates.
- **Exact prompts.** Run `data/export_prompts_cell.py` as a cell in the notebook to export the prompts used in the experiments (`prompts_all_tasks.csv`).
- **Original dataset.** Wang et al. publish code, not a data file. `generate_wang_ioi.py` regenerates their dataset by calling the functions of their repository: <https://github.com/redwoodresearch/Easy-Transformer> (MIT licence).
- **Known weaknesses.** Random filler words sometimes give odd sentences (for example "a apple"); there are only 3 hand-written templates per new family, and the Objects template contains the word "table", which is also in the Objects word list (2 of the 100 prompts are affected; see the paper).

## Validation of the implementation

Four unit tests run in the notebook and pass:

1. Path patching to the logits equals noising activation patching for last-layer heads (two independent code paths, agreement within 1e-3).
2. Patching a clean activation into the clean run changes nothing.
3. The clean–corrupted logit-difference gap is large.
4. Keeping all heads reproduces the clean run.

The pipeline also reproduces published numbers (logit difference 3.64 vs 3.56; Name Mover attention 0.590 vs about 0.59; sufficiency 0.84 vs about 0.87).

## Limitations

- Heads are kept or removed at all token positions, whereas the original analysis uses (head, position) pairs.
- Mean-ablation averages over equal-length prompt groups, not over each template.
- Small, hand-written prompt sets; 100 prompts per task and a single random seed.
- Only GPT-2 small was studied, and MLPs were not analysed.
- Fruits is weak and Vehicles is at chance, so Vehicles is excluded.
- Transfer thresholds are heuristic, and the predictor analysis has only seven points (exploratory).
- In the pruning analysis the ranking is signed, so removing counter-productive heads can push the retained fraction above 1; pruning here means mean-ablation, not weight removal.

## References

- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., Steinhardt, J. (2023). *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.* ICLR 2023. arXiv:2211.00593.
- Merullo, J., Eickhoff, C., Pavlick, E. (2024). *Circuit Component Reuse Across Tasks in Transformer Language Models.* ICLR 2024.
- Tigges, C., Hanna, M., Yu, Q., Biderman, S. (2024). *LLM Circuit Analyses Are Consistent Across Training and Scale.* NeurIPS 2024.
- Zhang, F., Nanda, N. (2024). *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.* ICLR 2024.
- Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., Garriga-Alonso, A. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023.
- Hanna, M., Pezzelle, S., Belinkov, Y. (2024). *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms.* COLM 2024.
- Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT.* NeurIPS 2022.
- Vig, J. et al. (2020). *Investigating Gender Bias in Language Models Using Causal Mediation Analysis.* NeurIPS 2020.
- Michel, P., Levy, O., Neubig, G. (2019). *Are Sixteen Heads Really Better than One?* NeurIPS 2019.
- Nanda, N., Bloom, J. (2022). *TransformerLens.* <https://github.com/TransformerLensOrg/TransformerLens>

## Citation

The manuscript is in preparation and has not been peer reviewed. Until it is published, please cite this repository:

```text
Rao, B. R., Reddy, P. V., Periyasami, K. (2026). Transfer of the Indirect Object Identification Circuit in GPT-2 Small
Across Entity-Tracking Tasks: A Causal-Intervention Study. GitHub repository: https://github.com/bhargavrrao-web/IOI_Circuit_GPT2
```

## Licence

MIT (add a `LICENSE` file to the repository). Templates for the IOI task come from Wang et al.; the original code at <https://github.com/redwoodresearch/Easy-Transformer> is MIT-licensed.

## Contact

Bhargav R Rao (corresponding author) · <bhargavrrao@gmail.com>  
Pushpak V Reddy · <pushpakvreddymtech25@rvu.edu.in>  
Karthikeyan Periyasami · <karthikeyanp@rvu.edu.in>  
RV University, Bangalore
