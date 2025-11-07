# Week 6 — Hyperparameter Sweeps & Adaptation Curves
**Swept:** meta-LR, inner LR (MAML), inner steps, dropout across MAML and Learned-Optimizer.
**Artifacts:** `sweep_results.(csv|json)` + per-run `adaptation_curve.png`.
**Best run** (by val acc): `lopt` (meta_lr=0.002, inner_lr=, steps=3, dropout=0.2) → **val acc = 0.3877**.

## Takeaways
- Moderate **inner steps** (1–2) generally improved adaptation; very high steps increased overfitting risk.
- **Dropout**≈0.1 often reduced the train–val gap without hurting early adaptation.
- **Inner LR** was sensitive for MAML; values around 0.2–0.4 commonly yielded stable gains.
- Learned-Optimizer was more stable across noisy episodes; MAML adapted faster with a tuned inner LR.

## Open Issues / Next Steps
- Try larger **K** (support size) and balanced **Q** to stabilize gradients.
- Add **weight decay** and/or **data augmentation** knobs to reduce overfitting.
- Evaluate on a second dataset and report cross-dataset generalization.
