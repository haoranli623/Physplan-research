# PhysPlan paper

Submission source for:

**Where Does Latent Planning Fail? Controlled Bottleneck Attribution for World-Model Planning**

Target: NeurIPS 2026 workshop, *Physical Understanding for Decision-Making: Bridging Foundation Models and Reliable Agents*.

## Build

The project uses the official NeurIPS 2026 style in anonymized double-blind workshop mode.

```powershell
cd D:\projects\physplan\paper
D:\anaconda\envs\torch-gpu\python.exe figures\make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The figure script performs no experiment: it reads frozen JSON summaries, asserts the authoritative headline values, and recreates the result plot. To clean auxiliary files:

```powershell
latexmk -c
```

An Overleaf build can upload the complete `paper/` tree and compile `main.tex` with pdfLaTeX.

## Deliverables

- `main.tex` and `sections/`: anonymized Full Paper source.
- `main.pdf`: locally compiled submission PDF.
- `references.bib`: bounded, primary-source bibliography.
- `appendix.tex`: frozen protocol and reproducibility supplement.
- `figures/attribution_pipeline.tex`: reproducible TikZ attribution diagram.
- `figures/cross_regime_results.pdf`: vector result figure generated from frozen artifacts.
- `tables/cross_regime.tex`: main cross-regime table.
- `CLAIM_AUDIT.md`: sentence-to-evidence audit and reviewer attack pass.
- `EVIDENCE_MAP.md`: private source map created from the full technical report.
- `../SUBMISSION_SPEC.md`: verified live workshop requirements and unresolved format details.

## Figures and tables

Main paper:

1. Figure 1: controlled bottleneck-attribution pipeline and adaptive CEM loop.
2. Table 1: prediction/metric gaps, diagnosis, and targeted intervention across three regimes.
3. Figure 2: cross-regime headroom and intervention improvements with confidence intervals.

Appendix:

1. Table 2: supporting frozen measurements and repair outcome counts.

## Page count and anonymity

- Main text through conclusion: **8 pages**.
- References: **2 pages**.
- Appendix: **2 pages**.
- Compiled PDF: **11 pages total**.
- Verified Full Paper limit: **9 main pages**, excluding references and appendix.
- The submission build uses `dblblindworkshop`; author identities, affiliations, acknowledgments, and repository URLs are absent.

## Submission TODOs

The live sources do not explicitly resolve whether the appendix must be concatenated with the paper PDF, whether the main-conference checklist is mandatory, notification/camera-ready dates, exact workshop day, or pre-conference public visibility. See `../SUBMISSION_SPEC.md`. Before upload, the authors must also complete OpenReview author profiles, reviewer nomination (or `None`), keywords, and administrative confirmations.

## Frozen evidence

Paper preparation did not rerun experiments. Final code commit: `ceb6122`. Frozen H6 protocol commit: `3cbd3c5`. Repository checks confirm 30/30 H6 raw JSON/NPZ pairs, 18/18 related tests, 53/53 immutable historical artifacts, and 117/117 SHA-256 manifest checks.
