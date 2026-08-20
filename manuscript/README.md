# LeWM adaptive-computation manuscript

## Document status

This directory contains the anonymous archival submission for the NeurIPS 2026
Physical World AI workshop. It uses the official NeurIPS 2026 double-blind
workshop style, has exactly eight main-paper pages, and places references,
supplementary material, and the required NeurIPS checklist after the
page-limited body. The main paper contains an allocator diagram, one
consolidated results figure, and one consolidated confirmation table.

The manuscript's bounded claim is:

> At matched expected counted FLOPs, history-conditioned allocation improved
> fresh-cohort next-latent prediction for two frozen artifact/population pairs:
> Cube PlanOracle and a fresh confirmation of a pilot-generated, PushT-specific
> binary depth rule. The result does not establish contact recognition,
> retraining stability, transfer, systems efficiency, planning, or control.

The submission is double-blind. No model experiment was run during manuscript
production; the edit reorganizes and consolidates the already-verified
evidence.

Primary files:

- `main.tex` — eight-page paper plus references and supplementary material
- `neurips_2026.sty` — official NeurIPS 2026 workshop style
- `checklist.tex` — completed official NeurIPS 2026 paper checklist
- `references.bib` — verified focused bibliography
- `figures/` — consolidated main evidence plus two supplementary figures
- `main.pdf` — ignored local build; the retained submission PDF lives outside Git
- `SOURCE_MAP.md` — claim/figure/table-to-artifact traceability
- `generate_figures.py` — the sole read-only extraction/figure script

## Build

The figure script expects the repository layout used by this checkout. From
the repository root:

```bash
/Users/rishisim/.cache/lewm-v2-venv/bin/python manuscript/generate_figures.py
```

The script hash-checks every indexed numerical source that it reads and then
writes only:

```text
manuscript/figures/figure1_confirmed_effects.pdf
manuscript/figures/figure2_cube_shift_map.pdf
manuscript/figures/figure3_multistep_planning.pdf
```

The bundled LaTeX workflow selected the installed TeX Live toolchain. Build
from the LaTeX skill root:

```bash
cd /Users/rishisim/.codex/plugins/cache/openai-bundled/latex/0.2.6
python3 scripts/compile_latex.py \
  /Users/rishisim/Documents/research/lewm-p2-contact/manuscript/main.tex
```

The command runs `latexmk`, BibTeX, and the required reruns and writes
`manuscript/main.pdf`.

## Validation record

- Synthesis checker before drafting: passed with
  `"all_checks_passed": true` at
  `2026-07-19T03:05:45.685851+00:00`.
- Figure reconstruction: `generate_figures.py` passed all source-hash,
  endpoint, bootstrap-quantile, sample-count, call-histogram, compute-ledger,
  and planning-adequacy assertions.
- Synthesis checker after the final visual pass: passed with
  `"all_checks_passed": true` at
  `2026-07-19T03:48:26.645307+00:00`.
- LaTeX compilation: run with TeX Live 2026 using the official NeurIPS style.
- PDF inspection: the submission build is rendered and checked page by page;
  the eight-page body precedes references, supplementary material, and the
  completed checklist.

To rerun the synthesis validation from the repository root:

```bash
/Users/rishisim/.cache/lewm-v2-venv/bin/python \
  runs/lewm_paper_evidence_synthesis/evidence_check.py
```

## Focused related-work search note

The related-work pass was performed on July 18, 2026 (America/Phoenix) using
primary sources only: original papers, official proceedings pages, OpenReview,
official conference pages, and arXiv records. It was intentionally bounded to
adaptive computation/dynamic depth, conditional exit/routing under compute
constraints, latent world-model prediction, and directly relevant adaptive
planning computation. It is not an exhaustive survey.

The closest scope checks were:

- [Looped World Models, arXiv:2606.18208](https://arxiv.org/abs/2606.18208):
  a parameter-shared recurrent transformer with a learned sigmoid early-exit
  gate and deferred decoding, evaluated in text-interactive ScienceWorld and
  ALFWorld. This is cited as adjacent rather than treated as the same visual
  transition-level matched-ledger test.
- [Adaptive Compute in Latent World Models,
  arXiv:2607.10203](https://arxiv.org/abs/2607.10203): fixed exits and depth
  regimes across nine DeepMind Control tasks, without a learned per-transition
  router.
- [Sparse Imagination, official ICLR 2026
  page](https://iclr.cc/virtual/2026/poster/10008222): adapts token sparsity in
  world-model planning rather than single-transition refiner depth.
- [Adaptive Rollout Length,
  arXiv:2206.02380](https://arxiv.org/abs/2206.02380): adapts rollout horizon,
  a distinct planning-compute axis.

Additional bibliography records were checked against the official sources for
ACT, Universal Transformers, PonderNet, BranchyNet, Adaptive Neural Networks,
SkipNet, BlockDrop, Mixture-of-Depths, Mixture of Recursions, PlaNet, Dreamer,
IRIS, DIAMOND, and LeWorldModel. Because priority remains a moving target, the
manuscript uses restrained complementary positioning and makes no priority
claim.

The August 2026 revision also checked the primary arXiv records for
Visuo-Tactile World Models (arXiv:2602.06001) and FeelWorld
(arXiv:2607.24267). They are cited to distinguish explicitly contact-grounded
world models from this paper's latent/action-history allocation mechanism.

## Scientific boundaries retained

- Counted arithmetic is separated from device latency and energy.
- Analytic fixed-depth mixtures are expectation-level comparators; runnable
  schedules are identified separately and their existing results are reported.
- PushT refiner, whitening, and gate fitting are environment-specific.
- The negative four-depth PushT pilot precedes the derived binary hypothesis
  and its disjoint confirmation.
- Prediction quality is separated from planning and control utility.
- All PushT WeakPolicy roles have zero task success.
- The frozen PushT fit-whitening endpoint uses a `1e-3` covariance floor; the
  post-outcome floor sensitivity appears only as a limitation.
