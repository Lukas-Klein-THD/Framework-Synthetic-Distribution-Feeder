# Model comparison: staged model vs improved FeederGAN (M1+)

Both models were trained in this framework on the same data: tensors_v2, 2,583 graphs of 128 nodes, originals and subfeeders. Both are scored with the same metric instances against the same 512 real reference feeders. The "real (floor)" column is a second, disjoint set of 512 real feeders, scored like a model. Both models use their **final** checkpoints. Neither uses best-checkpoint selection.

```bash
python examples/compare.py --n 512 --device cuda       # scores.json, table.md
python examples/plot_comparison.py --device cuda       # figures/
```

Checkpoints:
- `checkpoints/staged/`: topology at iteration 3000. The attribute stage was retrained after the class-set fix described below.
- `checkpoints/feedergan_improved/feedergan_improved_final.pt`

## Results (n = 512)

| | real floor | staged | improved FeederGAN |
|---|---|---|---|
| **Structure:** precision / recall / coverage | 0.97 / 0.99 / 0.97 | 1.00 / 0.72 / 0.51 | 0.93 / 0.96 / 0.94 |
| **Structure:** FID | 0.2 | 83.0 | **1.7** |
| **Structure:** degree MMD / spectrum MMD | 0.005 / 0.004 | 0.242 / 0.024 | **0.074 / 0.006** |
| **Attributed:** precision / recall / coverage | 0.96 / 0.98 / 0.96 | 0.49 / **0.90 / 0.54** | **0.94** / 0.01 / 0.14 |
| **Attributed:** FID | 11.3 | **178** | 477 |
| Phase rule satisfied: edges / feeders | 1.00 / 1.00 | **1.00 / 0.90** | 0.67 / 0.00 |
| Ampacity tapering: edges | 0.96 | **0.95** | 0.76 |
| Power flow: nodes energised | 1.00 | **1.00** | 0.44 |

All 512 feeders from each model are radial trees, so every electrical check covers the full set.

## Reading

- **Topology: M1+ is better.** It is close to the real floor on every structural metric (FID 1.7 vs 83). The staged topology GAN is precise but narrower: recall 0.72 and coverage 0.51.
- **Attributes: the staged model is better.** M1+ is precise but collapsed: recall 0.01 and coverage 0.14, so its attributed feeders all resemble a small part of the real data. Its per-attribute marginals look plausible (`figures/distributions`), so the collapse is in how attributes are arranged within a feeder, not in their overall frequencies. The staged model covers the data (recall 0.90) at about half the precision.
- **Physical validity: only the staged model is usable without repair.**
  - M1+ breaks the phase-subset rule on a third of its edges, and no feeder is fully valid. As a result, only 44 % of nodes are energised in power flow.
  - The staged model keeps the rule on 99.9 % of edges.
  - M1+'s perfect ANSI and ampacity rates are computed only on its energised part, so they are not comparable to the other columns.
- **Two architectures, two failure modes.** The joint generator learns topology better. The staged split learns attributes and their physical consistency better. This answers the reviewer's concern that the paper rests on a single model.

## Figures (IEEE single column, 3.5 in, 8 pt Times New Roman)

- `fidelity_diversity.pdf`: precision against recall for structure, and for structure with attributes.
- `distributions.pdf`: out-degree, line length, ampacity rating and phase connection.
  - Ratings other than the four common ones are pooled as "Other".
  - The two-phase codes AB, BC and AC are pooled as "Two".
  - About 2 % of the staged model's lines fall into the shortest bin (≤ 1 m), where real feeders have almost none.

## Consistency with custom-approach

The staged numbers here differ from `custom-approach/results/2026-09-30_v2_results`. These checks show the difference comes from training runs, not from the framework:

| check | result |
|---|---|
| custom-approach best topology, scored here | FID 47.2, P/R 0.977/0.844; custom-approach reported 48.5, 0.981/0.854 |
| sampling noise: same checkpoint, sample seed 1 | topology FID 82.7 vs 83.0 |
| framework topology retrain vs custom-approach run | FID 83 vs 43–47, same code and seed: a weaker training run |
| stage 2 on real topology, framework vs custom-approach | P/R/C 0.47/0.92/0.55, FID 165 vs 0.50/0.92/0.53, FID 147 |

Custom-approach's full-pipeline FID of 111 also comes from best-checkpoint selection on a validation split. GAN training runs vary by a factor of about 2 in topology FID. A claim in the paper that depends on the size of a gap should therefore rest on several seeds.

## Fixes made while producing these results

1. **The framework staged plugin was not a faithful port.** Its topology collapsed to chains. It was rewritten as an exact port of custom-approach, and parity was verified by loading custom-approach checkpoints. The broken checkpoints are kept in `checkpoints/_broken_port/`.
2. **The attribute stage took its class sets from the first 256 stored feeders.** It saw 14 of 17 ampacity ratings and 4 of 7 phase codes (no two-phase lines), so it could never generate the rest. It now fits on every training feeder, which brought attributed FID from 218 to 178. A regression test was added. The old checkpoint is `checkpoints/_broken_port/attributes_first256_classes.pt`.
