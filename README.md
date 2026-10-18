# gridgen

A pluggable framework for **generating and evaluating synthetic distribution
networks**.

Generative models for distribution feeders are usually evaluated with their own
datasets, splits and metrics, which makes results hard to compare. gridgen
separates the three concerns:

* **Interfaces** (`gridgen/`): what a feeder, a generator, a constraint and a
  metric are. The core imports nothing heavier than NumPy.
* **Plugins** (`plugins/`): datasets, models, constraints and metrics. Any of
  them can be replaced without touching the others.
* **A fixed evaluation protocol**: one held-out split, one scorecard covering
  statistical realism, physical consistency and operational performance, and a
  component-swap analysis that localises errors in modular pipelines.

Two reference models with opposite designs, a staged pipeline and a joint
generator, are included and evaluated end to end.

```bash
pip install -e .            # core (NumPy only)
pip install torch networkx matplotlib dss-python   # plugins, evaluation, figures
```

---

## Contents

1. [Core concepts](#core-concepts)
2. [Evaluation scorecard](#evaluation-scorecard)
3. [Reference models](#reference-models)
4. [Data and held-out split](#data-and-held-out-split)
5. [Reproducing the results](#reproducing-the-results)
6. [Layout](#layout)
7. [Adding your own model](#adding-your-own-model)

---

## Core concepts

### The one type everything shares

Every component consumes and produces a `Feeder`:

```python
Feeder(
    adjacency  = np.ndarray,                      # [M, M]
    attributes = {"length": ..., "phase": ...},   # name -> [M]
    metadata   = {"category": "rural"},
)
```

* **Attributes are named, not positional.** A model producing a different
  attribute set plugs in without changing metrics or exporters.
* **Graph assumptions are declared, not implied.** A `GraphSpec` states whether a
  graph is directed, radial, rooted and fixed-size, so components can *check* what
  they are given.

```python
schema = FeederSchema(
    graph      = GraphSpec(directed=True, radial=True, root=0, num_nodes=128),
    attributes = (AttributeSpec("length", AttributeKind.CONTINUOUS),
                  AttributeSpec("phase",  AttributeKind.CATEGORICAL, n_classes=7)),
)
problems = feeder.validate(schema)      # [] means well formed
```

### Interfaces

| Interface | Purpose |
|---|---|
| `DataSource` | A collection of real feeders, with category grouping and filtering |
| `CategoryLabeler` | Assigns rural / urban-suburban / industrial (or any scheme) |
| `FeederGenerator` | **Produces complete feeders**; what callers and evaluation depend on |
| `TopologyGenerator` | Graph only |
| `AttributeGenerator` | Attributes for graphs it is given |
| `DerivedAttributes` | Quantities computed deterministically, not learned |
| `Constraint` | A domain rule that can be checked and optionally repaired |
| `Descriptor` / `Distance` | The two halves most graph metrics factor into |
| `Metric` | Compares real and generated feeder sets |

### Joint and staged approaches are interchangeable

A model that generates everything at once implements `FeederGenerator`
directly:

```python
class MyJointModel(FeederGenerator):
    @property
    def schema(self): ...
    def sample(self, n, *, category=None, rng=None) -> list[Feeder]: ...
```

A staged pipeline implements the narrow interfaces and is composed:

```python
generator = StagedGenerator(
    topology   = MyTopologyGAN(),
    attributes = MyAttributeModel(),
    derived    = [DistanceToSource(), Depth()],
)
```

Both are `FeederGenerator`s, so evaluation cannot tell them apart and the two
designs can be compared directly. Leaving `attributes=None` yields
topology-only feeders, so a topology model can be developed and scored on its own.

### Categories, constraints, metrics

* **Categories.** A generator advertises the categories it supports
  (`categories = ("rural", ...)`) and receives the request in
  `sample(n, category=...)`. For a `StagedGenerator` the supported set is the
  intersection over its stages.
* **Constraints carry a hard/soft distinction.** `Constraint.calibrate(real)`
  measures a rule on real feeders first: a hard constraint holds on every real
  feeder (target 100 %), a soft tendency does not, and forcing it would overshoot
  reality. `ValidationResult` reports an element-wise and a feeder-wise rate.
  `repair()` is optional.
* **Metrics factor into a descriptor and a distance**
  (`DescriptorMetric(DegreeHistogram(), MaximumMeanDiscrepancy())`), so a new
  descriptor works with every distance and vice versa. `Evaluator` caches
  descriptors and reports a real-versus-real reference for every metric.

### Plugins

Components register themselves by name, so an experiment can be described by
configuration:

```python
from gridgen import register, build

@register("generator", "my-approach")
class MyGenerator(FeederGenerator): ...

generator = build("generator", "my-approach", device="cuda")
```

Importing a plugin package makes its entries available; the core registers
nothing.

---

## Evaluation scorecard

The distribution plugin (`plugins/distribution`) defines the evaluation used for
every model. It has three dimensions, each with its own score on a 0–10 scale
(`plugins/distribution/metrics.py`, `examples/summary_scores.py`).

### 1. Statistical realism

Each feeder is embedded by an **untrained graph isomorphism network**: three
layers, width 35, sum aggregation, jumping-knowledge readout, giving a
105-dimensional embedding. This follows Thompson et al., *On Evaluation Metrics
for Graph Generative Models* (ICLR 2022). Real and generated sets are compared
with **precision** and **recall** (k-nearest neighbours, k = 5, after Kynkäänniemi
et al., 2019) and the **Fréchet distance (FD)** in the embedding space. All
values are averaged over **five random initialisations** of the encoder.

Two metric families share the encoder and differ only in the node input:

| Family | Node input | Evaluates |
|---|---|---|
| Structural realism | node degree | topology alone |
| Attributed realism | log line length, ampacity (one-hot), phase (one-hot) | topology and attributes jointly |

**Score** per family, relative to an independent real sample ("real vs. real"):

$$S_f = 10 \cdot \frac{F_1(P_f, R_f)}{F_1(P_f^{\mathrm{real}}, R_f^{\mathrm{real}})}, \qquad F_1(P, R) = \frac{2PR}{P+R}$$

The harmonic mean prevents high precision from hiding a collapse in recall. FD
is reported but left out of the score, because its scale depends on the encoder
initialisation. The realism score is the mean of the two families.

### 2. Physical consistency

The **phase-subset rule**: a segment's phase set must be a subset of its
parent's. It holds on every real feeder. Two rates are reported:

* **Phase-valid edges** $C_e$: the share of parent–child connections that
  satisfy the rule.
* **Phase-valid feeders** $C_f$: the share of feeders with no violation.

Score: $S_{\mathrm{phys}} = 10 \cdot (C_e + C_f) / 2$, against the absolute
target of 100 %.

### 3. Operational performance

An **OpenDSS snapshot power flow**:
- **Circuit:** 12.47 kV, source at 1.03 pu, 50 kW per node at power factor 0.95.
- **Conductors:** chosen from the reference dataset's line codes by phase count
  and nearest ampacity.
- **Why these operating conditions:** the load level reproduces the voltage
  profile and line loading of a realistic load on the reference feeders.

The metrics follow the voltage, thermal and power-quality criteria of
distribution hosting-capacity analysis:

| Metric | Definition |
|---|---|
| Energised phases $E$ | share of bus phases with a voltage above 10⁻⁶ pu |
| Voltage acceptance $V$ | share of energised phases within 0.95–1.05 pu (from ANSI C84.1 Range A) |
| Thermal compliance $W$ | share of lines whose largest phase current is within the conductor rating |
| Voltage unbalance $U$ | share of energised three-phase buses with an unbalance of at most 3 % |

Score: $S_{\mathrm{op}} = 10 \cdot E \cdot (V + W + U) / 3$. $V$, $W$ and $U$ are
measured on the energised part of a network, so $E$ gates them: a mostly unfed
network cannot score well on the voltages of its few energised buses.

**Overall score:** the mean of the three dimension scores, weighted equally.

### Error localisation in modular pipelines

For a staged generator, each stage is replaced in turn by an untrained
component: a uniform random tree for the topology, and attributes sampled
independently per node for the attribute model. The topology swap moves the
structural metrics and the attribute swap moves the attributed recall, so each
stage is diagnosed by its own metric family (`examples/full_results.py`, section
`baselines`, and `examples/gin_seed_sweep.py`).

### Post-hoc phase repair

`PhaseSubset.repair()` clamps every segment to the phases of its parent, root
outward, and leaves topology, lengths and ratings unchanged. The repaired
variant of each generator is scored next to the raw output. Repaired feeders
satisfy the phase rule by construction, so their physical score is always 10.
The scorecard therefore attributes the generator's merit to the raw output and
shows what repair costs in realism and operation.

---

## Reference models

| | `staged` | `feedergan-improved` |
|---|---|---|
| Interface | `TopologyGenerator` + `AttributeGenerator`, composed by `StagedGenerator` | `FeederGenerator` directly |
| Topology | self-attention encoder, autoregressive parent selection with out-degree conditioning; a tree by construction | soft adjacency, reconstructed with FeederGAN's Algorithm 2 |
| Attributes | directed-GIN encoder, feeder profile; mixture density head for length, gated Markov heads for ampacity and phase (copy the parent's value unless a change is predicted) | generated with the topology in one pass; straight-through Gumbel-softmax categories, ampacity conditioned on phase |
| Critic | directed GIN | GCN with sum aggregation, degree features, mean + max readout |
| Training | WGAN-GP (topology), maximum likelihood (attributes) | WGAN-GP on the joint output |

`feedergan-improved` is FeederGAN (Liang et al., *IEEE Trans. Smart Grid*, 2021)
with the changes that make it trainable on this data. The details are in
`plugins/feedergan_improved/` (module docstrings and `config.py`). Each model
plugin depends only on `plugins/distribution`, so it doubles as a template.

---

## Data and held-out split

The models use preprocessed feeder graphs built from public synthetic
distribution-network models (NREL SMART-DS SFO-2018, and the GSO and SAF base
networks), one `.pt` file per feeder or subfeeder.
- **Nodes:** each node is a line segment. Switching and protection devices are
  merged into the segments they connect, and lengths are in km.
- **Size:** feeders with fewer than 128 nodes are dropped, and larger ones are
  truncated to their first 128 nodes in breadth-first order.

```bash
export GRIDGEN_DATA_DIR=/path/to/tensors_v2
export GRIDGEN_LINECODES=/path/to/Linecodes.dss     # for the power-flow metric
```

**Held-out split** (`plugins/distribution/splits/heldout_v2.json`, built by
`examples/make_split.py`):
- **Unit:** the source feeder. A feeder and every subfeeder sampled from it land
  in the same part.
- **Stratified by category,** with 25 % held out (seed 0).

| Part | Source feeders | Graphs |
|---|---|---|
| train | 1,220 | 1,924 |
| reference (test half 1) | 204 | 336 |
| comparison (test half 2) | 203 | 323 |

**What uses which part:**
- **Training data only:** the models, the staged model's profile pool, the
  attribute encoding of the embedding, and the marginals of the
  independent-attribute control.
- **Held-out data:** generated feeders are scored against the reference half.
  The real-vs-real row compares the two halves, which share no source feeder.

`TensorFeederSource(split="train" | "test" | "reference" | "comparison")` reads
any part.

---

## Reproducing the results

Trained weights are not included. Set a checkpoint directory and run the steps
in order:

```bash
export GRIDGEN_CHECKPOINT_DIR=checkpoints/heldout

python examples/make_split.py                        # only if the split file is missing
python examples/train.py --model feedergan-improved --device cuda    # ~10 min
python examples/train.py --model staged --device cuda                # ~4 h

OUT=results/heldout
python examples/full_results.py   --device cuda --out $OUT    # all sections, n = 256
python examples/gin_seed_sweep.py --device cuda --out $OUT    # five encoder seeds, realism scores
python examples/summary_scores.py $OUT                        # dimension and overall scores
python examples/make_paper_tables.py $OUT                     # LaTeX tables
python examples/plot_paper_figures.py --device cuda --out $OUT/figures
```

**Outputs:**
- **Data:** `numbers.json` (all sections), `gin_seeds.json` (per-seed realism
  metrics and scores), `summary_scores.json`.
- **Paper material:** `paper_tables.tex` (realism, physical, operational,
  localisation and summary tables) and IEEE-style figures.
- **Readable logs:** one per section in `logs/`.

**Flags:**
- `--all-data` reproduces the earlier protocol, where both real sets were drawn
  from all graphs.
- `train.py --split all` trains on every graph.

> The power-flow section of `full_results.py` currently calls the OpenDSS
> harness of the companion code (`--custom-approach`). The framework's own
> `PowerFlowFeasibility` metric (`plugins/distribution/powerflow.py`) computes
> the same four quantities and is the intended replacement.

`examples/quickstart.py` is a short end-to-end walkthrough. `examples/compare.py`
gives a quick side-by-side of both models with a single encoder seed.

---

## Layout

```
gridgen/                     the framework: interfaces only, NumPy only
  core/         Feeder, schemas, plugin registry
  data/         DataSource, CategoryLabeler
  generation/   generator interfaces + StagedGenerator composition
  constraints/  Constraint, ValidationResult
  evaluation/   Descriptor, Distance, Metric, Evaluator, Scorecard
  pipeline.py   end-to-end run

plugins/distribution/        shared domain plugin
  config.py     data location, graph size, split file, power-flow operating point
  data.py       DataSource over preprocessed tensors, held-out split, labeller
  splits/       the shipped held-out split
  constraints.py phase subset (hard, with repair) and ampacity tapering (soft)
  metrics.py    random-GIN embedding, distances, scores
  powerflow.py  OpenDSS snapshot solve as a metric
  derived.py    attributes computed from a finished feeder
  reconstruction.py  soft adjacency -> graph (Algorithm 2), BFS relabelling

plugins/staged/              reference model 1: topology, then attributes
plugins/feedergan_improved/  reference model 2: topology and attributes jointly

examples/
  make_split.py         build the held-out split
  train.py              train either model (training split by default)
  full_results.py       every evaluation section for both models
  gin_seed_sweep.py     realism metrics over five encoder seeds + scores
  summary_scores.py     realism / physical / operational / overall scores
  make_paper_tables.py  LaTeX tables from the result files
  plot_paper_figures.py, plot_comparison.py, paper_style.py   figures
  compare.py, quickstart.py

docs/adding_a_model.md       how to plug in your own components
tests/                       interface tests + plugin tests (pytest tests/)
```

---

## Adding your own model

Implement `FeederGenerator` (joint) or `TopologyGenerator` and
`AttributeGenerator` (staged), register it, and evaluate it with the same split
and scorecard. Start from `docs/adding_a_model.md`; either reference model
plugin serves as a template.

## Status

- **Interfaces:** complete.
- **Reference models:** both train and evaluate end to end under the held-out
  protocol.
- **Tests:** the suite has 42 tests (`pytest tests/`). Tests that need the
  dataset skip themselves when it is not available.

## License

See `LICENSE`.
