# gridgen

A pluggable framework for **generating and evaluating synthetic distribution
networks**.

The package contains **interfaces only**. Models, datasets, constraints and
metrics live in plugin packages; the core imports nothing heavier than NumPy. The
point is that an approach, a metric, a dataset or a constraint can be replaced
without touching anything else — and that results from different approaches are
comparable because they were produced by the same harness.

```
pip install -e .
```

---

## The one type everything shares

Every component consumes and produces a `Feeder`:

```python
Feeder(
    adjacency  = np.ndarray,              # [M, M]
    attributes = {"length": ..., "phase": ...},   # name -> [M]
    metadata   = {"category": "rural"},
)
```

Two decisions make it extensible:

* **Attributes are named, not positional.** A model producing a different
  attribute set plugs in without changing metrics or exporters.
* **Graph assumptions are declared, not implied.** A `GraphSpec` states whether a
  graph is directed, radial, rooted and fixed-size, so radial and meshed networks
  coexist and components can *check* what they are given.

```python
schema = FeederSchema(
    graph      = GraphSpec(directed=True, radial=True, root=0, num_nodes=128),
    attributes = (AttributeSpec("length", AttributeKind.CONTINUOUS),
                  AttributeSpec("phase",  AttributeKind.CATEGORICAL, n_classes=7)),
)
problems = feeder.validate(schema)      # [] means well formed
```

## Interfaces

| Interface | Purpose |
|---|---|
| `DataSource` | A collection of real feeders, with category grouping and filtering |
| `CategoryLabeler` | Assigns rural / urban-suburban / industrial (or any scheme) |
| `FeederGenerator` | **Produces complete feeders** — what callers and evaluation depend on |
| `TopologyGenerator` | Graph only |
| `AttributeGenerator` | Attributes for graphs it is given |
| `DerivedAttributes` | Quantities computed deterministically, not learned |
| `Constraint` | A domain rule that can be checked and optionally repaired |
| `Descriptor` / `Distance` | The two halves most graph metrics factor into |
| `Metric` | Compares real and generated feeder sets |

### Monolithic and staged approaches are interchangeable

This is the main design requirement. A model that generates everything at once
implements `FeederGenerator` directly:

```python
class MyDiffusionModel(FeederGenerator):
    @property
    def schema(self): ...
    def sample(self, n, *, category=None, rng=None) -> list[Feeder]: ...
```

A staged pipeline implements the narrow interfaces and is composed:

```python
generator = StagedGenerator(
    topology   = MyTopologyGAN(),
    attributes = MyAttributeMDN(),
    derived    = [DistanceToSource(), Depth()],
)
```

Both are `FeederGenerator`s. Evaluation, export and the pipeline cannot tell them
apart — so a staged and a monolithic approach can be compared directly.

Leaving `attributes=None` yields topology-only feeders, which lets a topology
model be developed and scored on its own.

### Categories

A generator advertises what it supports and receives the request:

```python
class MyModel(FeederGenerator):
    categories = ("rural", "urban-suburban", "industrial")
    def sample(self, n, *, category=None, rng=None): ...

feeders = generator.sample(64, category="rural")
```

For a `StagedGenerator` the supported set is the intersection over its stages, so
a pipeline only claims a category every stage can honour.

### Constraints carry a hard/soft distinction

```python
class PhaseSubset(Constraint):
    hard = True                       # real data satisfies it exactly
    def check(self, feeder) -> np.ndarray:   # True where satisfied
    def repair(self, feeder) -> Feeder:      # optional
```

`Constraint.calibrate(real)` measures the rule on **real** feeders first. That is
what separates a hard constraint (real rate 1.0, target 1.0) from a soft
engineering tendency (real rate well below 1.0, where forcing 1.0 would overshoot
reality). `ValidationResult` reports both an element-wise and a feeder-wise rate,
because a rule violated on a few percent of edges can still leave most feeders
invalid.

### Metrics

Most graph metrics factor into a descriptor and a distance, so they are separate:

```python
metric = DescriptorMetric(DegreeHistogram(), MaximumMeanDiscrepancy())
```

A new descriptor immediately works with every registered distance and vice versa.
Anything that does not factor this way implements `Metric` directly.

`Evaluator` caches descriptor output, so several distances sharing one expensive
embedding compute it once. It also reports a **real-versus-real reference** for
every metric — embedding scores have no absolute meaning and must be read against
the value two halves of the real data achieve.

## Putting it together

```python
pipeline = Pipeline(
    source      = MyDataSource(...),
    generator   = generator,
    constraints = [PhaseSubset(), AmpacityTapering()],
    metrics     = [degree_mmd, embedding_fid],
)
print(pipeline.run(n=512, category="rural"))
```

## Plugins

Components register themselves by name, so an experiment can be described by
configuration:

```python
from gridgen import register, build

@register("generator", "my-approach")
class MyGenerator(FeederGenerator): ...

generator = build("generator", "my-approach", device="cuda")
```

Importing a plugin package is what makes its entries available; the core
registers nothing.

## Layout

```
gridgen/                     the framework — interfaces only, NumPy only
  core/         Feeder, schemas, plugin registry
  data/         DataSource, CategoryLabeler
  generation/   generator interfaces + StagedGenerator composition
  constraints/  Constraint, ValidationResult
  evaluation/   Descriptor, Distance, Metric, Evaluator, Scorecard
  pipeline.py   end-to-end run

plugins/distribution/        shared domain plugin: data, constraints, metrics
  config.py     data location, graph size, power-flow operating point
  data.py       DataSource over preprocessed tensors + category labeller
  derived.py    attributes computed from a finished feeder
  constraints.py phase subset (hard) and ampacity tapering (soft)
  metrics.py    descriptors, distances, attribute encoder
  powerflow.py  OpenDSS snapshot solve as a metric
  reconstruction.py  soft adjacency -> graph (Algorithm 2), BFS relabelling

plugins/staged/              reference model 1: topology, then attributes
  topology.py   Stage 1 — adversarial topology model, radial by construction
  profile.py    feeder-level conditioning vector
  attributes.py Stage 2 — conditional mixture density model
  training.py   training loops for both stages

plugins/feedergan_improved/  reference model 2: topology and attributes jointly
  encoding.py   attribute representation
  networks.py   generator and critic
  generator.py  the FeederGenerator adapter
  training.py   WGAN-GP training

examples/train.py            train either model on the shared data
examples/compare.py          score both models under one evaluation
examples/quickstart.py       end-to-end walkthrough
docs/adding_a_model.md       how to plug in your own components
tests/                       interface tests + plugin tests
```

## Reference models

Two models with opposite designs show that the framework compares approaches
rather than describing one:

| | `staged` | `feedergan-improved` |
|---|---|---|
| Interface | `TopologyGenerator` + `AttributeGenerator`, composed by `StagedGenerator` | `FeederGenerator` directly |
| Topology | autoregressive parent selection, a tree by construction | soft adjacency, reconstructed with FeederGAN's Algorithm 2 |
| Attributes | maximum likelihood on given topologies, feeder-level profile | generated with the topology in one pass, judged by a graph critic |
| Training | WGAN-GP (Stage 1), MLE (Stage 2) | WGAN-GP on the joint output |

Both read the same data (every feeder and sub-feeder with at least 128 nodes,
truncated to its first 128 BFS nodes) and are scored by the same metric
instances against the same real reference set:

```bash
python examples/train.py --model staged --device cuda
python examples/train.py --model feedergan-improved --device cuda
python examples/compare.py --n 512 --device cuda
```

Each model plugin depends only on `plugins/distribution`, so it doubles as a
template. To add your own model, start from `docs/adding_a_model.md`.

## Data

The models are trained on preprocessed feeder graphs built from public
distribution-network models (NREL SMART-DS SFO-2018, and the GSO and SAF base
networks), one `.pt` file per feeder or sub-feeder. Each node is a line segment;
switching and protection devices are merged into the segments they connect, and
lengths are in km. Point the framework at the files with

```bash
export GRIDGEN_DATA_DIR=/path/to/tensors_v2
```

The power-flow metric also needs a SMART-DS `Linecodes.dss`
(`GRIDGEN_LINECODES`) and `dss-python`. Tests that need the data skip
themselves when it is not available.

## Status

Interfaces are complete, both reference models train and evaluate end to end,
and everything is covered by the test suite (`pytest tests/`).
