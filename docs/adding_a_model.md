# Adding your own model

The framework defines interfaces; everything else is a plugin. This walks through
the three cases you are most likely to want, using the reference plugin in
`plugins/feedergan/` as the worked example.

The rule to keep in mind: **implement the narrowest interface that fits.** A model
that only produces graphs should be a `TopologyGenerator`, not a
`FeederGenerator` — that is what lets it be evaluated on its own and swapped into
someone else's pipeline.

---

## 1. A model that generates everything at once

Implement `FeederGenerator` directly:

```python
from gridgen import Feeder, FeederGenerator, FeederSchema, GraphSpec, register

@register("generator", "my-joint-model")
class MyJointModel(FeederGenerator):
    categories = ("rural", "urban-suburban")     # omit if unconditional

    @property
    def schema(self) -> FeederSchema:
        return FeederSchema(GraphSpec(radial=False, num_nodes=None), MY_ATTRIBUTES)

    def sample(self, n, *, category=None, rng=None) -> list[Feeder]:
        self._check_category(category)
        return [Feeder(adjacency, attributes) for ... ]
```

Note the schema: this model declares `radial=False` and a variable node count.
Nothing else has to change — metrics and constraints read the schema rather than
assuming, and `Feeder.validate(schema)` will check conformance for you.

## 2. Replacing one stage of a staged pipeline

Implement `TopologyGenerator` (see `plugins/feedergan/topology.py`):

```python
@register("topology_generator", "my-topology")
class MyTopology(TopologyGenerator):
    @property
    def graph_spec(self) -> GraphSpec:
        return GraphSpec(directed=True, radial=True, root=0, num_nodes=128)

    def sample_topology(self, n, *, category=None, rng=None) -> list[np.ndarray]:
        ...
```

then compose it with the existing attribute stage:

```python
generator = StagedGenerator(topology=MyTopology(),
                            attributes=MDNAttributes(checkpoint=...),
                            derived=[TopologicalAttributes()])
```

Because both are `FeederGenerator`s, this is directly comparable to the joint
model above under the same metrics. Replacing the attribute stage instead means
implementing `AttributeGenerator.generate(adjacencies, ...)`, which receives
graphs rather than producing them — that is what makes it possible to measure
attribute quality on *real* topologies and separate the two stages' errors.

## 3. Adding a metric

Most graph metrics factor into a descriptor and a distance, so add whichever half
is new:

```python
class MyDescriptor(Descriptor):
    name = "my-descriptor"
    def __call__(self, feeders) -> np.ndarray:      # [N, D]
        ...

metric = DescriptorMetric(MyDescriptor(), MaximumMeanDiscrepancy())
```

A new descriptor immediately works with every existing distance, and vice versa.
For something that reads the graph directly — a power-flow solve, a domain
statistic — implement `Metric` instead and return a mapping of named scores
(see `plugins/feedergan/powerflow.py`).

**Declare a direction.** A scorecard mixing "lower is better" and "higher is
better" without labels is unreadable.

## 4. Adding a constraint

```python
@register("constraint", "my-rule")
class MyRule(Constraint):
    hard = True                      # real data satisfies it exactly

    @property
    def name(self): return "my-rule"

    def check(self, feeder) -> np.ndarray:     # True where satisfied
        ...

    def repair(self, feeder) -> Feeder:        # optional
        ...
```

Two things the interface asks you to be honest about:

- **`hard` versus soft.** Call `constraint.calibrate(real_feeders)` before
  reporting anything. If real feeders satisfy the rule on only 80% of feeders it
  is a soft engineering tendency, and enforcing it to 100% makes generated
  feeders *less* realistic, not more.
- **Only report elements the rule applies to.** Omit the rest rather than
  counting them as satisfied, or the rate is diluted and looks better than it is.

`ValidationResult` reports element-wise and feeder-wise rates separately, because
a rule violated on 3% of edges can still leave 90% of feeders invalid.

## 5. Adding a data source

Implement `__len__`, `__getitem__` and `schema` (see `plugins/feedergan/data.py`).
Grouping, filtering and category views come from the base class. Keep
classification in a separate `CategoryLabeler`: how a feeder is labelled — folder
layout, naming convention, clustering — is independent of how it is loaded.

---

## Interpreting results

Three habits the interfaces are built to encourage:

**Read scores against the real-versus-real reference.** Embedding metrics have no
absolute scale. `Evaluator.run(..., with_reference=True)` computes the value two
disjoint halves of the real data achieve, and that is the number your result
should be compared with — not 0, and not 1.

**Read precision and recall together.** A model emitting a narrow set of
plausible feeders scores high precision and near-zero recall. Either number alone
can rank a degenerate model above a good one.

**Marginal agreement is cheap.** Matching a one-dimensional distribution says
little: a model that samples each node independently from the right marginal, with
no structure at all, matches marginals almost perfectly. Judge quality on
quantities that require structure — constraint validity, attribute persistence
along edges, the attributed joint embedding, power-flow behaviour.
