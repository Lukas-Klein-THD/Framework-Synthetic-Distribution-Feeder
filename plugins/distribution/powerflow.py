"""
Power-flow feasibility as a metric (OpenDSS snapshot solve).

Structural and attributed metrics say whether a feeder *resembles* real ones.
This asks whether current actually flows within limits when it is solved — a
feeder can sit on the real manifold in embedding space and still be
electrically impossible.

Every reported quantity is a fraction where **higher is better**, so the metric
has a single direction and reads cleanly in a scorecard.  Because the evaluator
also scores real feeders against each other, the real-versus-real column gives
the reference values for free.

Two points of interpretation:

* ``energised`` is the one to read first.  A feeder whose phases are inconsistent
  leaves conductors unfed, and every voltage statistic then *flatters* it because
  the disconnected load is not present.
* ``within_ampacity`` is measured against the **generated** rating, so a model
  that assigns implausibly small conductors is penalised.

The harness omits capacitors, regulators and service transformers and uses one
linecode library for every feeder.  Even real feeders therefore fall partly
outside ANSI Range A at a realistic load, so the scores support comparisons
between generated and real feeders, not absolute compliance claims.

Requires ``dss-python``; the metric reports itself as unavailable if absent.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from gridgen.core.feeder import Feeder
from gridgen.core.registry import register
from gridgen.evaluation.base import Direction, Metric

from plugins.distribution import config

#: OpenDSS node suffixes per phase code, and the phase count.
PHASE_NODES = (".1", ".2", ".3", ".1.2", ".2.3", ".1.3", ".1.2.3")
PHASE_COUNT = (1, 1, 1, 2, 2, 2, 3)

_UNITS_TO_FEET = {"ft": 1.0, "kft": 1000.0, "mi": 5280.0, "km": 3280.84}


class LinecodeLibrary:
    """Conductor impedances taken from a source ``Linecodes.dss``.

    Impedances are parsed rather than redirected into OpenDSS: real datasets
    contain entries whose declared phase count disagrees with their matrix
    dimension, which the parser rejects.  Positive-sequence values are taken as
    self minus mutual.
    """

    def __init__(self, path: str):
        import re
        self.path = path
        header = re.compile(r"new\s+linecode\.(\S+)\s+nphases=(\d+).*?normamps=([0-9.]+)",
                            re.I | re.S)
        matrix = re.compile(r"([RX])matrix\s*=\s*\(([^)]*)\)", re.I)
        self.codes: list[tuple[str, int, float, float, float]] = []
        text = open(path, "r", errors="ignore").read()
        for block in re.split(r"(?i)(?=new\s+linecode\.)", text):
            m = header.search(block)
            if not m:
                continue
            seq = {}
            for kind, body in matrix.findall(block):
                rows = [r for r in body.split("|") if r.strip()]
                first = [float(v) for v in rows[0].replace(",", " ").split()]
                seq[kind.upper()] = first[0] - (first[1] if len(first) > 1 else 0.0)
            self.codes.append((m.group(1), int(m.group(2)), float(m.group(3)),
                               seq.get("R", 0.0), seq.get("X", 0.0)))
        if not self.codes:
            raise ValueError(f"no linecodes parsed from {path}")

    def pick(self, n_phases: int, normamps: float) -> tuple[float, float]:
        """Nearest rating with the same phase count -> (r1, x1) ohms per foot."""
        candidates = [c for c in self.codes if c[1] == n_phases] or self.codes
        best = min(candidates, key=lambda c: abs(c[2] - normamps))
        return best[3], best[4]


def to_commands(feeder: Feeder, library: LinecodeLibrary,
                cfg: config.PowerFlowConfig) -> list[str]:
    """Translate one feeder into an OpenDSS circuit definition."""
    adjacency = np.asarray(feeder.adjacency)
    parents = adjacency.argmax(axis=0)
    length = np.clip(np.asarray(feeder.attributes["length"], dtype=float), 1e-6, None)
    ampacity = np.clip(np.asarray(feeder.attributes["ampacity"], dtype=float), 1.0, None)
    phase = np.clip(np.asarray(feeder.attributes["phase"]).astype(int), 0, 6)
    to_feet = _UNITS_TO_FEET[cfg.length_units]

    cmds = ["clear",
            f"new circuit.feeder basekv={cfg.base_kv} pu={cfg.source_pu} phases=3 bus1=n0"]
    for j in range(1, adjacency.shape[0]):
        p = int(parents[j])
        nodes, n_ph = PHASE_NODES[phase[j]], PHASE_COUNT[phase[j]]
        bus1, bus2 = f"n{p}{nodes}", f"n{j}{nodes}"
        r1, x1 = library.pick(n_ph, float(ampacity[j]))
        # zero-sequence must be given explicitly: the OpenDSS defaults are in
        # ohms per mile and would dwarf these per-foot values
        cmds.append(f"new line.l{j} bus1={bus1} bus2={bus2} phases={n_ph} "
                    f"r1={r1:.8g} x1={x1:.8g} r0={r1:.8g} x0={x1:.8g} c1=0 c0=0 "
                    f"normamps={ampacity[j]:.1f} "
                    f"length={length[j] * to_feet:.4f} units=ft")

    for j in range(adjacency.shape[0]):
        n_ph = PHASE_COUNT[phase[j]]
        kv = cfg.base_kv if n_ph == 3 else cfg.base_kv / math.sqrt(3.0)
        cmds.append(f"new load.ld{j} bus1=n{j}{PHASE_NODES[phase[j]]} phases={n_ph} "
                    f"kv={kv:.4f} kw={cfg.kw_per_node:.4f} pf={cfg.power_factor} model=1")
    return cmds


def solve(feeder: Feeder, library: LinecodeLibrary,
          cfg: config.PowerFlowConfig) -> dict[str, float]:
    """Solve one feeder; all returned fractions are higher-is-better."""
    from dss import DSS

    for command in to_commands(feeder, library, cfg):
        DSS.Text.Command = command
    DSS.Text.Command = (f"set voltagebases=[{cfg.base_kv}, "
                        f"{cfg.base_kv / math.sqrt(3.0):.4f}]")
    DSS.Text.Command = "calcvoltagebases"
    DSS.Text.Command = "set mode=snap"
    try:
        DSS.Text.Command = "solve"
    except Exception:
        return {"converged": 0.0, "energised": 0.0, "ansi_a": 0.0,
                "unbalance_ok": 0.0, "within_ampacity": 0.0}

    circuit = DSS.ActiveCircuit
    voltages = np.asarray(circuit.AllBusVmagPu, dtype=float)
    voltages = voltages[np.isfinite(voltages)]
    live = voltages[voltages > 1e-6]
    lo, hi = cfg.ansi_a
    return {
        "converged": float(bool(circuit.Solution.Converged)),
        "energised": float(live.size / max(voltages.size, 1)),
        "ansi_a": float(((live >= lo) & (live <= hi)).mean()) if live.size else 0.0,
        "unbalance_ok": _unbalance_ok(circuit, cfg.unbalance_limit_pct),
        "within_ampacity": _within_ampacity(circuit),
    }


def _unbalance_ok(circuit, limit_pct: float) -> float:
    """Share of three-phase buses within the NEMA unbalance limit."""
    values = []
    for name in circuit.AllBusNames:
        bus = circuit.Buses(name)
        nodes = [n for n in list(bus.Nodes) if n in (1, 2, 3)]
        if len(nodes) < 3:
            continue                      # unbalance is undefined below three phases
        mags = np.asarray(list(bus.puVmagAngle)[0::2], dtype=float)[:len(nodes)]
        if mags.size < 3 or mags.min() <= 1e-6:
            continue
        average = mags.mean()
        values.append(np.max(np.abs(mags - average)) / average * 100.0)
    if not values:
        return float("nan")
    return float((np.asarray(values) <= limit_pct).mean())


def _within_ampacity(circuit) -> float:
    """Share of conductors carrying no more than their generated rating."""
    loadings = []
    index = circuit.Lines.First
    while index:
        rating = float(circuit.Lines.NormAmps)
        if rating > 0:
            mags = np.asarray(list(circuit.ActiveCktElement.CurrentsMagAng)[0::2],
                              dtype=float)
            sending = mags[:max(int(circuit.Lines.Phases), 1)]
            if sending.size and np.all(np.isfinite(sending)):
                loadings.append(sending.max() / rating)
        index = circuit.Lines.Next
    if not loadings:
        return float("nan")
    return float((np.asarray(loadings) <= 1.0).mean())


@register("metric", "power-flow")
class PowerFlowFeasibility(Metric):
    """Operational feasibility of the generated feeders.

    Scores are computed on the *generated* set; the evaluator's real-versus-real
    pass supplies the reference values, so a scorecard shows both without extra
    plumbing.
    """

    def __init__(self, linecodes: str | None = None,
                 cfg: config.PowerFlowConfig | None = None,
                 max_feeders: int = 64):
        self.cfg = cfg or config.PowerFlowConfig()
        path = linecodes or self.cfg.linecodes
        if not path:
            raise ValueError(
                "power-flow evaluation needs a Linecodes.dss; pass linecodes=... "
                "or set GRIDGEN_LINECODES")
        self.library = LinecodeLibrary(path)
        self.max_feeders = max_feeders

    @property
    def name(self) -> str:
        return "powerflow"

    @property
    def direction(self) -> Direction:
        return Direction.HIGHER_IS_BETTER

    def compute(self, real: Sequence[Feeder],
                generated: Sequence[Feeder]) -> Mapping[str, float]:
        subset = list(generated)[: self.max_feeders]
        results = [solve(f, self.library, self.cfg) for f in subset]
        if not results:
            return {}
        keys = results[0].keys()
        return {k: float(np.nanmean([r[k] for r in results])) for k in keys}
