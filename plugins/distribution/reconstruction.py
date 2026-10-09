"""
From a generator's soft adjacency to a binary feeder graph (shared by all models).

``algorithm2`` is the paper's Algorithm 2: for every node pair keep only the
stronger direction, row-normalise, and give each node its strongest remaining
parent.  The critic sees this reconstruction during training (straight-through),
so it is also the faithful one at evaluation.

Nothing in the procedure guarantees a tree: two nodes can end up as roots, or
parents can form a cycle.  ``to_rooted_order`` therefore relabels a result only
when it is a tree; anything else is returned as it is and reported by the
pipeline's schema check rather than silently repaired.
"""

from __future__ import annotations

import numpy as np
import torch


def algorithm2(soft: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Soft adjacency [B, M, M] (column = distribution over parents) -> binary."""
    b, m, _ = soft.shape
    eye = torch.eye(m, device=soft.device, dtype=soft.dtype).unsqueeze(0)
    a = soft.detach() * (1.0 - eye)

    # lines 1-6: keep the stronger direction of each pair (ties go upward)
    upper = torch.triu(torch.ones(m, m, device=soft.device, dtype=torch.bool), 1).unsqueeze(0)
    keep = (upper & (a >= a.transpose(1, 2))) | (upper.transpose(1, 2) & (a > a.transpose(1, 2)))
    directed = torch.where(keep, a, torch.zeros_like(a))

    # line 7: row normalisation
    normed = directed / directed.sum(dim=2, keepdim=True).clamp_min(eps)

    # lines 8-13: strongest parent per column, if any
    values, parents = normed.max(dim=1)                       # [B, M]
    binary = torch.zeros_like(normed)
    binary.scatter_(1, parents.unsqueeze(1), (values > 0).to(normed.dtype).unsqueeze(1))
    return binary * (1.0 - eye)


def straight_through(soft: torch.Tensor) -> torch.Tensor:
    """Forward: Algorithm 2's binary graph.  Backward: the soft adjacency's gradient."""
    return algorithm2(soft) - soft.detach() + soft


def to_rooted_order(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Relabel a tree in BFS order from its root.

    Returns ``(adjacency, order)``: for a tree, the relabelled adjacency (root =
    node 0, every parent before its children) and the permutation applied; for
    anything else the input unchanged and ``None``.
    """
    a = np.asarray(adjacency) != 0
    m = a.shape[0]
    in_degree = a.sum(0)
    roots = np.flatnonzero(in_degree == 0)
    if len(roots) != 1 or (np.delete(in_degree, roots) != 1).any():
        return np.asarray(adjacency, dtype=np.float32), None
    order, frontier, seen = [int(roots[0])], [int(roots[0])], {int(roots[0])}
    while frontier:
        nxt = []
        for node in frontier:
            for child in np.flatnonzero(a[node]):
                if int(child) not in seen:
                    seen.add(int(child))
                    nxt.append(int(child))
        order.extend(nxt)
        frontier = nxt
    if len(order) != m:                          # a cycle not reachable from the root
        return np.asarray(adjacency, dtype=np.float32), None
    perm = np.asarray(order)
    return a[np.ix_(perm, perm)].astype(np.float32), perm
