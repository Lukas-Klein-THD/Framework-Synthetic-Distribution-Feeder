"""
Graph helpers shared by the descriptors and by tests: an undirected view of an
adjacency matrix and a radial-tree check.
"""

from __future__ import annotations

import numpy as np


def undirected(adjacency: np.ndarray) -> np.ndarray:
    """Symmetric 0/1 matrix without self-loops."""
    a = (np.asarray(adjacency) != 0)
    u = (a | a.T).astype(float)
    np.fill_diagonal(u, 0.0)
    return u


def is_radial_tree(adjacency: np.ndarray) -> bool:
    """Connected, acyclic, single-rooted: one node without a parent, every other
    node with exactly one, and every node reachable from the root."""
    a = np.asarray(adjacency) != 0
    m = a.shape[0]
    in_degree = a.sum(0)
    roots = np.flatnonzero(in_degree == 0)
    if len(roots) != 1 or (np.delete(in_degree, roots) != 1).any():
        return False
    seen, frontier = {int(roots[0])}, [int(roots[0])]
    while frontier:
        children = np.flatnonzero(a[frontier].any(0))
        frontier = [int(c) for c in children if int(c) not in seen]
        seen.update(frontier)
    return len(seen) == m
