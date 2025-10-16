"""General utilities for clustering vector embeddings and finding nearest relationships.

Designed to be **lightweight and dependency-friendly**:
- **Required**: ``numpy``
- **Optional**: ``scikit-learn`` for clustering and non‑euclidean distance metrics
- **Optional**: ``pandas`` for tabular summaries

The API is **domain-agnostic** (works for any items with optional metadata), but
includes small compatibility shims for older, paper-centric function names.

Key concepts
------------
- *items*: arbitrary Python mappings (e.g., dicts) carrying metadata like a
  grouping key (default: ``"category"``) and a display name (default: ``"title"``).
- *groups*: any categorical partition of items (e.g., category, label, source).

Notes on distances
------------------
- If scikit‑learn is installed, any metric accepted by
  :func:`sklearn.metrics.pairwise.pairwise_distances` can be used.
- Without scikit‑learn, only Euclidean distance is supported.

"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:  # Optional sklearn imports
    from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-not-found]
    from sklearn.metrics import pairwise_distances as _sk_pairwise_distances  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    AgglomerativeClustering = None  # type: ignore[assignment]
    _sk_pairwise_distances = None  # type: ignore[assignment]


# -----------------------------
# Public type aliases
# -----------------------------
Item = Mapping[str, Any]
ClusterInfo = Dict[str, Any]
Pair = Tuple[float, int, int]  # (distance, index_a, index_b)

DEFAULT_GROUP = "unknown"
DEFAULT_GROUP_KEY = "category"
DEFAULT_TITLE_KEY = "title"


# -----------------------------
# Array helpers
# -----------------------------

def get_array2d(embeddings: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Return embeddings as a 2D ``numpy.ndarray`` with floating dtype.

    Accepts numpy arrays, PyTorch tensors, or any object implementing ``__array__``.
    Raises :class:`ValueError` if the result is not two-dimensional.
    """
    try:  # import torch lazily if available
        import torch  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - optional dependency
        torch = None  # type: ignore[assignment]

    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(embeddings):
        arr = embeddings.detach().cpu().numpy()
    else:
        arr = np.asarray(embeddings)

    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2D (num_items, dim); got shape {arr.shape}")

    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(dtype, copy=False)

    return arr


# Backward-compatibility alias
get_embedding_array = get_array2d


def _validate_lengths(X: np.ndarray, items: Sequence[Item]) -> None:
    if len(items) != X.shape[0]:
        raise ValueError(
            "Number of items must match number of embedding rows: "
            f"len(items)={len(items)}, embeddings={X.shape[0]}"
        )


def _normalize_group(item: Item, *, group_key: str) -> str:
    val = item.get(group_key, DEFAULT_GROUP) if isinstance(item, Mapping) else DEFAULT_GROUP
    text = DEFAULT_GROUP if val is None or val == "" else str(val)
    return text


def _estimate_cluster_count(num_items: int, max_clusters: int, members_per_cluster: int) -> int:
    if max_clusters < 1:
        raise ValueError("max_clusters must be >= 1")
    if members_per_cluster < 1:
        raise ValueError("members_per_cluster must be >= 1")
    return min(max_clusters, max(1, num_items // members_per_cluster))


def _summarize_cluster(
    X: np.ndarray,
    members: np.ndarray,
    group: str,
    cluster_id: int,
    *,
    representative: str = "nearest_centroid",  # or "first"
) -> ClusterInfo:
    centroid = X[members].mean(axis=0)
    if representative == "nearest_centroid" and members.size:
        d = np.linalg.norm(X[members] - centroid, axis=1)
        rep_index = int(members[int(np.argmin(d))])
    else:
        rep_index = int(members[0]) if members.size else -1

    return {
        "group": group,
        "cluster_id": int(cluster_id),
        "members": members.tolist(),
        "centroid": centroid,
        "rep_index": rep_index,
        "size": int(len(members)),
    }


# -----------------------------
# Distance utilities
# -----------------------------

def _pairwise_distances(X: np.ndarray, *, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distances.

    Uses scikit-learn if available; otherwise supports only Euclidean distances.
    Returns an (n, n) distance matrix.
    """
    if _sk_pairwise_distances is not None:
        return _sk_pairwise_distances(X, metric=metric)

    if metric not in {"euclidean", "l2"}:
        raise RuntimeError(
            "Non-euclidean metrics require scikit-learn. Install scikit-learn or use metric='euclidean'."
        )

    # Efficient Euclidean distance via Gram matrix
    # D^2 = ||x||^2 + ||y||^2 - 2 x·y
    G = X @ X.T
    sq = np.maximum(0.0, (np.sum(X * X, axis=1)[:, None] + np.sum(X * X, axis=1)[None, :] - 2.0 * G))
    np.sqrt(sq, out=sq)
    return sq


def _distances_to_index(X: np.ndarray, idx: int, *, metric: str = "euclidean") -> np.ndarray:
    if _sk_pairwise_distances is not None:
        # shape (1, n) -> ravel to (n,)
        return _sk_pairwise_distances(X[idx : idx + 1], X, metric=metric).ravel()

    if metric not in {"euclidean", "l2"}:
        raise RuntimeError(
            "Non-euclidean metrics require scikit-learn. Install scikit-learn or use metric='euclidean'."
        )
    return np.linalg.norm(X - X[int(idx)], axis=1)


def _top_k_pairs_from_matrix(D: np.ndarray, mask: np.ndarray, k: int) -> List[Pair]:
    """Return top-k (smallest) pairs from a full distance matrix ``D`` subject to ``mask``.

    ``mask`` is a boolean matrix selecting candidate (i, j) entries (typically the
    strict upper triangle with optional additional constraints). Pairs are returned
    as ``(dist, i, j)`` in ascending distance order.
    """
    if k <= 0:
        return []

    idxs = np.nonzero(mask)
    if len(idxs[0]) == 0:
        return []

    vals = D[idxs]
    k_eff = min(k, vals.size)

    # Use argpartition for speed, then stable order by distance
    top = np.argpartition(vals, k_eff - 1)[:k_eff]
    order = top[np.argsort(vals[top])]
    return [(float(vals[t]), int(idxs[0][t]), int(idxs[1][t])) for t in order]


# -----------------------------
# Clustering
# -----------------------------

def cluster_by_group(
    X: Any,
    items: Sequence[Item],
    *,
    group_key: str = DEFAULT_GROUP_KEY,
    max_clusters: int = 8,
    members_per_cluster: int = 12,
    metric: str = "euclidean",
    linkage: str = "ward",
) -> List[ClusterInfo]:
    """Cluster embeddings *within each group* and return summaries.

    Parameters
    ----------
    X:
        2D embeddings or array-like convertible to ``np.ndarray``.
    items:
        Sequence of item mappings (only ``group_key`` is accessed).
    group_key:
        Metadata key used to group items. Default ``"category"`` for backward compatibility.
    max_clusters:
        Hard cap per-group on the number of clusters.
    members_per_cluster:
        Target items per cluster; combined with ``max_clusters`` to estimate cluster count.
    metric, linkage:
        Passed to :class:`sklearn.cluster.AgglomerativeClustering` if available.
        If scikit-learn is not installed, each group will be returned as a single cluster.
    """
    X_arr = get_array2d(X)
    _validate_lengths(X_arr, items)

    idxs_by_group: Dict[str, List[int]] = defaultdict(list)
    for idx, it in enumerate(items):
        idxs_by_group[_normalize_group(it, group_key=group_key)].append(idx)

    out: List[ClusterInfo] = []

    for group, idxs in idxs_by_group.items():
        members = np.asarray(idxs, dtype=int)
        if members.size == 0:
            continue

        n_clusters = _estimate_cluster_count(members.size, max_clusters, members_per_cluster)

        # If we can't (or shouldn't) cluster, just return the whole group as one cluster
        if AgglomerativeClustering is None or n_clusters == 1 or members.size <= 2:
            out.append(_summarize_cluster(X_arr, members, group, 0))
            continue

        # Ward requires Euclidean metric; downgrade to 'average' if incompatible
        _linkage = linkage
        _metric = metric
        if _linkage == "ward" and _metric not in {"euclidean", "l2"}:
            _linkage = "average"

        model = AgglomerativeClustering(n_clusters=int(n_clusters), linkage=_linkage, metric=_metric)
        labels = model.fit_predict(X_arr[members])

        for lab in np.unique(labels):
            mask = labels == lab
            mem = members[mask]
            if mem.size:
                out.append(_summarize_cluster(X_arr, mem, group, int(lab)))

    return out


# Backward-compatibility alias (paper-centric name)
cluster_per_category = cluster_by_group


# -----------------------------
# Nearest relationships (clusters)
# -----------------------------

def find_closest_cross_group_clusters(
    cluster_infos: Sequence[ClusterInfo],
    *,
    top_k: int = 10,
    metric: str = "euclidean",
) -> List[Pair]:
    """Return the ``top_k`` closest *cross-group* cluster centroid pairs.

    Pairs are ``(distance, i, j)`` where ``i``/``j`` are indices into ``cluster_infos``.
    """
    if len(cluster_infos) < 2 or top_k <= 0:
        return []

    centroids = np.vstack([np.asarray(ci["centroid"]) for ci in cluster_infos])
    groups = np.asarray([str(ci.get("group", DEFAULT_GROUP)) for ci in cluster_infos], dtype=object)

    D = _pairwise_distances(centroids, metric=metric)
    n = D.shape[0]
    # upper triangle mask (i < j) & different groups
    tri = np.triu(np.ones_like(D, dtype=bool), k=1)
    mask = tri & (groups[:, None] != groups[None, :])
    return _top_k_pairs_from_matrix(D, mask, top_k)


# Backward-compatibility alias (paper-centric name)
find_closest_cross_category_clusters = find_closest_cross_group_clusters


# -----------------------------
# Nearest relationships (points)
# -----------------------------

def find_closest_point_pairs(
    X: Any,
    items: Sequence[Item],
    *,
    top_k: int = 10,
    require_different_group: bool = True,
    group_key: str = DEFAULT_GROUP_KEY,
    metric: str = "euclidean",
) -> List[Pair]:
    """Return the *top_k* closest embedding pairs.

    If ``require_different_group`` is True, only cross-group pairs are considered.
    Returns pairs as ``(distance, i, j)`` with ``i < j``.
    """
    X_arr = get_array2d(X)
    _validate_lengths(X_arr, items)

    n = X_arr.shape[0]
    if n < 2 or top_k <= 0:
        return []

    groups = np.asarray([_normalize_group(it, group_key=group_key) for it in items], dtype=object)

    D = _pairwise_distances(X_arr, metric=metric)

    tri = np.triu(np.ones_like(D, dtype=bool), k=1)
    if require_different_group:
        tri &= (groups[:, None] != groups[None, :])

    return _top_k_pairs_from_matrix(D, tri, top_k)


def find_closest_point(
    X: Any,
    items: Sequence[Item],
    idx: int,
    *,
    top_k: int = 1,
    require_different_group: bool = True,
    group_key: str = DEFAULT_GROUP_KEY,
    metric: str = "euclidean",
) -> List[Pair]:
    """Return the *top_k* closest points to the point at *idx*.

    Result is a list ``[(dist, query_idx, neighbor_idx), ...]`` sorted by increasing distance.
    If ``require_different_group`` is True, neighbors from the same group are excluded.
    """
    X_arr = get_array2d(X)
    _validate_lengths(X_arr, items)

    n = X_arr.shape[0]
    if not (0 <= idx < n):
        raise IndexError(f"idx out of range: {idx}")
    if n < 2 or top_k <= 0:
        return []

    groups = np.asarray([_normalize_group(it, group_key=group_key) for it in items], dtype=object)

    d = _distances_to_index(X_arr, int(idx), metric=metric)
    d[int(idx)] = np.inf  # exclude self

    mask = np.ones(n, dtype=bool)
    if require_different_group:
        mask &= (groups != groups[int(idx)])

    valid = np.nonzero(mask)[0]
    if valid.size == 0:
        return []

    vals = d[valid]
    k = min(int(top_k), valid.size)
    top = np.argpartition(vals, k - 1)[:k]
    order = top[np.argsort(vals[top])]
    return [(float(vals[i]), int(idx), int(valid[i])) for i in order]


# -----------------------------
# Tabular summaries (optional pandas)
# -----------------------------

def cluster_pairs_to_dataframe(
    pairs: Sequence[Pair],
    cluster_infos: Sequence[ClusterInfo],
    items: Sequence[Item],
    *,
    title_key: str = DEFAULT_TITLE_KEY,
    group_key: str = DEFAULT_GROUP_KEY,
):
    """Summarize cluster pairs in a pandas DataFrame (requires pandas)."""
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pandas is required for cluster_pairs_to_dataframe") from exc

    cols = [
        "dist",
        "group_a",
        "cluster_a",
        "size_a",
        "rep_idx_a",
        "title_a",
        "group_b",
        "cluster_b",
        "size_b",
        "rep_idx_b",
        "title_b",
    ]

    if not pairs:
        return pd.DataFrame(columns=cols)

    rows = []
    for dist, i, j in pairs:
        a, b = cluster_infos[int(i)], cluster_infos[int(j)]
        ra, rb = int(a["rep_index"]), int(b["rep_index"])  # representative indices in the *items* array
        rows.append(
            {
                "dist": float(dist),
                "group_a": a.get("group"),
                "cluster_a": a.get("cluster_id"),
                "size_a": a.get("size"),
                "rep_idx_a": ra,
                "title_a": (items[ra].get(title_key) if 0 <= ra < len(items) else None),
                "group_b": b.get("group"),
                "cluster_b": b.get("cluster_id"),
                "size_b": b.get("size"),
                "rep_idx_b": rb,
                "title_b": (items[rb].get(title_key) if 0 <= rb < len(items) else None),
            }
        )

    return pd.DataFrame(rows, columns=cols).sort_values("dist").reset_index(drop=True)


# Backward-compatibility alias name
pairs_to_dataframe = cluster_pairs_to_dataframe


def point_pairs_to_dataframe(
    pairs: Sequence[Pair],
    items: Sequence[Item],
    *,
    title_key: str = DEFAULT_TITLE_KEY,
    group_key: str = DEFAULT_GROUP_KEY,
):
    """Summarize point pairs in a pandas DataFrame (requires pandas)."""
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pandas is required for point_pairs_to_dataframe") from exc

    cols = ["dist", "idx_a", "group_a", "title_a", "idx_b", "group_b", "title_b"]

    if not pairs:
        return pd.DataFrame(columns=cols)

    rows = []
    for dist, i, j in pairs:
        ia, ib = int(i), int(j)
        a, b = items[ia], items[ib]
        rows.append(
            {
                "dist": float(dist),
                "idx_a": ia,
                "group_a": a.get(group_key),
                "title_a": a.get(title_key),
                "idx_b": ib,
                "group_b": b.get(group_key),
                "title_b": b.get(title_key),
            }
        )

    return pd.DataFrame(rows, columns=cols).sort_values("dist").reset_index(drop=True)


# Backward-compatibility alias (paper-centric name)
points_pairs_to_dataframe = point_pairs_to_dataframe


__all__ = [
    # array utils
    "get_array2d",
    "get_embedding_array",
    # clustering
    "cluster_by_group",
    "cluster_per_category",
    # cluster pairs
    "find_closest_cross_group_clusters",
    "find_closest_cross_category_clusters",
    "cluster_pairs_to_dataframe",
    "pairs_to_dataframe",
    # point pairs
    "find_closest_point_pairs",
    "find_closest_point",
    "point_pairs_to_dataframe",
    "points_pairs_to_dataframe",
]