"""Utilities for clustering paper embeddings and finding close relationships.

The module keeps dependencies lightweight: only ``numpy`` and ``scikit-learn`` are
required at runtime, while ``pandas`` is optional for tabular summaries.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-not-found]
from sklearn.metrics import pairwise_distances  # type: ignore[import-not-found]


Paper = Dict[str, Any]
ClusterInfo = Dict[str, Any]
Pair = Tuple[float, int, int]

DEFAULT_CATEGORY = "unknown"


def get_embedding_array(embeddings: Any) -> np.ndarray:
	"""Return a 2D ``numpy.ndarray`` representation of *embeddings*.

	The helper accepts numpy arrays, PyTorch tensors, or any object implementing
	``__array__``. A :class:`ValueError` is raised if the result is not a
	two-dimensional array.
	"""

	try:  # defer PyTorch import unless it is actually needed
		import torch  # type: ignore[import-not-found]
	except ImportError:  # pragma: no cover - optional dependency
		torch = None  # type: ignore[assignment]

	if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(embeddings):
		arr = embeddings.detach().cpu().numpy()
	else:
		arr = np.asarray(embeddings)

	if arr.ndim != 2:
		raise ValueError(f"embeddings must be 2D (num_items, dim); got shape {arr.shape}")

	# ensure floating dtype for downstream linear algebra
	if not np.issubdtype(arr.dtype, np.floating):
		arr = arr.astype(np.float32, copy=False)

	return arr


def _validate_lengths(X: np.ndarray, papers: Sequence[Paper]) -> None:
	if len(papers) != X.shape[0]:
		raise ValueError(
			"Number of papers must match number of embedding rows: "
			f"len(papers)={len(papers)}, embeddings={X.shape[0]}"
		)


def _normalise_category(paper: Paper) -> str:
	return str(paper.get("category", DEFAULT_CATEGORY) or DEFAULT_CATEGORY)


def _estimate_cluster_count(num_items: int, max_clusters: int, members_per_cluster: int) -> int:
	if max_clusters < 1:
		raise ValueError("max_clusters must be >= 1")
	if members_per_cluster < 1:
		raise ValueError("members_per_cluster must be >= 1")
	return min(max_clusters, max(1, num_items // members_per_cluster))


def _summarise_cluster(X: np.ndarray, members: np.ndarray, category: str, cluster_id: int) -> ClusterInfo:
	centroid = X[members].mean(axis=0)
	dists = np.linalg.norm(X[members] - centroid, axis=1)
	rep_index = int(members[int(np.argmin(dists))])
	return {
		"category": category,
		"cluster_id": int(cluster_id),
		"members": members.tolist(),
		"centroid": centroid,
		"rep_index": rep_index,
		"size": int(len(members)),
	}


def cluster_per_category(
	X: Any,
	papers: Sequence[Paper],
	*,
	max_clusters: int = 8,
	members_per_cluster: int = 12,
) -> List[ClusterInfo]:
	"""Cluster embeddings within each category and return summary dictionaries."""

	X_arr = get_embedding_array(X)
	_validate_lengths(X_arr, papers)

	idxs_by_cat: Dict[str, List[int]] = defaultdict(list)
	for idx, paper in enumerate(papers):
		idxs_by_cat[_normalise_category(paper)].append(idx)

	cluster_infos: List[ClusterInfo] = []
	for category, idxs in idxs_by_cat.items():
		members_array = np.asarray(idxs, dtype=int)
		if members_array.size == 0:
			continue

		n_clusters = _estimate_cluster_count(len(idxs), max_clusters, members_per_cluster)

		if n_clusters == 1 or members_array.size <= 2:
			cluster_infos.append(_summarise_cluster(X_arr, members_array, category, 0))
			continue

		model = AgglomerativeClustering(n_clusters=n_clusters)
		labels = model.fit_predict(X_arr[members_array])

		for label in np.unique(labels):
			mask = labels == label
			members = members_array[mask]
			if members.size == 0:
				continue
			cluster_infos.append(_summarise_cluster(X_arr, members, category, int(label)))

	return cluster_infos


def _top_k_pairs(distances: np.ndarray, mask: np.ndarray, top_k: int) -> List[Pair]:
	if top_k <= 0:
		return []

	idxs = np.column_stack(np.nonzero(mask))
	if idxs.size == 0:
		return []

	dist_values = distances[mask]
	k = min(top_k, len(dist_values))

	# ``argpartition`` avoids a full sort when k << n_pairs
	top_indices = np.argpartition(dist_values, k - 1)[:k]
	ordered = top_indices[np.argsort(dist_values[top_indices])]

	return [
		(float(dist_values[i]), int(idxs[i, 0]), int(idxs[i, 1]))
		for i in ordered
	]


def find_closest_cross_category_clusters(
	cluster_infos: Sequence[ClusterInfo],
	top_k: int = 10,
) -> List[Pair]:
	"""Return the *top_k* closest cross-category cluster centroid pairs."""

	# Treat cluster centroids as points and reuse the point-pair logic.
	if len(cluster_infos) < 2:
		return []

	centroids = np.vstack([ci["centroid"] for ci in cluster_infos])
	# create minimal paper-like records so find_closest_point_pairs can apply
	# category filtering
	fake_papers = [{"category": ci["category"]} for ci in cluster_infos]

	return find_closest_point_pairs(centroids, fake_papers, top_k=top_k, require_different_category=True)


def pairs_to_dataframe(
	pairs: Sequence[Pair],
	cluster_infos: Sequence[ClusterInfo],
	papers: Sequence[Paper],
):
	"""Summarise ``pairs`` in a pandas DataFrame (requires pandas)."""

	try:
		import pandas as pd  # type: ignore[import-not-found]
	except ImportError as exc:  # pragma: no cover - optional dependency
		raise RuntimeError("pandas is required for pairs_to_dataframe") from exc

	columns = [
		"dist",
		"cat_a",
		"cluster_a",
		"size_a",
		"rep_idx_a",
		"title_a",
		"cat_b",
		"cluster_b",
		"size_b",
		"rep_idx_b",
		"title_b",
	]

	if not pairs:
		return pd.DataFrame(columns=columns)

	rows = []
	for dist, i, j in pairs:
		ci = cluster_infos[i]
		cj = cluster_infos[j]
		rows.append(
			{
				"dist": dist,
				"cat_a": ci["category"],
				"cluster_a": ci["cluster_id"],
				"size_a": ci["size"],
				"rep_idx_a": ci["rep_index"],
				"title_a": papers[ci["rep_index"]]["title"],
				"cat_b": cj["category"],
				"cluster_b": cj["cluster_id"],
				"size_b": cj["size"],
				"rep_idx_b": cj["rep_index"],
				"title_b": papers[cj["rep_index"]]["title"],
			}
		)

	return pd.DataFrame(rows, columns=columns).sort_values("dist").reset_index(drop=True)


def find_closest_point_pairs(
	X: Any,
	papers: Sequence[Paper],
	*,
	top_k: int = 10,
	require_different_category: bool = True,
) -> List[Pair]:
	"""Return the *top_k* closest embedding pairs, optionally across categories only."""

	X_arr = get_embedding_array(X)
	_validate_lengths(X_arr, papers)

	n = X_arr.shape[0]
	if n < 2 or top_k <= 0:
		return []

	# aggregate neighbours by querying each point via find_closest_point
	seen = set()
	pairs: List[Pair] = []

	for i in range(n):
		neighs = find_closest_point(X_arr, papers, i, top_k=top_k, require_different_category=require_different_category)
		for dist, q, j in neighs:
			# normalize ordering so each unordered pair appears once (a < b)
			a, b = (int(q), int(j))
			if a == b:
				continue
			if a > b:
				a, b = b, a
			if (a, b) in seen:
				continue
			seen.add((a, b))
			pairs.append((dist, a, b))

	if not pairs:
		return []

	# sort and return top_k
	pairs.sort(key=lambda t: t[0])
	return pairs[:top_k]


def find_closest_point(
	X: Any,
	papers: Sequence[Paper],
	idx: int,
	*,
	top_k: int = 1,
	require_different_category: bool = True,
) -> List[Pair]:
	"""Return the *top_k* closest points to the point at *idx*.

	The result is a list of tuples ``(dist, query_idx, neighbor_idx)`` sorted
	by increasing distance. If *require_different_category* is True the
	neighbours will be filtered to omit papers with the same category as the
	query point.
	"""

	X_arr = get_embedding_array(X)
	_validate_lengths(X_arr, papers)

	n = X_arr.shape[0]
	if n < 2 or top_k <= 0:
		return []

	if not (0 <= idx < n):
		raise IndexError(f"idx out of range: {idx}")

	categories = np.asarray([_normalise_category(p) for p in papers], dtype=object)

	# distances from query point to all points
	dists = np.linalg.norm(X_arr - X_arr[int(idx)], axis=1)
	# exclude the point itself
	dists[int(idx)] = np.inf

	mask = np.ones(n, dtype=bool)
	if require_different_category:
		mask &= categories != categories[int(idx)]

	if not np.any(mask):
		return []

	valid_idxs = np.nonzero(mask)[0]
	vals = dists[valid_idxs]

	k = min(top_k, len(vals))
	top_inds = np.argpartition(vals, k - 1)[:k]
	ordered = top_inds[np.argsort(vals[top_inds])]

	return [
		(float(vals[i]), int(idx), int(valid_idxs[i]))
		for i in ordered
	]


def points_pairs_to_dataframe(pairs: Sequence[Pair], papers: Sequence[Paper]):
	"""Summarise point pairs in a pandas DataFrame (requires pandas)."""

	try:
		import pandas as pd  # type: ignore[import-not-found]
	except ImportError as exc:  # pragma: no cover - optional dependency
		raise RuntimeError("pandas is required for points_pairs_to_dataframe") from exc

	columns = ["dist", "idx_a", "cat_a", "title_a", "idx_b", "cat_b", "title_b"]

	if not pairs:
		return pd.DataFrame(columns=columns)

	rows = []
	for dist, i, j in pairs:
		rows.append(
			{
				"dist": dist,
				"idx_a": i,
				"cat_a": papers[i].get("category"),
				"title_a": papers[i].get("title"),
				"idx_b": j,
				"cat_b": papers[j].get("category"),
				"title_b": papers[j].get("title"),
			}
		)

	return pd.DataFrame(rows, columns=columns).sort_values("dist").reset_index(drop=True)


__all__ = [
	"get_embedding_array",
	"cluster_per_category",
	"find_closest_cross_category_clusters",
	"pairs_to_dataframe",
	"find_closest_point_pairs",
	"points_pairs_to_dataframe",
    "find_closest_point",
]


def _demo() -> None:  # pragma: no cover - convenience smoke test
	rng = np.random.default_rng(0)
	num_items, dim = 60, 16
	X = rng.normal(size=(num_items, dim))
	cats = ["hep-ph", "hep-th", "hep-ex"]
	papers = [
		{"title": f"paper_{i}", "category": cats[i % len(cats)]}
		for i in range(num_items)
	]

	clusters = cluster_per_category(X, papers, max_clusters=4, members_per_cluster=6)
	cluster_pairs = find_closest_cross_category_clusters(clusters, top_k=5)
	point_pairs = find_closest_point_pairs(X, papers, top_k=5)

	try:
		print("Cluster pairs:\n", pairs_to_dataframe(cluster_pairs, clusters, papers))
		print("\nPoint pairs:\n", points_pairs_to_dataframe(point_pairs, papers))
	except RuntimeError:
		print("Cluster pairs:", cluster_pairs)
		print("Point pairs:", point_pairs)


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
	_demo()
