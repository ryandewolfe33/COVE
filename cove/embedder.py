import numpy as np
import scipy.sparse as sp
from scipy.special import factorial
from typing import List, Iterable, Optional
from numba import njit, prange
from numba.typed import Dict
from numba.types import int32, int64, float64
from numba_progress import ProgressBar


@njit(nogil=True)
def cdf_rows(indptr, data):
    out = np.empty_like(data, dtype="float32")
    for i in range(len(indptr) - 1):
        entries = data[indptr[i] : indptr[i + 1]]
        out[indptr[i] : indptr[i + 1]] = np.cumsum(np.divide(entries, np.sum(entries)))
    return out


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _isin_sorted(a, x):
    return a[np.searchsorted(a, x)] == x


@njit(nogil=True, fastmath=True)
def update_cooccurances(recent, step, window, n, node_ids, weights, next_coo_id):
    current_id = step % len(recent)
    n1 = recent[current_id]
    for distance in range(1, len(recent)):
        other_id = current_id - distance
        if other_id < 0:
            if step < len(recent):  # Not enough steps yet for higher distances
                return
            other_id += len(recent)  # Wrap to back of recent
        n2 = recent[other_id]
        key = n1 * n + n2
        # print(f"{n1} {n2}, Key {key}, next_coo_id {next_coo_id[0]}, {len(node_ids)}")
        node_ids[next_coo_id[0]] = key
        weights[next_coo_id[0]] = window[distance]
        next_coo_id[0] += 1
        key = n2 * n + n1
        node_ids[next_coo_id[0]] = key
        weights[next_coo_id[0]] = window[distance]
        next_coo_id[0] += 1


@njit(nogil=True, fastmath=True)
def sample_walk(
    source,
    indptr,
    indices,
    data,
    window,
    prob_0,
    prob_1,
    prob_2,
    walk_length,
    rng,
    node_ids,
    weights,
    next_coo_id
):
    # print("Sample Walk: ", source, walk_length)
    n = len(indptr) - 1
    recent = np.empty_like(window, dtype="int32")
    recent[0] = source
    # Take one step at random before p and q have effects
    neighbors = _neighbors(indptr, indices, source)
    if not neighbors.size:  # In case source is isolated
        recent[1] = source
        update_cooccurances(recent, 1, window, n, node_ids, weights, next_coo_id)
        return
    recent[1] = neighbors[
        np.searchsorted(_neighbors(indptr, data, source), rng.uniform())
    ]
    update_cooccurances(recent, 1, window, n, node_ids, weights, next_coo_id)
    for step in range(2, walk_length):
        next_id = step % len(recent)
        current_id = (step - 1) % len(recent)
        prev_id = (step - 2) % len(recent)

        neighbors = _neighbors(indptr, indices, recent[current_id])
        if not neighbors.size:
            return
        neighbors_p = _neighbors(indptr, data, recent[current_id])
        if prob_0 == prob_1 == prob_2:  # p == q == 1
            # faster version
            new_node = neighbors[np.searchsorted(neighbors_p, rng.uniform())]
        else:
            for attempt in range(1000):  # after 1000 tries default to random option
                new_node = neighbors[np.searchsorted(neighbors_p, rng.uniform())]
                r = rng.uniform()
                if attempt == 1000 - 1:
                    r = 0
                if new_node == recent[prev_id]:
                    if r < prob_0:
                        break
                elif _isin_sorted(
                    _neighbors(indptr, indices, recent[prev_id]), new_node
                ):
                    if r < prob_1:
                        break
                elif r < prob_2:
                    break
        recent[next_id] = new_node
        update_cooccurances(recent, step, window, n, node_ids, weights, next_coo_id)


@njit(nogil=True, fastmath=True)
def sort_merge_format(n, node_ids, weights):
    # Sort
    argsort = np.argsort(node_ids)
    node_ids = node_ids[argsort]
    weights = weights[argsort]
    # Merge
    current_id = -1
    current_entry = -1
    for i in range(len(node_ids)):
        if node_ids[i] == current_entry:
            weights[current_id] += weights[i]
        else:
            current_id += 1
            current_entry = node_ids[i]
            node_ids[current_id] = current_entry
            weights[current_id] = weights[i]
    # Format
    indices = node_ids[:current_id+1]
    data = weights[:current_id+1]
    indptr = np.empty(n + 1, dtype="int32")
    pair_id = 0
    for row_id in range(len(indptr)):
        indptr[row_id] = pair_id
        while pair_id < len(indices) and indices[pair_id] < n * (row_id+1):
            node_ids[pair_id] -= n * row_id
            pair_id += 1
    return indptr, indices, data
    

@njit(nogil=True, fastmath=True)
def sample_cooccurance_matrix(
    indptr,
    indices,
    data,
    window,
    prob_0,
    prob_1,
    prob_2,
    walks_per_node,
    walk_length,
    rng,
    progress_bar,
):
    n = len(indptr) - 1
    #TODO split array across threads
    #TODO save memory by merging early (how often?)
    max_num_cooccurances = n * walks_per_node * walk_length * (len(window)-1) * 2
    # Slight over-count for boundaries and any walks that get stuck
    node_ids = np.empty(max_num_cooccurances, dtype="int64")
    weights = np.zeros(max_num_cooccurances, dtype="float32")
    next_coo_id = np.zeros(1, dtype="int64")
    for source in range(n):
        for _ in range(walks_per_node):
            sample_walk(
                source,
                indptr,
                indices,
                data,
                window,
                prob_0,
                prob_1,
                prob_2,
                walk_length,
                rng,
                node_ids,
                weights,
                next_coo_id
            )
            progress_bar.update()
    indptr, indices, data = sort_merge_format(n, node_ids, weights)
    return indptr, indices, data


@njit
def row_normalize_(indptr, data):
    for i in range(len(indptr) - 1):
        row_data = data[indptr[i] : indptr[i + 1]]
        data[indptr[i] : indptr[i + 1]] = row_data / np.sum(row_data)


def flat_window(length):
    return np.full(length, 1, dtype="float32")


def pr_window(length, alpha=0.85):
    window = np.empty(length, dtype="float32")
    for i in range(length):
        window[i] = alpha**i
    return window


def hk_window(length, t=3):
    window = np.empty(length, dtype="float32")
    for i in range(length):
        window[i] = np.exp(-t) * t ** (i) / factorial(i)
    return window


class COVE:
    """
    Co-Occurance Vertex Embedding

    Create high dimensional embedding vectors based on vertex co-occurance in random walks.

    Parameters
    ----------
    p:float=1.0 - p parameter for node2vec biased random walk
    q:float=1.0 - q parameter for node2vec biased random walk
    walks_per_node:int=10 - number of walks to start at each node
    walk_length:int=40 - length of each walk
    window:(list or array)=None - co-occurrence window to slide over each random walk storing. Overrides window_type and window_length if present.
    window_type:str="flat" - window kernel, choice of "flat", "ppr", or "hk"
    window_length:int=7 - window radius (i.e. max co-occurrence distance)
    alpha:float=0.85 - alpha parameter for ppr window type
    t:float=3 - temperature parameter for hk window type
    verbose:bool=False - print info (including progress bars)
    """
    def __init__(
        self,
        p=1.0,
        q=1.0,
        walks_per_node=10,
        walk_length=40,
        window=None,
        window_type="flat",
        window_length=7,
        alpha=0.85,
        t=3,
        rng=None,
        verbose=False,
    ):
        self.p = p
        self.q = q
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        if window:
            self.window = window
        elif window_type == "flat":
            self.window = flat_window(window_length)
        elif window_type == "pr":
            self.window = pr_window(window_length, alpha)
        elif window_type == "hk":
            self.window = hk_window(window_length, t)
        else:
            raise ValueError(
                "A custom window must be passed or window_type must be one of ['flat', 'pr', 'hk']."
            )
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        self.verbose = verbose

    def fit(self, adjacency, m=1):
        # Pre-compute probability of acceptance
        max_prob = max(1 / self.p, 1, 1 / self.q)
        prob_0 = 1 / self.p / max_prob
        prob_1 = 1 / max_prob
        prob_2 = 1 / self.q / max_prob
        neighbor_cdfs = cdf_rows(adjacency.indptr, adjacency.data)
        with ProgressBar(
            total=len(adjacency.indptr) - 1 * self.walks_per_node,
            disable=not self.verbose,
        ) as progress_bar:
            indptr, indices, data = sample_cooccurance_matrix(
                adjacency.indptr,
                adjacency.indices,
                neighbor_cdfs,
                self.window,
                prob_0,
                prob_1,
                prob_2,
                self.walks_per_node,
                self.walk_length,
                self.rng,
                progress_bar,
            )
        self.cooccurance_csr_ = sp.csr_matrix((data, indices, indptr), shape=adjacency.shape)
        row_normalize_(self.cooccurance_csr_.indptr, self.cooccurance_csr_.data)
        return self

    def fit_transform(self, adjacency):
        self.fit(adjacency)
        return self.cooccurance_csr_