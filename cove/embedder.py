import numpy as np
import scipy.sparse as sp
from scipy.special import factorial
from typing import List, Iterable, Optional
from numba import njit, prange
from numba.typed import Dict
from numba.types import int32, int64, float64
from numba_progress import ProgressBar


@njit(nogil=True)
def merge_int32s(x: int32, y: int32):
    """
    Store two int32s in an int64. The first 32 bits are for x, the second 32 for y.
    """
    return (np.int64(x) << 32) | np.int64(y)


@njit(nogil=True)
def split_int32s(x: int64):
    """
    Split an int64 into two int32s. Returns a two element tuple of the first and second 32 bits.
    Opposite of the function merge_int32s
    """
    first = np.int32(x >> 32)
    second = np.int32(x & 0xFFFFFFFF)
    return first, second


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
def update_cooccurances(recent, step, window, cooccurances):
    # print(recent, step, window)
    current_id = step % len(recent)
    for distance in range(1, len(recent)):
        other_id = current_id - distance
        if other_id < 0:
            if step < len(recent):  # Not enough steps yet for higher distances
                return
            other_id += len(recent)  # Wrap to back of recent
        # print(distance, recent[current_id], recent[other_id], window[distance])
        key = merge_int32s(recent[current_id], recent[other_id])
        if key not in cooccurances:
            cooccurances[key] = 0.0
        cooccurances[key] = cooccurances[key] + window[distance]
        # Co-occurance is symmetric
        key = merge_int32s(recent[other_id], recent[current_id])
        if key not in cooccurances:
            cooccurances[key] = 0.0
        cooccurances[key] = cooccurances[key] + window[distance]
    return


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
    cooccurances,
):
    # print("Sample Walk: ", source, walk_length)
    recent = np.empty_like(window, dtype="int32")
    recent[0] = source
    # Take one step at random before p and q have effects
    neighbors = _neighbors(indptr, indices, source)
    if not neighbors.size:  # In case source is isolated
        return
    recent[1] = neighbors[
        np.searchsorted(_neighbors(indptr, data, source), rng.uniform())
    ]
    update_cooccurances(recent, 1, window, cooccurances)
    for step in range(2, walk_length):
        # print("Step: ", step)
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
        update_cooccurances(recent, step, window, cooccurances)
    return


@njit
def numba_dict_to_csr(numba_dict):
    row = np.empty(len(numba_dict), dtype="int32")
    col = np.empty(len(numba_dict), dtype="int32")
    data = np.empty(len(numba_dict), dtype="float64")
    i = 0
    for key, w in numba_dict.items():
        node1, node2 = split_int32s(key)
        row[i] = node1
        col[i] = node2
        data[i] = w
        i += 1
    return row, col, data


# TODO make sure there are no isolated nodes
@njit(nogil=True, fastmath=True)
def sample_cooccurance_matrix_(
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
    cooccurances = Dict.empty(
        key_type=int64, value_type=float64, n_keys=(len(indptr) - 1) * 100
    )
    for source in prange(len(indptr) - 1):
        for _ in prange(walks_per_node):
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
                cooccurances,
            )
            progress_bar.update()
    row, col, data = numba_dict_to_csr(cooccurances)
    return row, col, data


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
    window : the sliding window over each random walk storing weights for each co-occurance distance
    """

    def __init__(
        self,
        p=1.0,
        q=1.0,
        walks_per_node=20,
        walk_length=80,
        window=None,
        window_type="flat",
        window_length=5,
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

    def fit(self, adjacency):
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
            row, col, data = sample_cooccurance_matrix_(
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
        cooccurance_csr = sp.csr_matrix((data, (row, col)), shape=adjacency.shape)
        row_normalize_(cooccurance_csr.indptr, cooccurance_csr.data)
        self.cooccurance_csr_ = cooccurance_csr
        return self

    def fit_transform(self, adjacency):
        self.fit(adjacency)
        return self.cooccurance_csr_