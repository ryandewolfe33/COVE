import scipy.sparse as sp
from umap import UMAP
from cove import COVE
from cove.dimension_reduction import UMAPLE
from utils import *


graphs = [
    # "football",
    # "primary1",
    # "primary2",
    "eu-core",
    "eurosis",
    # "cora_small",
    # "airport",
    # "blogcatalog",
    # "cora",
    # "as"
]
RUNS=10

rng = np.random.default_rng(seed=42)

n_walks=10
walk_len = 40
window_size=7

cove = COVE(walks_per_node=n_walks, walk_length=walk_len, window_length=window_size)
umap2 = UMAP(n_components=2, metric="hellinger")
umaple2 = UMAPLE(dimension=2, metric="hellinger")
umap16 = UMAP(n_components=16, metric="hellinger")
umaple16 = UMAPLE(dimension=16, metric="hellinger")
umap128 = UMAP(n_components=128, metric="hellinger")
umaple128 = UMAPLE(dimension=128, metric="hellinger")

for graph in graphs:
    adj, lab = load_graph(graph)
    for run in range(RUNS):
        print(graph, run)

        high = cove.fit_transform(adj, 2)
        sp.save_npz(f"embeddings/{graph}_cove_{run}.npz", high)
        #high = sp.load_npz(f"embeddings/{graph}_cove_{run}.npz")

        low = umap2.fit_transform(high)
        while np.any(np.isnan(low)):
            low = umap2.fit_transform(high)
        np.save(f"embeddings/{graph}_coveumap_d2_{run}.npy", low)

        low = umaple2.fit_transform(high, adj)
        while np.any(np.isnan(low)):
            low = umaple2.fit_transform(high, adj)
        np.save(f"embeddings/{graph}_coveumaple_d2_{run}.npy", low)

        low = umap16.fit_transform(high)
        while np.any(np.isnan(low)):
            low = umap16.fit_transform(high)
        np.save(f"embeddings/{graph}_coveumap_d16_{run}.npy", low)

        low = umaple16.fit_transform(high, adj)
        while np.any(np.isnan(low)):
            low = umaple16.fit_transform(high, adj)
        np.save(f"embeddings/{graph}_coveumaple_d16_{run}.npy", low)

        low = umap128.fit_transform(high)
        while np.any(np.isnan(low)):
            low = umap16.fit_transform(high)
        np.save(f"embeddings/{graph}_coveumap_d128_{run}.npy", low)

        low = umaple128.fit_transform(high, adj)
        while np.any(np.isnan(low)):
            low = umaple16.fit_transform(high, adj)
        np.save(f"embeddings/{graph}_coveumaple_d128_{run}.npy", low)

        low = node2vec(adj, 2, num_walks=n_walks, walk_length=walk_len, window_size=window_size)
        np.save(f"embeddings/{graph}_n2v_d2_{run}.npy", low)

        low = node2vec(adj, 16, num_walks=n_walks, walk_length=walk_len, window_size=window_size)
        np.save(f"embeddings/{graph}_n2v_d16_{run}.npy", low)

        high = node2vec(adj, 128, num_walks=n_walks, walk_length=walk_len, window_size=window_size)
        np.save(f"embeddings/{graph}_n2v_d128_{run}.npy", high)

        low = UMAP(n_components=2, metric="cosine").fit_transform(high)
        while np.any(np.isnan(low)):
            low = MAP(n_components=2, metric="cosine").fit_transform(high)
        np.save(f"embeddings/{graph}_n2vumap_d2_{run}.npy", low)

        low = UMAP(n_components=16, metric="cosine").fit_transform(high)
        while np.any(np.isnan(low)):
            low = UMAP(n_components=16, metric="cosine").fit_transform(high)
        np.save(f"embeddings/{graph}_n2vumap_d16_{run}.npy", low)

        low = UMAP(n_components=128, metric="cosine").fit_transform(high)
        while np.any(np.isnan(low)):
            low = UMAP(n_components=128, metric="cosine").fit_transform(high)
