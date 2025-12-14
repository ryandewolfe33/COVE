# Co-Occurance Vetex Embedding (COVE)

This package is provides an implementation of co-occurance vectorization using random walks on graphs, similar to node2vec.

# It is currently under construction.

## Getting Started

Currently this package can be installed by cloning the repository and the using pip locally.
```{shell}
git clone https://github.com/ryandewolfe33/COVE.git
cd COVE
pip install .
```

## How it works

```{python}
from cove import COVE, UMAPLE
import sknetwork as sn
import matplotlib.pyplot as plt

# Get the adjacency matrix of a graph
block_sizes = [100 for _ in range(10)]
adjacency = sn.data.models.block_model(block_sizes, 0.15, 0.025, seed=42)

# Instantiate a COVE vectorizer and fit an embedding.
vectorizer = COVE(window_length=6, walks_per_node =20, walk_length=20)
embedding = vectorizer.fit_transform(adjacency)

# Reduce dimension to 2d for visualization with UMAPLE (UMAP with laplacian eigenvector initialization).
mapper =  UMAPLE(dimension=2, metric="hellinger")
data_map = mapper.fit_transform(embedding, adjacency)

# Plot the clusters (colored by block)
labels = [i for i,block_size in enumerate(block_sizes) for _ in range(block_size)]
plt.scatter(data_map[:, 0], data_map[:, 1], c=labels)
```

We

There are also more complicated example available in the experiments folder, but be warned they are not polished.