import umap
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD


def spectral_layout(
    data,
    graph,
    dim,
    rng,
    init="random",
    method=None,
    tol=0.0,
    maxiter=0,
):
    """General implementation of the spectral embedding of the graph, derived as
    a subset of the eigenvectors of the normalized Laplacian of the graph. The numerical
    method for computing the eigendecomposition is chosen through heuristics.

    This is modified from https://github.com/lmcinnes/umap/blob/master/umap/spectral.py
    to remove multi-component layouts and any reference to high dimensional vectors.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data

    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.

    dim: int
        The dimension of the space into which to embed.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    init: string, either "random" or "tsvd"
        Indicates to initialize the eigensolver. Use "random" (the default) to
        use uniformly distributed random initialization; use "tsvd" to warm-start the
        eigensolver with singular vectors of the Laplacian associated to the largest
        singular values. This latter option also forces usage of the LOBPCG eigensolver;
        with the former, ARPACK's solver ``eigsh`` will be used for smaller Laplacians.

    method: string -- either "eigsh" or "lobpcg" -- or None
        Name of the eigenvalue computation method to use to compute the spectral
        embedding. If left to None (or empty string), as by default, the method is
        chosen from the number of vectors in play: larger vector collections are
        handled with lobpcg, smaller collections with eigsh. Method names correspond
        to SciPy routines in scipy.sparse.linalg.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    n_samples = graph.shape[0]

    # This is the part we modified, since there are no high vectors we cannot
    # do a multi-component layout.
    # n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    # if n_components > 1:
    #     return multi_component_layout(
    #         data,
    #         graph,
    #         n_components,
    #         labels,
    #         dim,
    #         random_state,
    #         metric=metric,
    #         metric_kwds=metric_kwds,
    #     )

    sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    # standard Laplacian
    # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    # L = D - graph
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(1.0 / sqrt_deg, 0, graph.shape[0], graph.shape[0])
    L = I - D * graph * D
    if not scipy.sparse.issparse(L):
        L = np.asarray(L)

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    gen = (
        random_state
        if isinstance(random_state, (np.random.Generator, np.random.RandomState))
        else np.random.default_rng(seed=random_state)
    )
    if not method:
        method = "eigsh" if L.shape[0] < 2000000 else "lobpcg"

    try:
        if init == "random":
            X = gen.normal(size=(L.shape[0], k))
        elif init == "tsvd":
            X = TruncatedSVD(
                n_components=k,
                random_state=random_state,
                # algorithm="arpack"
            ).fit_transform(L)
        else:
            raise ValueError(
                "The init parameter must be either 'random' or 'tsvd': "
                f"{init} is invalid."
            )
        # For such a normalized Laplacian, the first eigenvector is always
        # proportional to sqrt(degrees). We thus replace the first t-SVD guess
        # with the exact value.
        X[:, 0] = sqrt_deg / np.linalg.norm(sqrt_deg)

        if method == "eigsh":
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=tol or 1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=maxiter or graph.shape[0] * 5,
            )
        elif method == "lobpcg":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    category=UserWarning,
                    message=r"(?ms).*not reaching the requested tolerance",
                    action="error",
                )
                eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                    L,
                    np.asarray(X),
                    largest=False,
                    tol=tol or 1e-4,
                    maxiter=maxiter or 5 * graph.shape[0],
                )
        else:
            raise ValueError("Method should either be None, 'eigsh' or 'lobpcg'")

        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except (scipy.sparse.linalg.ArpackError, UserWarning):
        warn(
            "Spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        return gen.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))


class LaplacianEigenmap:
    def __init__(self, dimensions, random_state=None):
        self.dimensions = dimensions
        self.random_state = random_state

    def fit(self, adjacency):
        if self.random_state is None:
            self.random_state = np.random.RandomState()
        self.layout_ = spectral_layout(
            adjacency,
            self.dimensions,
            self.random_state,
        )
        return self

    def fit_transform(self, adjacency):
        self.fit(adjacency)
        return self.layout_


class UMAPLE:
    """
    A simple wrapper for running UMAP using a spectral embdding of the graph for initialization.
    """

    def __init__(self, dimension, metric="hellinger", random_state=None, **umap_kwargs):
        self.dimension = dimension
        self.metric = metric
        self.random_state = random_state
        self.umap_kwargs = umap_kwargs

    def fit(self, embedding, adjacency):
        le_init = umap.spectral.spectral_layout(
            embedding,
            adjacency,
            self.dimension,
            (
                self.random_state
                if self.random_state is not None
                else np.random.RandomState()
            ),
            metric=self.metric,
        )
        self.mapper_ = umap.UMAP(
            n_components=self.dimension,
            metric=self.metric,
            init=le_init,
            random_state=self.random_state,
            **self.umap_kwargs,
        )
        self.X_ = self.mapper_.fit_transform(embedding)
        return self

    def fit_transform(self, embedding, adjacency):
        self.fit(embedding, adjacency)
        return self.X_
