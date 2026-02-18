import numpy as np
import scipy.sparse as sp
import abcd_graph

import os
import subprocess


def node2vec(
    adjacency,
    dimensions,
    num_walks=40,
    walk_length=80,
    window_size=10,
    random_state=None
    ):
    """
    Weight a graph using node2vec embedding distances.
    This method requires the existance of a conda environment named `n2v` with pecanpy installed.

    Inputs:
    adjacency:csr_matrix - The adjacency matrix of the graph.
    dimensions:int       - The number of dimensions used for node2vec
    num_walks:int=40     - The number of random walks used per node

    Returns:
    similarity:csr_matrix - The adjacency matrix of the similarity graph (upper triangluar, implicilty symmetric)
    """
    if random_state is None:
        random_state = np.random.RandomState()
    seed = random_state.randint(2**32)
    unweighted = all(adjacency.data == 1)
    sp.save_npz(f"adjacency_{seed}.npz", adjacency)
    output_file = f"emb_{seed}.npz"
    pecanpy_cmd = f"mamba run -n n2v pecanpy --input adjacency_{seed}.npz --output {output_file} --dimensions {dimensions} --num-walks {num_walks} --walk-length {walk_length} --window-size {window_size} --random_state {seed}"
    if unweighted:
        pecanpy_cmd += " --mode FirstOrderUnweighted"
    else:
        pecanpy_cmd += "  --mode PreCompFirstOrder --weighted"
    os.system(pecanpy_cmd+' >/dev/null 2>&1')
    #print("Load similarity graph")
    output = np.load(output_file)
    id_argsort = np.argsort(output["IDs"].astype("int"))
    emb = output["data"][id_argsort]
    os.system(f"rm adjacency_{seed}.npz")
    os.system(f"rm {output_file}")
    return emb


def abcd(n, xi, nout=0):
    params = abcd_graph.ABCDParams(
        vcount=n,
        xi=xi,
        num_outliers=nout,
        min_degree=3,
        max_degree=int(np.sqrt(n)),
        min_community_size=15,
        max_community_size=10*int(np.sqrt(n))
    )
    graph = abcd_graph.ABCDGraph(params).build()
    labels = np.empty(n, dtype="int32")
    for i, com in enumerate(graph.communities):
        labels[com.vertices] = i
    if nout > 0:
        labels[labels == np.max(labels)] = -1
    adjacency = graph.exporter.to_sparse_adjacency_matrix()
    return adjacency, labels


def load_graph(name, force_sparse=False):
    adjacency = sp.load_npz(f"data/{name}_adjacency.npz")
    try:
        labels = np.load(f"data/{name}_labels.npy")
        if force_sparse:
            labels_sparse = sp.lil_array((np.max(labels)+1, adjacency.shape[0]), dtype="bool")
            for l in range(np.max(labels)+1):
                labels_sparse[l, labels==l] = True
            labels = labels_sparse.tocsr()
    except FileNotFoundError as e:
        try:
            labels = sp.load_npz(f"data/{name}_labels.npz")
        except FileNotFoundError as e:
            raise ValueError("Can't find labels file as npy or npz :(")
    return adjacency, labels


def JS(adjacency, embedding, julia, cge, labels=None, random_state=None):
    if labels is None:
        labels = ECG(random_state=random_state).fit_predict(adjacency)
    if random_state is None:
        random_state = np.random.RandomState()
    # Write edgelist file
    edgelist = sp.triu(adjacency).nonzero()
    edgelist_file = f"edgelist.dat"
    with open(edgelist_file, "w") as f:
        for i,j in zip(*edgelist):
            line = f"{i+1}\t{j+1}" # Julia uses 1 based indexing
            print(line, file=f)
    np.savetxt(f"emb.dat", embedding, fmt="%.10f")
    np.savetxt("labels.dat", labels, fmt="%d")
    cmd = julia+' '+cge+' -g '+edgelist_file+' -c labels.dat -e emb.dat --seed '+str(random_state.randint(2**32))
    if adjacency.shape[0] > 3000:
        cmd += ' -l 1000'
    cmd += ' 2>_stderr'
    s = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    x = s.stdout.decode().split(',')
    # Delete temp files
    os.system(f"rm edgelist.dat")
    os.system(f"rm emb.dat")
    os.system(f"rm labels.dat")

    if len(x[0]) == 0:
        print('Error running CLI command:\n', cmd )
        print('Make sure Julia and CGE are correctly installed')
        return -1

    return (float(x[1]),float(x[5]))