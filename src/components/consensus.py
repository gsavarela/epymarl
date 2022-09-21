"""Consensus is the local estimation of a global state.

Given a network of processes where each node has an initial scalar value,
we consider the problem of computing their average asymptotically using a
distributed, linear iterative algorithm. At each iteration, each node
replaces its own value with a weighted average of its previous value and
the values of its neighbors. We introduce the Metropolis weights, a simple
choice for the averaging weights used in each step. We show that with these
weights, the values at every node converge to the average, provided the
infinitely occurring communication graphs are jointly connected. [3]

Reference
---------
..[1] Lin Xiao and Stephen Boyd, 2004,
  "Fast Linear Iterations For Distributed Averaging".
..[2] https://mathworld.wolfram.com/LaplacianMatrix.html
..[3] Lin Xiao, Stephen Boyd, Sanjay Lall, 2006,
  "Distributed Average Consensus with Time-Varying Metropolis Weights"

"""
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

from typing import List

Array = np.ndarray


def consensus_matrices(
    n_nodes: int,
    cm_n_edges: int = 0,
    cm_type: str = "metropolis",
) -> List[Array]:
    """Generates a list of consensus matrices

    Parameters
    ----------
    * n_nodes: int
        A two dimension array representing an adjacency matrix.

    * cm_n_edges: int
        Number of edges

    * cm_type: str = 'metropolis'
        A string with the algorithm for consensus.

    Returns
    -------
    * cwms: List[Array]
        A list containing consensus weights matrices
    """
    if n_nodes == 1:
        raise ValueError("%s invalid" % str(n_nodes))
    if cm_n_edges < 0 or cm_n_edges > (n_nodes * (n_nodes - 1) // 2):
        raise ValueError("max_edges: %s invalid" % str(cm_n_edges))
    if cm_type not in ("metropolis", "normalized_laplacian", "laplacian"):
        raise ValueError("%s invalid" % cm_type)
    else:
        fn = eval("%s_weights_matrix" % cm_type)

    cwms = []
    ams = adjacency_matrices(n_nodes, cm_n_edges)
    cwms += map(fn, ams)
    return cwms

# def consensus_cliques(
#     n_nodes: int,
#     cm_type: str = "metropolis",
# ) -> List[Array]:
#     """List of consensus matrices that originate from cliques
#
#     Parameters
#     ----------
#     * n_nodes: int
#         A two dimension array representing an adjacency matrix.
#
#     * cm_type: str = 'metropolis'
#         A string with the algorithm for consensus.
#
#     Returns
#     -------
#     * cwms: List[Array]
#         A list containing consensus weights matrices
#     """
#     if n_nodes == 1:
#         raise ValueError("%s invalid" % str(n_nodes))
#     if cm_n_edges < 0 or cm_n_edges > (n_nodes * (n_nodes - 1) // 2):
#         raise ValueError("max_edges: %s invalid" % str(cm_n_edges))
#     if cm_type not in ("metropolis", "normalized_laplacian", "laplacian"):
#         raise ValueError("%s invalid" % cm_type)
#     else:
#         fn = eval("%s_weights_matrix" % cm_type)
#
#     cwms = []
#     ams = adjacency_cliques(n_nodes)
#     cwms += map(fn, ams)
#     return cwms
#

def metropolis_weights_matrix(am: Array) -> Array:
    """Consensus matrix[3] from an adjacency matrix

    Parameters
    ----------
    * adjacency: Array
        A two dimension array representing an adjacency matrix.

    Returns
    -------
    * mwm: Array
        A two dimension array the metropolis weights matrix
    """
    adj = np.array(am)
    degree = np.sum(adj, axis=1)
    mwm = np.zeros_like(am, dtype=float)
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if am[i, j] > 0:
                mwm[i, j] = 1 / (1 + max(degree[i], degree[j]))
                mwm[j, i] = mwm[i, j]  # symmetrical
        mwm[i, i] = 1 - (mwm[i, :].sum())
    return mwm


def normalized_laplacian_weights_matrix(am: Array) -> Array:
    """Consensus matrix[2] from an adjacency matrix

    Parameters
    ----------
    * am: Array
        A two dimension array representing an adjacency matrix.

    Returns
    -------
    * nlwm: Array
        A two dimension array the normalized laplacian weights matrix
    """
    adj = np.array(am)
    degree = np.sum(adj, axis=1)
    nlwm = np.diag(degree) - am
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if am[i, j] > 0:
                nlwm[i, j] = -(1 / np.sqrt(degree[i] * degree[j]))
                nlwm[j, i] = nlwm[i, j]  # symmetrical
        nlwm[i, i] = 1

    return nlwm


def laplacian_weights_matrix(am: Array, fast: bool = True) -> Array:
    """Consensus matrix[1] from an adjacency matrix

    Parameters
    ----------
    * adjacency: Array
        A two dimension array representing an adjacency matrix.

    Returns
    -------
    * lwm: Array
        A two dimension array the laplacian weights matrix
    """
    eye = np.eye(*am.shape)
    degree = np.sum(am, axis=1)
    laplacian = np.diag(degree) - am

    # fast computation -- two largest
    if fast:
        alpha = 1 / sum(sorted(degree, reverse=True)[:2])
    else:
        eig, _ = np.linalg.eig(laplacian)
        alpha = 2 / (eig[0] + eig[-2])

    lwm = eye - alpha * laplacian
    return np.array(lwm)


def random_adjacency_matrix(n_nodes: int, n_edges: int) -> Array:
    """Randomly produce an adjacency matrix

                             A[i, i] = 0
                            /
     A is adjacency matrix < --->  A[i, j] = 1 iff i, j are neighbors
                            \
                            A[i, j] = 0 otherwise

    Parameters
    ----------
    n_nodes: int
    n_edges: int

    Returns
    -------
    * ram: Array
        A two dimension array representing an adjacency matrix.
    """
    if n_edges == 0:
        return np.zeros((n_nodes, n_nodes))

    full_edge_list = [(i, j) for i in range(n_nodes - 1) for j in range(i + 1, n_nodes)]

    n_choices = min(len(full_edge_list), n_edges)
    edge_ids = np.random.choice(len(full_edge_list), replace=False, size=n_choices)

    edge_list = [full_edge_list[i] for i in sorted(edge_ids)]

    data = (np.ones(len(edge_list), dtype=int), zip(*edge_list))
    ram = csr_matrix(data, dtype=int, shape=(n_nodes, n_nodes)).toarray()
    ram = ram + ram.T
    return ram


def adjacency_matrices(n_nodes: int, n_edges: int) -> Array:
    """Produce all adjacency matrices with n_edges

                             A[i, i] = 0
                            /
     A is adjacency matrix < --->  A[i, j] = 1 iff i, j are neighbors
                            \
                            A[i, j] = 0 otherwise

    Parameters
    ----------
    n_nodes: int
    n_edges: int

    Returns
    -------
    * ams: Array
        A list of two dimension array representing an adjacency matrix.
    """
    if n_edges == 0:
        return [np.zeros((n_nodes, n_nodes))]

    full_edge_list = [(i, j) for i in range(n_nodes - 1) for j in range(i + 1, n_nodes)]
    ones = np.ones(n_edges, dtype=int)

    ams = []
    for edge_set in combinations(full_edge_list, n_edges):
        am = csr_matrix(
            (ones, zip(*edge_set)), dtype=int, shape=(n_nodes, n_nodes)
        ).toarray()
        ams.append(am + am.T)

    return ams

# def adjacency_cliques(n_nodes: int) -> Array:
#     """Produce cliques: Adjacency matrix where there is a path to 
#     every agent.
#
#                              A[i, i] = 0
#                             /
#      A is adjacency matrix < --->  A[i, j] = 1 iff i, j are neighbors
#                             \
#                             A[i, j] = 0 otherwise
#
#     Parameters
#     ----------
#     n_nodes: int
#
#     Returns
#     -------
#     * ams: Array
#         A list of two dimension array representing an adjacency matrix.
#     """
#     node_list = [*range(n_nodes)]
#     full_edge_list = [(i, j) for i in range(n_nodes - 1) for j in range(i + 1, n_nodes)]
#     edge_set = set()
#
#     
#     ones = np.ones(n_edges, dtype=int)
#
#     ams = []
#     for edge_set in combinations(full_edge_list, n_edges):
#         am = csr_matrix(
#             (ones, zip(*edge_set)), dtype=int, shape=(n_nodes, n_nodes)
#         ).toarray()
#         ams.append(am + am.T)
#
#     return ams


def main(n_nodes: int = 5, target: int = 3):
    """Performs distributed averaging on a simple graph.

    Parameters
    ----------
    n_nodes: int = 5
        The side of the square matrix
    target: int = 3
        The integer with the average the nodes should agree on.
    """

    n_edges = 2 * (n_nodes - 1)

    adjacency = random_adjacency_matrix(n_nodes, n_edges)

    print("ADJACENCY:")
    print(adjacency)

    # generate an array with average == target
    x = np.random.randint(low=0, high=2 * target, size=n_nodes)
    res = target - np.mean(x)
    x = x.astype(np.float32) + res

    print("DATA:")
    print(dict(enumerate(x.tolist())))

    print("Laplacian:")
    C = laplacian_weights_matrix(adjacency, fast=True)
    print(C)

    log = [x]
    n_steps = 99
    for _ in range(n_steps):
        x = C @ x
        log.append(x)

    X = np.linspace(1, n_steps + 1, n_steps + 1)
    Y = np.stack(log)

    # Beware that the graph must be fully connected
    plt.axhline(y=target, color=(0.2, 1.0, 0.2), linestyle="-")
    plt.suptitle("Consensus Iterations (%s, %s)" % (n_nodes, target))
    plt.ylabel("Data")
    plt.xlabel("Time")
    plt.plot(X, Y)
    plt.show()


if __name__ == "__main__":
    main()
