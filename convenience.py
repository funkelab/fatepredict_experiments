"""
Given a Linajea Candidate Graph, for each time point we:
    - Get the merge graph with `get_merge_graph(candidate_graph, t)`
    - Get the conflict set for this graph with `get_conflict_sets(merge_graph)`
Combine all of the conflict sets and add a constraint with
`set_conflict_sets(conflict_sets)`
"""
import networkx as nx
import pylp


def get_merge_graph_from_array(merge_tree, scores):
    """Get graph representation from array representation of the merge tree

    merge_tree: ndarray
        (n, 3) array where n is the number of merges and each item holds
        indices (u, v, w): u and v merge to form w.
        We assume (and therefore do not check) that the indices are unique and
        there are no loops.
    scores: ndarray
        (n,) array defining the merge score for each of the merges in the merge
        tree

    Example
    -------
    >>> import numpy as np
    >>> from convenience import get_merge_graph
    >>> merge_tree = np.array([[1, 2, 3], [3, 4, 5]])
    >>> scores = np.array([0.2, 0.01])
    >>> g = get_merge_graph(merge_tree, scores)
    >>> g.order()
    5
    """
    G = nx.DiGraph()

    for (u, v, w), score in zip(merge_tree, scores):
        G.add_node(w, score=score)
        G.add_edge(u, w)
        G.add_edge(v, w)
    return G


def get_merge_graph(candidate_graph, t):
    """Get merge graph from Linajea CandidateGraph

    The nodes can be chosen for that time point.
    The edges can be obtained from the "parent" attribute of the node.

    Parameters
    ----------
    candidate_graph: linajea.CandidateGraph
        The Candidate graph containing all the nodes of interest
    t: int
        The time to consider
    """
    g = nx.DiGraph()
    nodes = [nid for nid in candidate_graph.nodes()
             if candidate_graph[nid]['t'] == t]
    # Iteratively add nodes
    for nid in nodes:
        node = candidate_graph[nid]
        if nid not in g:
            g.add_node(node)
        if node['parent'] != node['id']:  # How we determine a root
            g.add_edge(nid, node['parent'])
    return g


def get_conflict_sets(graph):
    """Get conflict sets from merge tree.

    Nodes are in conflict if they are along the same path.

    Example
    >>> import numpy as np
    >>> from convenience import get_conflict_set
    """
    # Get all leaves - no incoming edges
    leaves = [x for x in graph.nodes() if graph.in_degree(x) == 0]
    # Get all roots - no outgoing edges
    roots = [x for x in graph.nodes() if graph.out_degree(x) == 0]
    # Get all paths from a leaf to a root
    conflict_sets = []
    for root in roots:
        for leaf in leaves:
            s = nx.all_simple_paths(graph, source=leaf, target=root)
            ts = tuple(s)
            if len(ts) > 0:
                conflict_sets.append(ts)
    return conflict_sets


def set_conflict_sets(graph, indicators, conflict_sets):
    """Certain sets of nodes are mutually exclusive, e.g. they cannot be chosen
    at the same time.

    For example: if we choose a merged node, we cannot at the same time choose
    one of the fragments from which it was made.

    Constraint:
        sum( selected(node) for node in conflict_set ) <= 1

    Parameters
    ----------
    graph: TrackGraph
        Graph containing the node and edge candidates, the ILP will be solved
        for this graph.
    indicators: dict str: dict (int or pair of int): int
        Contains a dict for every indicator type (str).
        Each dict maps from node (int) or edge (pair of int) candidate to
        the corresponding indicator variable/index (int). Each candidate can
        have indicators of different types associated to it.
    conflict_sets: List[Set]
        Sets of nodes that cannot be chosen at the same time.
    """
    constraints = []
    for cs in conflict_sets:

        constraint = pylp.LinearConstraint()

        for node in cs:
            constraint.set_coefficient(indicators["node_selected"][node], 1)

        # Relation, value
        constraint.set_relation(pylp.Relation.LessEqual)
        constraint.set_value(1)
        constraints.append(constraint)

    return constraints


if __name__ == "__main__":
    import doctest
    doctest.testmod()
