import zarr
import daisy
import numpy as np
import networkx as nx
from linajea_cost_test import get_merge_graph_from_array
from funlib.math import encode64, decode64
import linajea
from linajea_cost_test import get_edge_indicator_costs, get_node_indicator_costs, get_constraints

class TrackingConfig():
    def __init__(self, solve_config):
        self.solve = solve_config

def overlap(frg_pre, frg_next):
    f_pre = frg_pre.flatten()
    f_next = frg_next.flatten()
    pairs, counts = np.unique(np.stack([f_pre, f_next]), axis=1,
                              return_counts=True)
    return pairs.T, counts

# find root and leaves and connect edge


def provide_root(graph):
    roots = [x for x in graph.nodes() if graph.in_degree(x) == 0]
    return roots

def provide_leaves(graph):
    leaves = [x for x in graph.nodes() if graph.out_degree(x) == 0]
    return leaves

def iterate_tree(graph,r):
    iter_list = [r,*list(nx.dfs_predecessors(graph,r))]
    iter_list.reverse()
    return iter_list

def connect_edge(sub_a,sub_b,pairs,score):
    count=0
    for a in sub_a:
        for b in sub_b:
            if (a,b) in pairs:
                ind = pairs.index((a,b))
                count += score[ind] 
    return count

def add_nodes_from_merge_tree(G1,G2):
        for node in G1.nodes():
                # Add the node to G2, copying over the attributes
                G2.add_node(node, **G1.nodes[node])
                # Check if the node has any incoming edges in G1
                parent = None
                for edge in G1.in_edges(node):
                        parent = edge[0]
                # update the node with `parent`
                G2.nodes[node]['parent'] = parent
                G2.nodes[node]['id'] = node
        return G2


if __name__ == "__main__":

    file_name = "/groups/funke/home/xuz2/Alice_demo_2.zarr"
    z = zarr.open ( '/groups/funke/home/xuz2/Alice_demo_2.zarr' , 'r' )

    fragments = z['Fragments'][:]
    raw = z['Raw'][:]
    #track = z['gt_trackimage']
    print(z.tree(level=2))
    
    t_begin, t_end = 0,2
    frg= fragments[:]
    # fragments_edge
    #graph_fragments = nx.DiGraph()
    candidate_edge = {}
    graph_fragments=nx.Graph()

    for t in range(t_begin, t_end):
        
        pre = t 
        nex = t + 1
        z = zarr.open(file_name,'r')
        ids_pre = z['Fragment_stats/id/'+str(pre)]
        ids_nex = z['Fragment_stats/id/'+str(nex)]
        merge_tree_pre = z['Merge_tree/Merge/'+str(pre)]
        merge_tree_nex = z['Merge_tree/Merge/'+str(nex)]
        scores_pre = z['Merge_tree/Scoring/'+str(pre)]
        scores_nex = z['Merge_tree/Scoring/'+str(nex)]

        merge_tree_pre = get_merge_graph_from_array(merge_tree_pre,scores_pre)
        merge_tree_nex = get_merge_graph_from_array(merge_tree_nex,scores_nex)

        graph_fragments = add_nodes_from_merge_tree(merge_tree_pre,graph_fragments)
        graph_fragments = add_nodes_from_merge_tree(merge_tree_nex,graph_fragments)

        # find overlaping pairs
        pairs, counts = overlap (frg[pre], frg[nex]) 
        for p,c in zip(pairs,counts):
            graph_fragments.add_edge(p[0], p[1], overlap = c)
        
        # iterate tree to extract edge
        root = provide_root(merge_tree_pre)
        iter_list_A = iterate_tree(merge_tree_pre,root[0])
        root = provide_root(merge_tree_nex)
        iter_list_B = iterate_tree(merge_tree_nex,root[0])
        for a in iter_list_A:
            sub_a = iterate_tree(merge_tree_pre,a)
            for b in iter_list_B: 
                sub_b = iterate_tree(merge_tree_nex,b)
                count = connect_edge(sub_a,sub_b,pairs,counts)
                # add edegs
                if count != 0:
                    graph_fragments.add_edge(a, b, overlap = count)
                #print('add',a,b , 'count',count)
    
    
    roi = daisy.Roi((0, 0, 0, 0), (3, 15, 25, 25))

    # input graph
    graph = linajea.tracking.TrackGraph(graph_fragments, frame_key='t', roi=roi)
    ps = {
            "track_cost": 4.0,
            "weight_edge_score": -0.1,
            "weight_node_score": -0.1,
            "selection_constant": -1.0,
            "max_cell_move": 0.0,
            "block_size": [5, 100, 100, 100],
            }
    job = {"num_workers": 5, "queue": "normal"}

    solve_config = linajea.config.SolveConfig(parameters=ps,
                                                job=job,
                                                context=[2, 100, 100, 100])

    solve_config.solver_type = None
    


    import linajea
    print(linajea.__file__)


#solve_config.solver_type = None

    config = TrackingConfig(solve_config)
    constrs = get_constraints()

    linajea.tracking.track(
                    graph,
                    config,
                    frame_key='t',
                    selected_key='selected',
                    edge_indicator_costs=get_edge_indicator_costs,
                    node_indicator_costs=get_node_indicator_costs,
                    constraints_fns=constrs[0],
                    pin_constraints_fns=constrs[1]
                    )

    
    #all_pre_cells = cells_by_t[pre]



