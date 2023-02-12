import logging
import time
import zarr
import numpy as np
import daisy
from .daisy_check_functions import write_done, check_function
import linajea
from linajea_cost_test import get_merge_graph_from_array

logger = logging.getLogger(__name__)


def extract_edges_blockwise(linajea_config):

    """
    Function to extract edges (compute edge candidates between
    neighboring object candidates in adjacent frames)
    Starts a number of worker processes to process the blocks.
    Args
    ----
    linajea_config: TrackingConfig
        Configuration object
    """
    data = linajea_config.inference_data.data_source
    extract_roi = daisy.Roi(offset=data.roi.offset,
                            shape=data.roi.shape)
    

    

    assert linajea_config.solve.context, (
        "config.solve.context is not set, cannot determine how much spatial"
        " context is necessary for tracking (cells moving in and out of a"
        " block at the boundary). Please set it!")
    

    # set roi
    extract_roi = extract_roi.grow(
            daisy.Coordinate(linajea_config.solve.context),
            daisy.Coordinate(linajea_config.solve.context))

    # block size in world units
    block_write_roi = daisy.Roi(
        (0,)*4,
        daisy.Coordinate(linajea_config.extract.block_size))

    # set thresgold
    max_edge_move_th = max(linajea_config.extract.edge_move_threshold.values())
    pos_context = daisy.Coordinate((0,) + (max_edge_move_th,)*3)
    neg_context = daisy.Coordinate((1,) + (max_edge_move_th,)*3)


    input_roi = extract_roi.grow(neg_context, pos_context)
    block_read_roi = block_write_roi.grow(neg_context, pos_context)

    logger.info("Following ROIs in world units:")
    logger.info("Input ROI       = %s", input_roi)
    logger.info("Block read  ROI = %s", block_read_roi)
    logger.info("Block write ROI = %s", block_write_roi)
    logger.info("Output ROI      = %s", extract_roi)

    logger.info("Starting block-wise processing...")

    container = daisy.prepare_ds(
                        data,
                        'Fragments')


    def extract_edges_in_block(linajea_config, block):


        graph_provider = linajea.utils.CandidateDatabase(data.db_name,
                                                         linajea_config.general.db_host,
                                                          mode='r+')
        
        graph = graph_provider[block.read_roi]

        fragments = container[block.read_roi]
        t_begin = block.read_roi.begin[0]
        t_end = block.read_roi.end[0]
        for t in range(t_begin,t_end-1):
            pre = t 
            nex = t + 1

            z = zarr.open(linajea_config.inference_data.data_source.datafile,'r')
            ids_pre = z['Fragment_stats/id/'+str(pre)]
            ids_nex = z['Fragment_stats/id/'+str(nex)]
            merge_tree_pre = z['Merge_tree/Merge/'+str(pre)]
            merge_tree_nex = z['Merge_tree/Merge/'+str(nex)]
            scores_pre = z['Merge_tree/Scoring/'+str(pre)]
            scores_nex = z['Merge_tree/Scoring/'+str(nex)]

            merge_tree_pre = get_merge_graph_from_array(merge_tree_pre,scores_pre)
            merge_tree_nex = get_merge_graph_from_array(merge_tree_nex,scores_nex)

            '''
            graph_fragments = add_nodes_from_merge_tree(merge_tree_pre,graph_fragments)
            graph_fragments = add_nodes_from_merge_tree(merge_tree_nex,graph_fragments)
            '''

            pairs, counts = overlap (fragments[pre], fragments[nex])

            root = provide_root(merge_tree_pre)
            # set_oder to iterate merge_tree
            iter_list_A = iterate_tree(merge_tree_pre,root[0])
            root = provide_root(merge_tree_nex)
            iter_list_B = iterate_tree(merge_tree_nex,root[0])

            # iterate two merge_tree and connect new edges
            for a in iter_list_A:
                sub_a = iterate_tree(merge_tree_pre,a)
                for b in iter_list_B: 
                    sub_b = iterate_tree(merge_tree_nex,b)
                    # create edges for connecting the node a in merge tree merge_pre and node a in merge tree merge_nex
                    count = connect_edge(sub_a,sub_b,ids_pre,ids_nex,pairs,counts)
                    # add edegs
                    if count != 0:
                        graph.add_edge(b, a, source = b, target = a, overlap = count)
            
            graph.write_edges(block.write_roi)

        return 0




    task = daisy.Task(
        "cabdidategraph_extract_edges",
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: extract_edges_in_block(
            linajea_config,
            b),
        num_workers=linajea_config.extract.job.num_workers,
        read_write_conflict=False,
        fit='overhang')

    daisy.run_blockwise([task])





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

def connect_edge(sub_a,sub_b,ids_pre,ids_nex,pairs,score):
    # create edges for connecting the node a in merge tree merge_pre and node a in merge tree merge_nex
    count=0
    for a in sub_a:
        for b in sub_b:
            if a in list(ids_pre) and b in list(ids_nex):
                label_a = list(ids_pre).index(a)+1
                label_b = list(ids_nex).index(b)+1
                ind = np.where(np.all(pairs == [[label_a,label_b]], axis=1))
                if len(ind[0]>0):
                    #a,b is id will not be show in pairs
                    index = ind[0].item()
                    count += score[index] 
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