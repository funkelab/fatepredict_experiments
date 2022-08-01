"""Mostly copied from https://github.com/Kainmueller-Lab/linajea"""
import logging

import linajea
import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def affine_fn(x, transform_matrix):
    # Currently an identity
    # TODO how to apply an affine function here for registration???
    return x


def extract_edges(block, t, transform, db_name, db_host,
                  edge_move_threshold):
    """Extract edges between times t and t+1 in the tracking CandidateGraph

    block: daisy.Roi
        Block of data + context from the zarr file
    """
    # Get candidate graph obtained from the extracion of nodes
    # This comes from the MONGO db
    graph_provider = linajea.utils.CandidateDatabase(
        db_name,
        db_host,
        mode='r+')
    # TODO need an ROI to get timepoints t - t+1
    graph = graph_provider[block.read_roi]

    t_begin = t
    t_end = t + 1

    cells_by_t = {
        t: [
            (
                cell,
                np.array([attrs[d] for d in ['z', 'y', 'x']]),
            )
            for cell, attrs in graph.nodes(data=True)
            if 't' in attrs and attrs['t'] == t
        ]
        for t in range(t_begin,
                       t_end)
    }

    pre, nex = t_begin, t_end

    if len(cells_by_t[pre]) == 0 or len(cells_by_t[nex]) == 0:
        logger.debug("There are no edges between these frames, skipping")
        continue

    # prepare KD tree for fast lookup of 'pre' cells
    logger.debug("Preparing KD tree...")
    all_pre_cells = cells_by_t[pre]
    # TODO double_check this
    kd_data = [cell[1] for cell in all_pre_cells]
    pre_kd_tree = KDTree(kd_data)

    # Loop over the cells at time point nex
    for i, nex_cell in enumerate(cells_by_t[nex]):
        # TODO CHECK THIS INFORMATION
        # Get the cell information from the database
        nex_cell_id = nex_cell[0]
        nex_cell_center = nex_cell[1]
        nex_parent_center = nex_cell_center + nex_cell[2]

        # Get all cells in previous time point within a certain distance
        # TODO Affine transform!
        pre_cells_indices = pre_kd_tree.query_ball_point(
            nex_cell_center,
            edge_move_threshold)

        # Loop over accepted pre-cells and add edge
        for idx in pre_cells_indices:
            pre_cell = all_pre_cells[idx]

            pre_cell_id = pre_cell[0]
            pre_cell_center = pre_cell[1]

            moved = (pre_cell_center - nex_cell_center)
            distance = np.linalg.norm(moved)

            prediction_offset = (pre_cell_center - nex_parent_center)
            prediction_distance = np.linalg.norm(prediction_offset)
            # TODO add overlap as a score to the edge
            # Add an edge to the graph
            graph.add_edge(
                nex_cell_id, pre_cell_id,
                distance=distance,
                prediction_distance=prediction_distance)
