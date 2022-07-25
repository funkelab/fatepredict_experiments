"""Mostly copied from https://github.com/Kainmueller-Lab/linajea"""
import logging

import daisy
import linajea
import numpy as np
import pymongo
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def affine_fn(x, transform_matrix):
    # Currently an identity
    # TODO how to apply an affine function here for registration???
    return x


def check_function(block, step_name, db_name, db_host):
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    daisy_coll = db[step_name + '_daisy']
    result = daisy_coll.find_one({'_id': block.block_id[1]})
    if result is None:
        return False
    else:
        return True


def extract_edges_blockwise(db_name, db_host, data, transform,
                            roi, context,
                            block_size=10,
                            max_edge_move_threshold=10,
                            num_workers=4):
    """Function to extract edges (compute edge candidates between
    neighboring object candidates in adjacent frames)

    Starts a number of worker processes to process the blocks.

    Parameters
    ----
    db_name: str
        Name of the data "table" in the database
    db_host: str
        Hostname for mongoDB server
    data: Zarr file
        Open zarr file
    transform: zarr dataset
        ND-arrray + offset? that describes the affine transform for each point
        in time. i.e. at index t, there is the transform between t-1 and t.
        Index 0 just has identity matrix.
    roi: daisy.Roi
        Describes the full extent of the data in the zarr file
        Note: might not be necessary if using full file
    context: int
        Size by which to grow the roi for extra context.
        Note: might not be necessary if using full file
    block_size: int or tuple(int)
        Size (z, y, x) of a single block on which the task is run
    max_edge_move_threshold: int
        Size of the neighborhood for creating edges between frames
    num_workers: int
        Number of simultaneous CPU workers running a daisy block
    """

    extract_roi = daisy.Roi(offset=roi.offset,
                            shape=roi.shape)
    # allow for solve context
    extract_roi = extract_roi.grow(
            daisy.Coordinate(context),
            daisy.Coordinate(context))

    # block size in world units
    block_write_roi = daisy.Roi(
        (0,)*4,
        daisy.Coordinate(block_size))

    pos_context = daisy.Coordinate((0,) + (max_edge_move_threshold,)*3)
    neg_context = daisy.Coordinate((1,) + (max_edge_move_threshold,)*3)
    logger.debug("Set neg context to %s", neg_context)

    input_roi = extract_roi.grow(neg_context, pos_context)
    block_read_roi = block_write_roi.grow(neg_context, pos_context)

    logger.info("Following ROIs in world units:")
    logger.info("Input ROI       = %s", input_roi)
    logger.info("Block read  ROI = %s", block_read_roi)
    logger.info("Block write ROI = %s", block_write_roi)
    logger.info("Output ROI      = %s", extract_roi)

    logger.info("Starting block-wise processing...")
    if data.datafile is not None:
        logger.info("Sample: %s", data.datafile.filename)
    logger.info("DB: %s", data.db_name)

    # process block-wise
    task = daisy.Task(
        "linajea_extract_edges",
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: extract_edges_in_block(
            b, transform, db_name, db_host, max_edge_move_threshold),
        check_function=lambda b: check_function(
            b,
            'extract_edges',
            db_name,
            db_host),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='overhang')

    daisy.run_blockwise([task])


def extract_edges_in_block(block, transform, db_name, db_host,
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
    graph = graph_provider[block.read_roi]

    t_begin = 0
    t_end = 1  # TODO

    # TODO: This does all times
    cells_by_t = {
        t: [
            (
                cell,
                np.array([attrs[d] for d in ['z', 'y', 'x']]),
            )
            for cell, attrs in graph.nodes(data=True)
            if 't' in attrs and attrs['t'] == t
        ]
        for t in range(t_begin - 1,
                       t_end)
    }

    for pre, nex in zip(range(t_begin, t_end - 1),
                        range(t_begin + 1, t_end)):

        if len(cells_by_t[pre]) == 0 or len(cells_by_t[nex]) == 0:

            logger.debug("There are no edges between these frames, skipping")
            continue

        # prepare KD tree for fast lookup of 'pre' cells
        logger.debug("Preparing KD tree...")
        all_pre_cells = cells_by_t[pre]
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
