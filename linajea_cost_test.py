import logging
import unittest

import networkx as nx

import daisy

import linajea.config
import linajea.tracking
import linajea.tracking.track
import linajea.tracking.cost_functions
import linajea.utils
from linajea.tracking.cost_functions import is_nth_frame
from linajea.tracking.cost_functions import is_close_to_roi_border
from linajea.tracking.constraints import ensure_edge_endpoints,ensure_at_most_two_successors,ensure_one_predecessor,ensure_pinned_edge,ensure_split_set_for_divs




class TrackingConfig():
    def __init__(self, solve_config):
        self.solve = solve_config






'''

    Edge cost

'''
def feature_times_weight_costs_fn(weight,key="score",
                                  feature_func=lambda x: x):

    def fn(obj):
        feature = feature_func(obj[key])
        return feature, weight

    return fn
def get_default_edge_indicator_costs(parameters,graph):
    """Get a predefined map of edge indicator costs functions
    Args
    ----
    config: TrackingConfig
        Configuration object used, should contain information on which solver
        type to use.
    parameters: SolveParametersConfig
        Current set of weights and parameters used to compute costs.
    graph: TrackGraph
        Graph containing the node candidates for which the costs will be
        computed (not used for the default edge costs).
    """


    fn_map = {
        "edge_selected": [
            feature_times_weight_costs_fn(parameters.weight_edge_score,
                                          key="overlap",
                                          feature_func=lambda x: x)]
    }


    return fn_map


'''

node cost


'''


def constant_costs_fn(weight, zero_if_true=lambda _: False):

    def fn(obj):
        feature = 1
        cond_weight = 0 if zero_if_true(obj) else weight
        return feature, cond_weight

    return fn


def get_default_node_indicator_costs(parameters, graph):
    """Get a predefined map of node indicator costs functions
    Args
    ----
    config: TrackingConfig
        Configuration object used, should contain information on which solver
        type to use.
    parameters: SolveParametersConfig
        Current set of weights and parameters used to compute costs.
    graph: TrackGraph
        Graph containing the node candidates for which the costs will be
        computed.
    """
    feature_func = lambda x: x
 
    fn_map = {
        "node_selected": [
            feature_times_weight_costs_fn(
                parameters.weight_node_score,
                key="score", feature_func=feature_func),
            constant_costs_fn(parameters.selection_constant)],
        "node_appear": [
            constant_costs_fn(
                parameters.track_cost,
                zero_if_true=lambda obj: (
                    is_nth_frame(graph.begin)(obj) or
                    (config.solve.check_node_close_to_roi and
                     is_close_to_roi_border(
                         graph.roi, parameters.max_cell_move)(obj))))],
        "node_split":  [
            constant_costs_fn(1)]
    }

    return fn_map



'''

Constrains

'''




def get_default_constraints(config):
    pin_constraints_fn_list = [ensure_pinned_edge]
    constraints_fn_list = [ensure_edge_endpoints, 
                               ensure_one_predecessor,
                               ensure_at_most_two_successors,
                               ensure_split_set_for_divs]
    return (constraints_fn_list,
            pin_constraints_fn_list)






if __name__ == "__main__":
    '''x
          3|         /-4
          2|        /--3 --// 8
          1|   0---1 --5 -//- 7
          0|        \--2 /--- 6
            ------------------------------------ t
               0   1   2   3

        Should select 0, 1, 2, 3, 5
    '''

    cells = [
            {'id': 0, 't': 0, 'z': 1, 'y': 1, 'x': 1, 'score': 2.0},
            {'id': 1, 't': 1, 'z': 1, 'y': 1, 'x': 1, 'score': 2.0},
            {'id': 2, 't': 2, 'z': 1, 'y': 1, 'x': 0, 'score': 2.0},
            {'id': 3, 't': 2, 'z': 1, 'y': 1, 'x': 2, 'score': 2.0},
            {'id': 4, 't': 2, 'z': 1, 'y': 1, 'x': 3, 'score': 2.0},
            {'id': 5, 't': 2, 'z': 1, 'y': 1, 'x': 1, 'score': 2.0},
            {'id': 6, 't': 3, 'z': 1, 'y': 1, 'x': 0, 'score': 2.0},
            {'id': 7, 't': 3, 'z': 1, 'y': 1, 'x': 1, 'score': 2.0},
            {'id': 8, 't': 3, 'z': 1, 'y': 1, 'x': 3, 'score': 2.0}

    ]

    edges = [
                {'source': 1, 'target': 0, 'score': 1.0,
                'overlap': 1.0},
                {'source': 2, 'target': 1, 'score': 1.0,
                'overlap': 0.4},
                {'source': 3, 'target': 1, 'score': 1.0,
                'overlap': 0.3},
                {'source': 4, 'target': 1, 'score': 1.0,
                'overlap': 0.1},
                {'source': 5, 'target': 1, 'score': 1.0,
                'overlap': 0.9},
                {'source': 6, 'target': 2, 'score': 1.0,
                'overlap': 0.9},
                {'source': 7, 'target': 5, 'score': 1.0,
                'overlap': 0.8},
                {'source': 7, 'target': 2, 'score': 1.0,
                'overlap': 0.4},
                {'source': 8, 'target': 3, 'score': 1.0,
                'overlap': 1.0},
                {'source': 8, 'target': 2, 'score': 1.0,
                'overlap': 0.1},
                {'source': 8, 'target': 5, 'score': 1.0,
                'overlap': 0.6},
    ]

    roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
    graph = nx.DiGraph()
    graph.add_nodes_from([(cell['id'], cell) for cell in cells])
    graph.add_edges_from([(edge['source'], edge['target'], edge)
                            for edge in edges])
    graph = linajea.tracking.TrackGraph(graph, frame_key='t', roi=roi)
    ps = {
        "track_cost": 4.0,
        "weight_edge_score": -0.1,
        "weight_node_score": -0.1,
        "selection_constant": -1.0,
        "max_cell_move": 0.0,
        "block_size": [5, 100, 100, 100],
        }
    job = {"num_workers": 5, "queue": "normal"}
    solve_config = linajea.config.SolveConfig(
        parameters=ps, job=job, context=[2, 100, 100, 100])
    solve_config.solver_type = None
    solve_config.timeout =1000
    config = TrackingConfig(solve_config)
    
    
    import pylp
    
    s = pylp.LinearSolver(10, pylp.Binary, preference=pylp.Scip)
    s.set_num_threads(2)


    constrs = get_default_constraints(config)
    

    linajea.tracking.track(
                graph,
                config,
                frame_key='t',
                selected_key='selected',
                edge_indicator_costs=get_default_edge_indicator_costs,
                node_indicator_costs=get_default_node_indicator_costs,
                constraints_fns=constrs[0],
                pin_constraints_fns=constrs[1]
                )

    selected_edges = []
    edges=[]
    for u, v, data in graph.edges(data=True):
        edges.append((u,v))
        if data['selected']:
            selected_edges.append((u, v))

    print('''x
          3|         /-4
          2|        /--3 --// 8
          1|   0---1 --5 -//- 7
          0|        \--2 /--- 6
            ------------------------------------ t
               0   1   2   3

        Should select 0, 1, 2, 3, 5
    ''')
    print('all edges graph:',edges)
    print('selected edges:',selected_edges)
    












