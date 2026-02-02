"""
This module provides utility functions for the simulation.
"""

import math
import numpy as np
# from mobref.graph_utils import create_pdn_graph
import pandas as pd
import random


def compute_od_matrix(agents, network):
    """
    Computes the Origin-Destination (OD) matrix representing transport costs
    between agents.

    Args:
        agents (dict): A dictionary of agents.
        network (object): The network object (can be None for Euclidean distances).

    Returns:
        pd.DataFrame: A DataFrame representing the OD matrix.
    """

    if network:
        agents_data = [(a.id, a.location[0], a.location[1]) for a in agents.values()]
        pois = pd.DataFrame(agents_data, columns=["id", "lon", "lat"])
        pdn = create_pdn_graph(network.nodes, network.edges, impedence="length")
        pois_nodes = pdn.get_node_ids(pois.lon, pois.lat)
        origs = [o for o in pois_nodes.values for d in pois_nodes.values]
        dests = [d for o in pois_nodes.values for d in pois_nodes.values]
        distances = pdn.shortest_path_lengths(origs, dests)
        n = int(math.sqrt(len(distances)))
        distances = np.array(distances).reshape((n, n))
        dist_matrix = pd.DataFrame(distances, index=pois.id, columns=pois.id)
    
    else:
        # Compute Euclidean distances
        ids = [a.id for a in agents.values()]
        locations = np.array([a.location for a in agents.values()])
        dist_matrix = pd.DataFrame(
            np.linalg.norm(locations[:, np.newaxis, :] - locations[np.newaxis, :, :], axis=2),
            index=ids,
            columns=ids
        )
    return dist_matrix


def normalize(arr):
    """
    Normalizes a numpy array to a range of 0 to 1.

    Args:
        arr (np.ndarray): The input numpy array.

    Returns:
        np.ndarray: The normalized array.
    """
    arr = np.array(arr)
    min_val = arr.min()
    max_val = arr.max()
    if min_val == max_val:
        return np.zeros_like(arr, dtype=float)  # or np.full_like(arr, 0.5)
    return (arr - min_val) / (max_val - min_val)


def norm_softmax(weights, temperature=1.0, val_min=None, val_max=None):
    """
    Computes the normalized softmax of a given array of weights.

    Args:
        weights (np.ndarray): The input array of weights.
        temperature (float, optional): The temperature parameter for softmax.
            Defaults to 1.0.
        val_min (float, optional): The minimum value for normalization. Defaults to None.
        val_max (float, optional): The maximum value for normalization. Defaults to None.

    Returns:
        np.ndarray: The array of softmax probabilities.
    """
    # t -> 0 : greedy (max); t -> inf : uniform
    if not val_min:
        val_min = np.min(weights)
    if not val_max:
        val_max = np.max(weights)
    if val_max==val_min:
        return np.ones_like(weights) / len(weights)
    weights = (weights - val_min)/(val_max-val_min) 
    if np.all(weights == 0):
        return np.ones_like(weights) / len(weights)
    scaled = weights / temperature
    exp_weights = np.exp(scaled - np.max(scaled))
    p = exp_weights / exp_weights.sum()
    return p

def uniform_cluster_centers(area, n_clusters):
    n_side = int(np.ceil(np.sqrt(n_clusters)))

    step_x = area.width / n_side
    step_y = area.width / n_side

    xs = np.linspace(step_x/2, area.width - step_x/2, n_side)
    ys = np.linspace(step_y/2, area.width - step_y/2, n_side)

    grid = [(x, y) for x in xs for y in ys]

    random.shuffle(grid)
    return grid[:n_clusters]