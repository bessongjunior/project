# -*- coding: utf-8 -*-
"""
Routing baselines for grading (evaluation.md §3).

Each baseline takes one trajectory's stops and returns a predicted visit order
as a sequence of input-node indices — the same convention as the dataset
`label` (visit_seq) — so it can be graded with pre_processing/utils.Metric.

Numpy-only (no torch). OR-Tools is optional: if `ortools` is not installed,
`ortools_tsp` degrades to `distance_greedy`.

Routing setup: the courier's starting stop is treated as known (the ground-truth
first stop), and each baseline orders the remaining stops from there.
"""
import numpy as np

_R = 6371000.0  # earth radius, metres


def haversine_matrix(coords):
    """Pairwise great-circle distance (metres) for (N,2) lat/lng array."""
    lat = np.radians(coords[:, 0])[:, None]
    lng = np.radians(coords[:, 1])[:, None]
    dlat = lat - lat.T
    dlng = lng - lng.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlng / 2) ** 2
    return 2 * _R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_greedy(coords, start=0, **_):
    """Nearest-unvisited-stop heuristic (haversine distance proxy)."""
    n = len(coords)
    D = haversine_matrix(coords)
    visited, seen, cur = [start], {start}, start
    while len(visited) < n:
        for j in np.argsort(D[cur]):
            j = int(j)
            if j not in seen:
                visited.append(j); seen.add(j); cur = j
                break
    return np.array(visited, dtype=np.int64)


def time_greedy(coords, times, start=0, **_):
    """Earliest-expected-time-first (deadline ordering) from `start`.

    `times` is any per-stop time signal whose ascending order is the heuristic
    (e.g. expect_finish_time or accept_time); only the ordering matters."""
    n = len(times)
    rest = sorted((i for i in range(n) if i != start), key=lambda i: times[i])
    return np.array([start] + rest, dtype=np.int64)


def ortools_tsp(coords, start=0, time_limit=2, **_):
    """Exact-ish single-vehicle TSP via OR-Tools; falls back to distance_greedy."""
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        return distance_greedy(coords, start=start)

    n = len(coords)
    if n <= 2:
        return distance_greedy(coords, start=start)
    D = haversine_matrix(coords).astype(np.int64)

    mgr = pywrapcp.RoutingIndexManager(n, 1, start)
    routing = pywrapcp.RoutingModel(mgr)

    def cb(i, j):
        return int(D[mgr.IndexToNode(i)][mgr.IndexToNode(j)])

    transit = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(time_limit)

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return distance_greedy(coords, start=start)

    order, idx = [], routing.Start(0)
    while not routing.IsEnd(idx):
        order.append(mgr.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    return np.array(order, dtype=np.int64)


BASELINES = {
    'distance_greedy': distance_greedy,
    'time_greedy': time_greedy,
    'ortools': ortools_tsp,
}


def run_baseline(name, coords, times=None, start=0):
    """Dispatch one baseline -> predicted visit order (input-node indices)."""
    fn = BASELINES[name]
    if name == 'time_greedy':
        if times is None:
            raise ValueError("time_greedy requires `times`")
        return fn(coords, times=times, start=start)
    return fn(coords, start=start)
