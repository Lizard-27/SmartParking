"""
search_algorithms.py
--------------------
Implements BFS, DFS, and A* search on the campus graph.

Each function returns a dict:
    path            : list of node IDs from source to target
    total_distance  : total walking distance in metres
    nodes_explored  : number of nodes expanded
    algorithm       : name string ('BFS' | 'DFS' | 'A*')
    walking_minutes : total_distance / 80 (80 m/min walking speed)

A* uses a Euclidean heuristic on node grid positions (admissible).
"""

import math
import heapq
from collections import deque

from campus_graph import CAMPUS_GRAPH, NODES

WALKING_SPEED = 80  # metres per minute


# ---------------------------------------------------------------------------
# Helper — total cost of a path
# ---------------------------------------------------------------------------
def _path_cost(G, path):
    return sum(G[u][v]["weight"] for u, v in zip(path, path[1:]))


def _make_result(algorithm, path, cost, explored):
    return {
        "algorithm":       algorithm,
        "path":            path,
        "total_distance":  round(cost),
        "nodes_explored":  explored,
        "walking_minutes": round(cost / WALKING_SPEED, 1),
    }


# ---------------------------------------------------------------------------
# BFS — fewest hops (not necessarily shortest distance)
# ---------------------------------------------------------------------------
def bfs(source, target, G=CAMPUS_GRAPH):
    """
    Breadth-First Search.
    Expands nodes level by level; guarantees fewest hops, not minimum distance.
    """
    if source == target:
        return _make_result("BFS", [source], 0, 0)

    visited = {source}
    queue   = deque([(source, [source])])
    explored = 0

    while queue:
        node, path = queue.popleft()
        explored += 1

        for neighbour in G.neighbors(node):
            if neighbour == target:
                full_path = path + [neighbour]
                return _make_result("BFS", full_path, _path_cost(G, full_path), explored)
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, path + [neighbour]))

    return None  # no path


# ---------------------------------------------------------------------------
# DFS — depth-first (not optimal; included for comparison)
# ---------------------------------------------------------------------------
def dfs(source, target, G=CAMPUS_GRAPH):
    """
    Depth-First Search (iterative, explicit stack).
    Not guaranteed to find shortest path — included to contrast with A*.
    """
    if source == target:
        return _make_result("DFS", [source], 0, 0)

    visited  = set()
    stack    = [(source, [source])]
    explored = 0

    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        explored += 1

        if node == target:
            return _make_result("DFS", path, _path_cost(G, path), explored)

        for neighbour in G.neighbors(node):
            if neighbour not in visited:
                stack.append((neighbour, path + [neighbour]))

    return None


# ---------------------------------------------------------------------------
# A* — optimal weighted path with Euclidean heuristic
# ---------------------------------------------------------------------------
def _euclidean(a, b):
    """
    Straight-line distance between two nodes (grid units × 50 m/unit).
    This is admissible: it never overestimates the true walking distance.
    """
    ax, ay = NODES[a]["pos"]
    bx, by = NODES[b]["pos"]
    return math.hypot(ax - bx, ay - by) * 50


def astar(source, target, G=CAMPUS_GRAPH):
    """
    A* Search.
    Uses g(n) + h(n) where h is the Euclidean distance heuristic.
    Guarantees the shortest weighted path.
    """
    if source == target:
        return _make_result("A*", [source], 0, 0)

    # heap: (f_score, g_score, node, path)
    heap   = [(_euclidean(source, target), 0.0, source, [source])]
    best_g = {source: 0.0}
    explored = 0

    while heap:
        f, g, node, path = heapq.heappop(heap)
        explored += 1

        if node == target:
            return _make_result("A*", path, g, explored)

        if g > best_g.get(node, float("inf")):
            continue  # stale entry

        for neighbour, edge_data in G[node].items():
            new_g = g + edge_data["weight"]
            if new_g < best_g.get(neighbour, float("inf")):
                best_g[neighbour] = new_g
                h = _euclidean(neighbour, target)
                heapq.heappush(heap, (new_g + h, new_g, neighbour, path + [neighbour]))

    return None


# ---------------------------------------------------------------------------
# Run all three and return a dict
# ---------------------------------------------------------------------------
def compare_all(source, target):
    results = {}
    for name, fn in [("BFS", bfs), ("DFS", dfs), ("A*", astar)]:
        r = fn(source, target)
        if r:
            results[name] = r
    return results


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Route: P_NORTH → B_SCI ===\n")
    for name, r in compare_all("P_NORTH", "B_SCI").items():
        path_str = " → ".join(NODES[n]["label"] if n in NODES else n for n in r["path"])
        print(f"[{name}] {path_str}")
        print(f"  Distance : {r['total_distance']} m")
        print(f"  Walk time: {r['walking_minutes']} min")
        print(f"  Explored : {r['nodes_explored']} nodes\n")
