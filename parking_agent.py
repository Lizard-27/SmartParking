"""
parking_agent.py
----------------
The Smart Parking Agent.

Pipeline:
  1. Predict occupancy for every lot (neural network).
  2. Compute the route from each lot to the destination using the chosen algorithm.
  3. Score and rank lots using a weighted composite of availability + distance.
  4. Return the best recommendation with an explanation and list of alternatives.

Also supports:
  - accessible_only  : only lots with accessible spaces
  - staff_only       : only lots with reserved staff spaces
  - re-routing       : if the recommended lot is full on arrival, pick the next best
"""

from campus_graph  import CAMPUS_GRAPH, NODES, PARKING_LOTS
from occupancy_model import OccupancyPredictor
from search_algorithms import astar, bfs, dfs, compare_all

WALKING_SPEED = 80   # metres per minute
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------
def _score(occ, distance, preference):
    """
    Composite score in [0, 1] — higher is better.

    Weights shift depending on user preference:
      'nearest'   → distance dominates
      'available' → free-space probability dominates
      'fastest'   → same as nearest on a walking graph
    """
    MAX_DIST   = 1200.0
    free_score = 1.0 - occ
    dist_score = 1.0 - min(distance / MAX_DIST, 1.0)

    if preference == "available":
        w_free, w_dist = 0.80, 0.20
    elif preference in ("nearest", "fastest"):
        w_free, w_dist = 0.25, 0.75
    else:
        w_free, w_dist = 0.50, 0.50

    return round(w_free * free_score + w_dist * dist_score, 4)


# ---------------------------------------------------------------------------
# ParkingAgent
# ---------------------------------------------------------------------------
class ParkingAgent:
    """
    Autonomous parking recommendation agent.

    Typical use
    -----------
        agent = ParkingAgent()
        agent.initialise()          # trains neural network once
        result = agent.recommend(
            destination = "B_SCI",
            hour        = 9,
            weekday     = 1,        # Tuesday
            preference  = "nearest",
        )
        print(agent.format_report(result))
    """

    def __init__(self):
        self.predictor = OccupancyPredictor()
        self._ready    = False

    def initialise(self, verbose=True):
        """Train the occupancy model. Must be called before recommend()."""
        self.predictor.train(verbose=verbose)
        self._ready = True

    # ------------------------------------------------------------------
    def recommend(
        self,
        destination,
        hour,
        weekday,
        preference  = "available",
        event       = 0,
        algorithm   = "astar",
        accessible_only = False,
        staff_only      = False,
    ):
        """
        Run the full pipeline and return a result dict.

        Parameters
        ----------
        destination     : node ID of target building (e.g. 'B_SCI')
        hour            : 0-23
        weekday         : 0 (Mon) – 6 (Sun)
        preference      : 'nearest' | 'fastest' | 'available'
        event           : 1 if campus event today
        algorithm       : 'astar' | 'bfs' | 'dfs'  (used for route search)
        accessible_only : restrict to lots with accessible_spaces > 0
        staff_only      : restrict to lots with reserved_staff > 0
        """
        if not self._ready:
            raise RuntimeError("Call initialise() first.")
        if destination not in CAMPUS_GRAPH:
            raise ValueError(f"Unknown destination: '{destination}'")

        algo_fn = {"astar": astar, "bfs": bfs, "dfs": dfs}[algorithm]

        # Step 1 — predict occupancy for all lots
        occupancy = self.predictor.predict_all(hour, weekday, event)

        # Step 2 — route search from each eligible lot → destination
        candidates = []
        for lot_id, meta in PARKING_LOTS.items():
            if accessible_only and meta["accessible_spaces"] == 0:
                continue
            if staff_only and meta["reserved_staff"] == 0:
                continue

            occ   = occupancy.get(lot_id, 0.5)
            free  = max(0, round(meta["capacity"] * (1 - occ)))
            route = algo_fn(lot_id, destination)
            if route is None:
                continue

            sc = _score(occ, route["total_distance"], preference)
            candidates.append({
                "lot_id":       lot_id,
                "lot_label":    NODES[lot_id]["label"],
                "occ":          occ,
                "free_spaces":  free,
                "walk_dist":    route["total_distance"],
                "walk_min":     route["walking_minutes"],
                "score":        sc,
                "route":        route,
            })

        if not candidates:
            raise RuntimeError("No lots available after applying filters.")

        # Step 3 — rank
        candidates.sort(key=lambda c: c["score"], reverse=True)
        for i, c in enumerate(candidates):
            c["rank"] = i + 1

        best         = candidates[0]
        alternatives = candidates[1:]

        # Step 4 — run all 3 algorithms on the best lot for the comparison section
        algo_comparison = compare_all(best["lot_id"], destination)

        return {
            "best":            best,
            "alternatives":    alternatives,
            "algo_comparison": algo_comparison,
            "destination":     destination,
            "dest_label":      NODES[destination]["label"],
            "hour":            hour,
            "weekday":         weekday,
            "preference":      preference,
            "algorithm":       algorithm,
            "event":           event,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def format_report(result):
        """Print-friendly text report of a recommendation result."""
        b    = result["best"]
        sep  = "─" * 62

        lines = [
            "",
            sep,
            "  SMART PARKING FINDER — RECOMMENDATION REPORT",
            "  CET251: Artificial Intelligence | El Sewedy University",
            sep,
            f"  Destination : {result['dest_label']}",
            f"  Arrival     : {DAYS[result['weekday']]}  {result['hour']:02d}:00",
            f"  Preference  : {result['preference']}",
            f"  Algorithm   : {result['algorithm'].upper()}",
            sep,
            f"  ★  RECOMMENDED: {b['lot_label']}",
            f"     Predicted occupancy : {b['occ']:.1%}",
            f"     Estimated free spots: {b['free_spaces']}",
            f"     Walking distance    : {b['walk_dist']} m",
            f"     Walking time        : {b['walk_min']} min",
            "",
        ]

        # Route steps
        path     = b["route"]["path"]
        path_str = " → ".join(NODES[n]["label"] if n in NODES else n for n in path)
        lines += [
            f"  ROUTE ({result['algorithm'].upper()}):",
            f"     {path_str}",
            f"     Nodes explored: {b['route']['nodes_explored']}",
            sep,
        ]

        # Algorithm comparison
        lines += ["  ALGORITHM COMPARISON (same lot → destination):"]
        lines.append(f"  {'Algorithm':<10} {'Distance (m)':<16} {'Nodes explored':<18} {'Walk (min)'}")
        lines.append(f"  {'─'*8:<10} {'─'*12:<16} {'─'*14:<18} {'─'*9}")
        for name, r in result["algo_comparison"].items():
            lines.append(
                f"  {name:<10} {r['total_distance']:<16} {r['nodes_explored']:<18} {r['walking_minutes']}"
            )
        lines += [
            "",
            "  A* explores fewer nodes than BFS/DFS because its Euclidean",
            "  heuristic guides it directly toward the goal.",
            sep,
        ]

        # Alternatives
        if result["alternatives"]:
            lines.append("  ALTERNATIVE OPTIONS:")
            for alt in result["alternatives"]:
                lines.append(
                    f"  #{alt['rank']}  {alt['lot_label']:<14}  "
                    f"occ={alt['occ']:.0%}  "
                    f"free={alt['free_spaces']:3d}  "
                    f"walk={alt['walk_min']} min"
                )
            lines.append(sep)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def reroute(self, result, full_lot_id):
        """
        Stretch feature: if the recommended lot is full on arrival,
        return the next available alternative.
        """
        alternatives = [a for a in result["alternatives"]
                        if a["lot_id"] != full_lot_id and a["free_spaces"] > 0]
        if not alternatives:
            return None
        new_best = alternatives[0]
        print(f"\n  [Re-routing] '{full_lot_id}' was full — switching to {new_best['lot_label']}")
        return new_best
