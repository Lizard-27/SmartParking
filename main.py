"""
main.py
-------
Entry point for the Smart Parking Finder & Route Advisor.
CET251: Artificial Intelligence | El Sewedy University of Technology

Run modes:
  Interactive (prompts you for input):
      python main.py

  Command-line arguments (skip prompts):
      python main.py --dest B_SCI --hour 9 --day 1 --pref available
      python main.py --dest B_LIB --hour 14 --day 3 --pref nearest --event
      python main.py --dest B_GYM --hour 18 --day 4 --pref fastest --accessible

Arguments:
  --dest        Destination building ID
                  B_MAIN  = Main Hall
                  B_SCI   = Science Block
                  B_ENG   = Engineering
                  B_LIB   = Library
                  B_GYM   = Sports Complex
                  B_ADMIN = Admin Building
  --hour        Arrival hour (0-23)
  --day         Weekday number: 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri 5=Sat 6=Sun
  --pref        nearest | fastest | available
  --algo        astar (default) | bfs | dfs
  --event       Flag: campus event today (raises occupancy)
  --accessible  Flag: only show lots with accessible spaces
  --staff       Flag: only show lots with staff spaces
  --no-charts   Skip saving PNG charts to output/
"""

import argparse
import os
import sys

from parking_agent import ParkingAgent
from campus_graph  import NODES, PARKING_LOTS
from visualise     import draw_campus_graph, draw_occupancy_chart, draw_algo_comparison

BUILDINGS = {k: v["label"] for k, v in NODES.items() if v["type"] == "building"}
DAYS      = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
SEP       = "─" * 62


# ---------------------------------------------------------------------------
# Interactive prompt helpers
# ---------------------------------------------------------------------------

def _pick_building():
    items = list(BUILDINGS.items())
    print("\n  Destination building:")
    for i, (nid, label) in enumerate(items, 1):
        print(f"    {i}. {label}  [{nid}]")
    while True:
        try:
            idx = int(input("\n  Enter number: ").strip()) - 1
            if 0 <= idx < len(items):
                return items[idx][0]
        except ValueError:
            pass
        print("  Invalid — try again.")


def _pick_hour():
    while True:
        try:
            h = int(input("  Arrival hour (0-23): ").strip())
            if 0 <= h <= 23:
                return h
        except ValueError:
            pass
        print("  Enter a number 0-23.")


def _pick_day():
    print("  Day: 0=Mon  1=Tue  2=Wed  3=Thu  4=Fri  5=Sat  6=Sun")
    while True:
        try:
            d = int(input("  Enter number: ").strip())
            if 0 <= d <= 6:
                return d
        except ValueError:
            pass
        print("  Enter 0-6.")


def _pick_option(prompt, options):
    print(f"  {prompt}")
    for i, (key, label) in enumerate(options.items(), 1):
        print(f"    {i}. {label}")
    while True:
        try:
            idx = int(input("  Enter number: ").strip()) - 1
            keys = list(options.keys())
            if 0 <= idx < len(keys):
                return keys[idx]
        except ValueError:
            pass
        print("  Invalid — try again.")


def _yes_no(prompt):
    ans = input(f"  {prompt} (y/n) [n]: ").strip().lower()
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# Main run logic
# ---------------------------------------------------------------------------

def run(dest, hour, day, pref, algo, event, accessible, staff, save_charts):
    print()
    print("=" * 62)
    print("  SMART PARKING FINDER & ROUTE ADVISOR")
    print("  CET251: Artificial Intelligence | El Sewedy University")
    print("=" * 62)
    print("\n  Training neural network occupancy model …")

    agent = ParkingAgent()
    agent.initialise(verbose=True)

    print("\n  Running search …")
    result = agent.recommend(
        destination     = dest,
        hour            = hour,
        weekday         = day,
        preference      = pref,
        algorithm       = algo,
        event           = int(event),
        accessible_only = accessible,
        staff_only      = staff,
    )

    print(agent.format_report(result))

    if save_charts:
        print("  Saving visualisations …")
        draw_campus_graph(result)
        draw_occupancy_chart(
            agent.predictor.predict_all(hour, day, int(event)),
            hour,
        )
        draw_algo_comparison(result["best"]["lot_id"], dest)
        print(f"\n  Charts saved to: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')}/")

    # Stretch: simulate the lot being full on arrival
    print()
    if _yes_no("Simulate the recommended lot being full on arrival?"):
        alt = agent.reroute(result, result["best"]["lot_id"])
        if alt:
            print(f"\n  NEW RECOMMENDATION: {alt['lot_label']}")
            print(f"  Occupancy : {alt['occ']:.1%}  |  Free: {alt['free_spaces']}  |  Walk: {alt['walk_min']} min")
        else:
            print("  No alternatives available — all nearby lots appear full.")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart Parking Finder — CET251 AI")
    parser.add_argument("--dest",       default=None, choices=list(BUILDINGS))
    parser.add_argument("--hour",       type=int, default=None)
    parser.add_argument("--day",        type=int, default=None)
    parser.add_argument("--pref",       default=None,
                        choices=["nearest", "fastest", "available"])
    parser.add_argument("--algo",       default="astar",
                        choices=["astar", "bfs", "dfs"])
    parser.add_argument("--event",      action="store_true")
    parser.add_argument("--accessible", action="store_true")
    parser.add_argument("--staff",      action="store_true")
    parser.add_argument("--no-charts",  action="store_true")
    args = parser.parse_args()

    # Interactive mode if any required arg is missing
    interactive = (args.dest is None or args.hour is None
                   or args.day is None or args.pref is None)

    if interactive:
        print("\n=== Smart Parking Finder — Interactive Mode ===")
        dest = args.dest or _pick_building()
        hour = args.hour if args.hour is not None else _pick_hour()
        day  = args.day  if args.day  is not None else _pick_day()
        pref = args.pref or _pick_option("Preference:", {
            "nearest":   "Nearest lot (minimise walking distance)",
            "fastest":   "Fastest walk (same as nearest on this graph)",
            "available": "Best availability (maximise free spaces)",
        })
        algo = _pick_option("Search algorithm:", {
            "astar": "A*  — optimal, guided by Euclidean heuristic (recommended)",
            "bfs":   "BFS — fewest hops",
            "dfs":   "DFS — depth-first (exploratory, not optimal)",
        })
        event      = _yes_no("Campus event today?")
        accessible = _yes_no("Need accessible parking?")
        staff      = _yes_no("Staff lots only?")
    else:
        dest, hour, day, pref, algo = args.dest, args.hour, args.day, args.pref, args.algo
        event, accessible, staff    = args.event, args.accessible, args.staff

    run(dest, hour, day, pref, algo, event, accessible, staff,
        save_charts=not args.no_charts)


if __name__ == "__main__":
    main()
