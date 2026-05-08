"""
visualise.py
------------
Simple matplotlib + networkx visualisations.

Three charts saved to the output/ folder:
  1. campus_graph.png      — campus walking graph with highlighted route
  2. occupancy_chart.png   — bar chart of ML-predicted occupancy per lot
  3. algo_comparison.png   — BFS vs DFS vs A* (nodes explored + path distance)

Per the project spec: keep graphs simple and functional.
Focus is on the AI logic, not visual polish.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from campus_graph      import CAMPUS_GRAPH, NODES, PARKING_LOTS
from search_algorithms import compare_all

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _ensure_output():
    os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Campus graph with highlighted route
# ---------------------------------------------------------------------------
def draw_campus_graph(result=None, filename="campus_graph.png"):
    """
    Draw the campus walking graph.
    If result is provided, highlights the recommended lot and A* route.
    """
    _ensure_output()
    G   = CAMPUS_GRAPH
    pos = {n: data["pos"] for n, data in NODES.items()}

    fig, ax = plt.subplots(figsize=(10, 8))

    # All edges (grey)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=1.2, alpha=0.7)

    # Edge weight labels
    edge_labels = {(u, v): f"{d['weight']}m" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=6, font_color="#888888")

    # Highlight route edges
    if result:
        route_path  = result["best"]["route"]["path"]
        route_edges = list(zip(route_path, route_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, ax=ax,
                               edge_color="#2ecc71", width=3.5, alpha=0.9)

    # Draw nodes
    node_colours = []
    for node in G.nodes():
        ntype = NODES[node]["type"]
        if result and node == result["best"]["lot_id"]:
            node_colours.append("#e74c3c")   # recommended lot — red
        elif result and node == result["destination"]:
            node_colours.append("#2ecc71")   # destination — green
        elif ntype == "lot":
            node_colours.append("#e67e22")   # other lots — orange
        elif ntype == "building":
            node_colours.append("#3498db")   # buildings — blue
        else:
            node_colours.append("#95a5a6")   # junctions — grey

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=350, edgecolors="white", linewidths=0.8)

    # Labels
    labels = {n: NODES[n]["label"].replace(" ", "\n") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6.5)

    # Legend
    legend = [
        mpatches.Patch(color="#e74c3c", label="Recommended lot"),
        mpatches.Patch(color="#2ecc71", label="Destination / Route"),
        mpatches.Patch(color="#e67e22", label="Other parking lot"),
        mpatches.Patch(color="#3498db", label="Building"),
        mpatches.Patch(color="#95a5a6", label="Junction"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8)

    title = "Campus Walking Graph"
    if result:
        title += f"  —  Route to {result['dest_label']}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [Viz] Saved: {path}")


# ---------------------------------------------------------------------------
# 2. Occupancy bar chart
# ---------------------------------------------------------------------------
def draw_occupancy_chart(occupancy, hour, filename="occupancy_chart.png"):
    """
    Bar chart of predicted occupancy fraction for all parking lots.

    Parameters
    ----------
    occupancy : dict  {lot_id: fraction}
    hour      : int   arrival hour (for chart title)
    """
    _ensure_output()

    lots     = sorted(occupancy)
    labels   = [NODES[l]["label"] for l in lots]
    occ      = [occupancy[l] * 100 for l in lots]
    free     = [round(PARKING_LOTS[l]["capacity"] * (1 - occupancy[l])) for l in lots]
    capacity = [PARKING_LOTS[l]["capacity"] for l in lots]

    colours = ["#e74c3c" if o > 85 else "#e67e22" if o > 60 else "#2ecc71" for o in occ]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Minimum visible bar height of 4% so bars are never invisible
    display_occ = [max(o, 4.0) for o in occ]
    bars = ax.bar(labels, display_occ, color=colours, width=0.55,
                  edgecolor="white", linewidth=0.8)

    # Annotate each bar with occ% on top and "X free / capacity" below it
    for bar, real_occ, f, cap in zip(bars, occ, free, capacity):
        x = bar.get_x() + bar.get_width() / 2
        top = bar.get_height()
        # Percentage label above bar
        ax.text(x, top + 1.5, f"{real_occ:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        # Free spaces label further above
        ax.text(x, top + 5.5, f"{f}/{cap} free",
                ha="center", va="bottom", fontsize=8, color="#555555")

    ax.axhline(85, color="#e74c3c", linestyle="--", linewidth=1,
               alpha=0.6, label="85% threshold (high occupancy)")
    ax.set_ylim(0, 118)
    ax.set_ylabel("Predicted Occupancy (%)")
    ax.set_title(f"ML-Predicted Lot Occupancy  —  {hour:02d}:00", fontweight="bold", fontsize=12)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [Viz] Saved: {path}")


# ---------------------------------------------------------------------------
# 3. Algorithm comparison chart
# ---------------------------------------------------------------------------
def draw_algo_comparison(source, target, filename="algo_comparison.png"):
    """
    Side-by-side bars comparing BFS, DFS, and A*:
      Left  — nodes explored
      Right — path distance in metres
    """
    _ensure_output()

    results = compare_all(source, target)
    if not results:
        return

    names    = list(results)
    explored = [results[n]["nodes_explored"]  for n in names]
    dists    = [results[n]["total_distance"]   for n in names]
    colours  = ["#3498db", "#9b59b6", "#2ecc71"][:len(names)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.bar(names, explored, color=colours, width=0.5)
    ax1.set_title("Nodes Explored", fontweight="bold")
    ax1.set_ylabel("Count")
    for i, v in enumerate(explored):
        ax1.text(i, v + 0.2, str(v), ha="center", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.bar(names, dists, color=colours, width=0.5)
    ax2.set_title("Path Distance (m)", fontweight="bold")
    ax2.set_ylabel("Metres")
    for i, v in enumerate(dists):
        ax2.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    src_lbl = NODES[source]["label"]
    tgt_lbl = NODES[target]["label"]
    fig.suptitle(f"Algorithm Comparison: {src_lbl} → {tgt_lbl}", fontweight="bold", fontsize=11)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [Viz] Saved: {path}")
