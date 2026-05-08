# Smart Parking Finder — How to Run
**CET251: Artificial Intelligence | El Sewedy University of Technology**

---

## Requirements
- Python 3.10 or newer
- pip

---

## Step 1 — Install dependencies

Open a terminal, navigate to the project folder, and run:

```
pip install -r requirements.txt
```

This installs: `networkx`, `matplotlib`, `scikit-learn`, `numpy`

---

## Step 2 — Run the program

### Option A: Interactive mode (recommended)

Just run:

```
python main.py
```

The program will ask you:
1. Which building you are going to
2. What hour you are arriving (0–23)
3. Which day of the week (0=Mon … 6=Sun)
4. Your preference (nearest / fastest / available)
5. Which search algorithm to use (A* / BFS / DFS)
6. Whether there is a campus event today
7. Whether you need accessible parking
8. Whether you want staff lots only

---

### Option B: Command-line arguments (skip all prompts)

```
python main.py --dest B_SCI --hour 9 --day 1 --pref available
```

More examples:

```bash
# Library, 2 PM Wednesday, nearest lot, BFS search
python main.py --dest B_LIB --hour 14 --day 2 --pref nearest --algo bfs

# Sports Complex, Friday evening, campus event today
python main.py --dest B_GYM --hour 18 --day 4 --pref fastest --event

# Accessible parking only, Admin Building, Monday morning
python main.py --dest B_ADMIN --hour 8 --day 0 --pref available --accessible

# Skip saving charts
python main.py --dest B_ENG --hour 10 --day 1 --pref nearest --no-charts
```

---

## Destination building IDs

| ID       | Building          |
|----------|-------------------|
| B_MAIN   | Main Hall         |
| B_SCI    | Science Block     |
| B_ENG    | Engineering       |
| B_LIB    | Library           |
| B_GYM    | Sports Complex    |
| B_ADMIN  | Admin Building    |

---

## Output

After running, the program prints:
- The recommended parking lot
- Predicted occupancy % and estimated free spaces
- The walking route (node by node)
- A comparison of BFS vs DFS vs A* (distance and nodes explored)
- Alternative lots ranked by score

Three PNG charts are saved to the `output/` folder:

| File                    | Contents                                      |
|-------------------------|-----------------------------------------------|
| `campus_graph.png`      | Campus graph with recommended lot and route highlighted |
| `occupancy_chart.png`   | Bar chart of ML-predicted occupancy per lot   |
| `algo_comparison.png`   | BFS vs DFS vs A*: nodes explored + distance   |

---

## Project file structure

```
SmartParking/
├── main.py              ← run this
├── campus_graph.py      ← campus graph: nodes, edges, parking lot data
├── search_algorithms.py ← BFS, DFS, A* implementations
├── occupancy_model.py   ← MLP neural network occupancy predictor
├── parking_agent.py     ← AI agent: combines ML + search + scoring
├── visualise.py         ← matplotlib/networkx charts
├── requirements.txt     ← dependencies
├── HOW_TO_RUN.md        ← this file
└── output/              ← generated charts saved here
```

---

## What each file does (for the report)

| File                  | Role |
|-----------------------|------|
| `campus_graph.py`     | Defines the environment: 5 parking lots, 6 buildings, 8 junctions, 23 weighted edges |
| `search_algorithms.py`| BFS (fewest hops), DFS (depth-first), A* (optimal, Euclidean heuristic) |
| `occupancy_model.py`  | Generates synthetic training data; trains MLPRegressor (64→32 hidden layers) on 5 features: hour, weekday, lot ID, class density, event flag |
| `parking_agent.py`    | The agent: predicts occupancy → searches routes → scores lots → returns best recommendation with explanation |
| `visualise.py`        | Campus graph, occupancy bar chart, algorithm comparison chart |
| `main.py`             | Interactive CLI + command-line argument support |
