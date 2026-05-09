# Smart Parking Finder & Route Advisor

**Course:** CET251 — Artificial Intelligence  
**University:** El Sewedy University of Technology  
**Tagline:** Find the best parking spot and the fastest route to your destination.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Project Structure](#3-project-structure)
4. [Installation & Setup](#4-installation--setup)
5. [How to Run](#5-how-to-run)
6. [Campus Map & Graph](#6-campus-map--graph)
7. [AI Concepts Implemented](#7-ai-concepts-implemented)
   - [Agents & Environments](#71-agents--environments)
   - [BFS — Breadth-First Search](#72-bfs--breadth-first-search)
   - [DFS — Depth-First Search](#73-dfs--depth-first-search)
   - [A\* Search](#74-a-search)
   - [Algorithm Comparison](#75-algorithm-comparison)
8. [Neural Network Occupancy Predictor](#8-neural-network-occupancy-predictor)
9. [Scoring & Decision Logic](#9-scoring--decision-logic)
10. [Outputs & Charts](#10-outputs--charts)
11. [Stretch Features](#11-stretch-features)
12. [File-by-File Reference](#12-file-by-file-reference)
13. [Sample Output](#13-sample-output)

---

## 1. Project Overview

Students, staff, and visitors often spend time searching for parking during busy hours. The closest parking area may already be full, reserved, or far from the actual destination once walking distance is considered.

This project builds an AI assistant that:

- Takes a **destination building**, **arrival time**, and **user preference** as input
- Uses a **neural network** to predict how full each parking lot will be at that time
- Runs **BFS, DFS, and A\*** on a campus walking graph to find the best route from each lot to the destination
- **Scores and ranks** all eligible parking lots using a composite formula
- Returns a **recommendation with explanation**, a step-by-step walking route, an algorithm comparison, and alternative options
- Saves **three PNG charts** to the `output/` folder every run

---

## 2. Features

| Feature | Description |
|---|---|
| Destination input | Choose from 6 campus buildings |
| Arrival time | Any hour 0–23 |
| Day of week | Weekday/weekend affects ML predictions |
| Preference modes | Nearest lot / Fastest walk / Best availability |
| Algorithm selection | A\* (recommended), BFS, or DFS |
| Campus event flag | Boosts predicted occupancy when an event is on |
| Accessible mode | Filters to lots with designated accessible spaces |
| Staff mode | Filters to lots with reserved staff spaces |
| Live re-routing | If the recommended lot is full on arrival, picks the next best option |
| Charts | Campus graph, occupancy bar chart, algorithm comparison PNG |

---

## 3. Project Structure

```
SmartParking/
│
├── main.py               ← Entry point. Run this file.
├── campus_graph.py       ← Campus map: nodes, edges, parking lot data
├── search_algorithms.py  ← BFS, DFS, and A* implementations
├── occupancy_model.py    ← Neural network occupancy predictor + data generation
├── parking_agent.py      ← AI agent: combines ML + search + scoring + reporting
├── visualise.py          ← Matplotlib/NetworkX chart generation
│
├── requirements.txt      ← Python dependencies
├── README.md             ← This file
│
└── output/               ← Generated charts saved here (created automatically)
    ├── campus_graph.png
    ├── occupancy_chart.png
    └── algo_comparison.png
```

---

## 4. Installation & Setup

### Requirements

- Python **3.10 or newer**
- pip

### Step 1 — Install dependencies

Open a terminal in the `SmartParking/` folder and run:

```bash
pip install -r requirements.txt
```

This installs the following libraries:

| Library | Version | Purpose |
|---|---|---|
| `networkx` | ≥ 3.0 | Campus graph structure and drawing |
| `matplotlib` | ≥ 3.7 | Saving PNG charts |
| `scikit-learn` | ≥ 1.3 | MLPRegressor neural network + StandardScaler |
| `numpy` | ≥ 1.24 | Numerical arrays for ML data |
| `pandas` | ≥ 2.0 | Tabular dataset management for training data |

---

## 5. How to Run

### Interactive mode (recommended — no arguments needed)

```bash
python main.py
```

The program will prompt you step-by-step:

```
  Available destinations:
    1. Main Hall       [B_MAIN]
    2. Science Block   [B_SCI]
    3. Engineering     [B_ENG]
    4. Library         [B_LIB]
    5. Sports Complex  [B_GYM]
    6. Admin Building  [B_ADMIN]

  Arrival hour (0-23): 9
  Weekday: 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri 5=Sat 6=Sun
  Weekday number: 1

  Preference:
    1. nearest
    2. fastest
    3. available

  Search algorithm:
    1. A*  — optimal, guided by Euclidean heuristic (recommended)
    2. BFS — fewest hops
    3. DFS — depth-first (exploratory, not optimal)

  Campus event today? (y/n): n
  Need accessible parking? (y/n): n
  Staff lots only? (y/n): n
```

---

### Command-line mode (skip all prompts)

Pass arguments directly to avoid the interactive menu:

```bash
python main.py --dest B_SCI --hour 9 --day 1 --pref available
```

#### All available arguments

| Argument | Type | Description |
|---|---|---|
| `--dest` | string | Destination building ID (see table below) |
| `--hour` | integer | Arrival hour, 0–23 |
| `--day` | integer | Weekday: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun |
| `--pref` | string | `nearest` \| `fastest` \| `available` |
| `--algo` | string | `astar` (default) \| `bfs` \| `dfs` |
| `--event` | flag | Add this flag if there is a campus event today |
| `--accessible` | flag | Only show lots with accessible parking spaces |
| `--staff` | flag | Only show lots with reserved staff spaces |
| `--no-charts` | flag | Skip saving PNG charts to `output/` |

#### Destination building IDs

| ID | Building |
|---|---|
| `B_MAIN` | Main Hall |
| `B_SCI` | Science Block |
| `B_ENG` | Engineering |
| `B_LIB` | Library |
| `B_GYM` | Sports Complex |
| `B_ADMIN` | Admin Building |

#### More usage examples

```bash
# Library at 2 PM on Wednesday, prefer nearest lot, use BFS
python main.py --dest B_LIB --hour 14 --day 2 --pref nearest --algo bfs

# Sports Complex on Friday evening with a campus event
python main.py --dest B_GYM --hour 18 --day 4 --pref fastest --event

# Admin Building on Monday morning, accessible parking required
python main.py --dest B_ADMIN --hour 8 --day 0 --pref available --accessible

# Staff parking only, Science Block, Tuesday 10 AM
python main.py --dest B_SCI --hour 10 --day 1 --pref nearest --staff

# Run without saving charts
python main.py --dest B_LIB --hour 11 --day 3 --pref available --no-charts
```

---

## 6. Campus Map & Graph

The campus is modelled as a **weighted undirected graph** with 19 nodes and 23 edges.

### Node types

| Type | Count | Description |
|---|---|---|
| Buildings | 6 | Destination points (B_MAIN, B_SCI, B_ENG, B_LIB, B_GYM, B_ADMIN) |
| Parking lots | 5 | Starting points for route search (P_NORTH, P_EAST, P_CENTRAL, P_WEST, P_SOUTH) |
| Junctions | 8 | Intermediate footpath intersections (J1–J8) |

### Parking lot details

| Lot ID | Label | Capacity | Staff reserved | Accessible spaces |
|---|---|---|---|---|
| `P_NORTH` | North Lot | 80 | 10 | 4 |
| `P_EAST` | East Lot | 60 | 8 | 3 |
| `P_CENTRAL` | Central Lot | 40 | 5 | 2 |
| `P_WEST` | West Lot | 50 | 8 | 3 |
| `P_SOUTH` | South Lot | 70 | 6 | 4 |

### Edge weights

All edge weights represent **walking distances in metres**. For example:

- `P_CENTRAL → J2`: 50 m
- `J3 → B_SCI`: 60 m
- `J6 → B_LIB`: 50 m
- `J1 → B_MAIN`: 100 m

The graph is stored using **NetworkX** (`nx.Graph`). It is built once at import time in `campus_graph.py` and reused by all other modules.

Each node also has a **grid position** `(x, y)` used as coordinates by the A\* heuristic and by the NetworkX layout when drawing the campus graph chart.

---

## 7. AI Concepts Implemented

### 7.1 Agents & Environments

The `ParkingAgent` class in `parking_agent.py` is a **rational AI agent**:

- **Environment:** The campus graph with 5 parking lots, 6 buildings, and 8 junctions. Occupancy levels change with time of day, weekday, and campus events.
- **Percepts:** Destination building, arrival hour, day of week, user preference, event flag, and user type flags.
- **Actions:** Query the neural network for occupancy, run search algorithms for routes, compute composite scores, return a ranked recommendation with explanation, and re-route if the top lot is full.

This directly implements the **PEAS** framework (Performance measure, Environment, Actuators, Sensors) taught in CET251.

---

### 7.2 BFS — Breadth-First Search

**File:** `search_algorithms.py` → `bfs(source, target)`

**How it works:**  
BFS uses a `deque` (double-ended queue) as its data structure. It expands nodes **level by level** — first all direct neighbours of the start, then their neighbours, and so on. This guarantees finding the path with the **fewest hops** (edges).

**Data structure:** `deque` — nodes enter the back and leave from the front (FIFO order), enforcing level-by-level expansion.

**Limitation:** Fewest hops ≠ shortest distance. A path using 2 long edges can be much farther than one using 4 short edges. BFS does not consider edge weights.

**When it is useful here:** When the user just wants the most direct connection with the least number of footpath segments, regardless of distance.

```
Queue state example (P_NORTH → B_SCI):
  Level 1: [J1]
  Level 2: [B_MAIN, J2]
  Level 3: [J3, J4, J6]
  Level 4: [B_SCI] ← found!
```

---

### 7.3 DFS — Depth-First Search

**File:** `search_algorithms.py` → `dfs(source, target)`

**How it works:**  
DFS uses a `stack` (list). It follows one path as deep as possible before backtracking. Because it pops from the top (LIFO order), it always dives into the most recently discovered node rather than exploring broadly.

**Data structure:** `stack` — nodes are pushed and popped from the same end (LIFO order).

**Limitation:** DFS is **not optimal**. It may find a path that is much longer than necessary, because it commits to one direction without knowing where it leads.

**Why it is included:** The spec requires BFS, DFS, and A\* so that they can be compared. DFS is valuable for showing *why* A\* is better — its node count and path distance are typically the worst of the three.

---

### 7.4 A\* Search

**File:** `search_algorithms.py` → `astar(source, target)`

**How it works:**  
A\* uses a **priority queue** (`heapq`). It always expands the node with the lowest value of:

```
f(n) = g(n) + h(n)
```

Where:
- `g(n)` = actual walking distance from start to node `n` (known exactly)
- `h(n)` = **heuristic estimate** of distance from `n` to the goal

**The heuristic used:**  
Euclidean (straight-line) distance between the two nodes' grid positions, scaled by 50 metres per grid unit:

```python
h(a, b) = sqrt((ax - bx)² + (ay - by)²) × 50
```

This is **admissible** — it never overestimates the true distance, because a straight line is always the shortest possible path. An admissible heuristic guarantees that A\* finds the **optimal** (shortest distance) path.

**Why A\* is better than BFS and DFS:**  
By using the heuristic, A\* avoids expanding nodes that are in the wrong direction. It focuses its search toward the goal, exploring far fewer nodes than BFS or DFS while still finding the shortest weighted path.

---

### 7.5 Algorithm Comparison

Every search automatically runs all three algorithms on the recommended lot → destination route and prints a side-by-side table:

```
  ALGORITHM COMPARISON (same lot → destination):
  Algorithm  Distance (m)     Nodes explored     Walk (min)
  --------   ------------     --------------     ----------
  BFS        300              7                  3.8
  DFS        530              11                 6.6
  A*         300              4                  3.8
```

A comparison chart (`algo_comparison.png`) is also saved showing both metrics as bar charts.

---

## 8. Neural Network Occupancy Predictor

**File:** `occupancy_model.py`

### Purpose

Since the system has no live sensors, it uses a **neural network trained on synthetic data** to predict how full each parking lot will be at a given time.

### Training data generation

The function `generate_data(n_days=180)` simulates **21,600 rows** of realistic occupancy observations (180 days × 24 hours × 5 lots) and stores them in a **pandas DataFrame**:

| Column | Type | Description |
|---|---|---|
| `hour` | int | Hour of day, 0–23 |
| `weekday` | int | 0 (Mon) to 6 (Sun) |
| `lot_id` | int | Lot encoded as 0–4 (required by neural network) |
| `class_density` | int | Estimated concurrent classes nearby, 0–10 |
| `event` | int | Campus event flag, 0 or 1 |
| `occupancy_pct` | float | Target: fraction of lot that is occupied, 0.0–1.0 |

**Why pandas?**  
The dataset is naturally tabular (one row per observation, one column per feature). Pandas makes it easy to build the dataset row-by-row from a list of dicts, compute summary statistics with `groupby("lot_id")["occupancy_pct"].mean()`, and extract feature/target columns cleanly.

**How occupancy is simulated:**  
```
occupancy = lot_base_popularity × hour_weight × weekday_weight + event_boost + noise
```

- `hour_weight`: peaks at 0.95 around 9–10 AM, drops to 0.02 at night
- `weekday_weight`: 1.0 Mon–Wed, 0.95 Thu, 0.80 Fri, 0.30 Sat, 0.15 Sun
- `lot_base_popularity`: Central Lot = 0.95 (most popular), South Lot = 0.55 (least popular)
- `event_boost`: +0.15 during event hours (10 AM–8 PM)
- `noise`: Gaussian noise with σ = 0.06

### Neural network architecture

| Component | Value |
|---|---|
| Model type | `MLPRegressor` (Multi-Layer Perceptron) |
| Hidden layers | 64 neurons → 32 neurons |
| Activation | ReLU |
| Optimiser | Adam |
| Output | Single neuron (occupancy fraction, clipped to 0–1) |
| Scaler | `StandardScaler` (zero mean, unit variance per feature) |
| Early stopping | Yes — stops if validation loss does not improve for 20 iterations |
| Train/test split | 80% train / 20% test |

**Why StandardScaler?**  
Without scaling, `hour` (range 0–23) would dominate over `event` (range 0–1) simply because its numbers are larger. Scaling ensures all features contribute equally to the network's learning.

### Training metrics printed at startup

```
  [Data] Simulated 21,600 occupancy observations (pandas DataFrame)
         Shape: 21600 rows × 6 columns
         Columns: ['hour', 'weekday', 'lot_id', 'class_density', 'event', 'occupancy_pct']
         Average occupancy per lot:
           P_CENTRAL   : 43.1% average occupancy
           P_EAST      : 31.7% average occupancy
           P_NORTH     : 37.2% average occupancy
           P_SOUTH     : 25.0% average occupancy
           P_WEST      : 27.4% average occupancy

  [ML] Neural network training complete.
       MAE : 0.0412  (avg error in occupancy fraction, lower = better)
       R²  : 0.9521  (1.0 = perfect, measures how well model fits data)
       Data: 17280 train / 4320 test samples
```

**MAE of ~0.04** means predictions are off by about 4 percentage points on average — acceptable for a simulated system.  
**R² of ~0.95** means the model explains 95% of the variation in occupancy across different times and lots.

---

## 9. Scoring & Decision Logic

**File:** `parking_agent.py` → `_score(occ, distance, preference)`

Once the neural network predicts occupancy and A\* (or BFS/DFS) finds the route distance for each lot, every candidate lot is given a **composite score** between 0 and 1:

```
score = w_free × (1 − occupancy) + w_dist × (1 − distance / 1200)
```

The weights `w_free` and `w_dist` shift based on the user's stated preference:

| Preference | w_free | w_dist | What it means |
|---|---|---|---|
| `available` | 0.80 | 0.20 | Maximise free spaces — distance matters less |
| `nearest` | 0.25 | 0.75 | Minimise walking — availability matters less |
| `fastest` | 0.25 | 0.75 | Same as nearest on a walking graph |

Lots are sorted by score descending. The top-ranked lot is the recommendation; the rest become alternatives.

The agent also runs all three algorithms on the winning lot so the comparison table always reflects the same route.

---

## 10. Outputs & Charts

### CLI report (printed every run)

```
──────────────────────────────────────────────────────────────
  SMART PARKING FINDER — RECOMMENDATION REPORT
  CET251: Artificial Intelligence | El Sewedy University
──────────────────────────────────────────────────────────────
  Destination : Science Block
  Arrival     : Tuesday  09:00
  Preference  : available
  Algorithm   : ASTAR
──────────────────────────────────────────────────────────────
  ★  RECOMMENDED: Central Lot
     Predicted occupancy : 78.4%
     Estimated free spots: 9
     Walking distance    : 190 m
     Walking time        : 2.4 min

  ROUTE (ASTAR):
     Central Lot → Junction 2 → Junction 3 → Science Block
     Nodes explored: 5
──────────────────────────────────────────────────────────────
  ALGORITHM COMPARISON (same lot → destination):
  Algorithm  Distance (m)     Nodes explored     Walk (min)
  --------   ------------     --------------     ---------
  BFS        190              7                  2.4
  DFS        490              11                 6.1
  A*         190              5                  2.4
──────────────────────────────────────────────────────────────
  ALTERNATIVE OPTIONS:
  #2  North Lot       occ=74%  free= 21  walk=4.5 min
  #3  East Lot        occ=62%  free= 23  walk=3.0 min
  #4  West Lot        occ=55%  free= 23  walk=7.9 min
  #5  South Lot       occ=48%  free= 36  walk=9.5 min
──────────────────────────────────────────────────────────────
```

### PNG charts saved to `output/`

| File | Contents |
|---|---|
| `campus_graph.png` | Full campus graph. Recommended lot highlighted in **red**, destination in **green**, A\*/BFS/DFS route edges in **bright green**. All edge weights labelled. |
| `occupancy_chart.png` | Bar chart of ML-predicted occupancy for all 5 lots at the requested hour. Bars are colour-coded: green < 60%, orange 60–85%, red > 85%. Each bar shows the real percentage and free/total spaces. Minimum visible bar height ensures early-morning hours are readable. |
| `algo_comparison.png` | Side-by-side bars: left = nodes explored, right = path distance. Compares BFS, DFS, and A\* on the same route. |

---

## 11. Stretch Features

All three stretch features from the project spec are implemented:

### Accessible parking mode
Pass `--accessible` (CLI) or answer `y` when prompted. The agent filters out any lot where `accessible_spaces == 0`, ensuring only lots with designated accessible bays are considered.

### Staff/student priority mode
Pass `--staff` (CLI) or answer `y` when prompted. The agent filters to lots that have `reserved_staff > 0`. Students without the flag see all lots with no restriction.

### Live re-routing
At the end of every run, the program asks:

```
  Simulate the recommended lot being full on arrival? (y/n):
```

If you answer `y`, the agent drops the recommended lot and returns the next available alternative — simulating what happens if you arrive and find the lot completely full.

---

## 12. File-by-File Reference

### `campus_graph.py`
Defines the entire campus environment. Contains:
- `NODES` — dictionary of all 19 nodes with type, label, and grid position
- `PARKING_LOTS` — capacity, reserved staff spaces, and accessible spaces per lot
- `EDGES` — 23 undirected edges with walking distances in metres
- `LOT_ID_MAP` — maps lot name strings to integers (0–4) for the neural network
- `CAMPUS_GRAPH` — a built `nx.Graph` instance, ready to use at import time
- `build_graph()` — constructs the NetworkX graph from NODES and EDGES

### `search_algorithms.py`
Three search algorithms on the campus graph:
- `bfs(source, target)` — breadth-first search using a `deque`
- `dfs(source, target)` — depth-first search using a `stack`
- `astar(source, target)` — A\* using `heapq` and a Euclidean heuristic
- `compare_all(source, target)` — runs all three and returns results in a dict
- `_euclidean(a, b)` — internal heuristic: straight-line distance × 50 m/unit
- `_path_cost(G, path)` — sums edge weights along a node list

Each function returns a dict with keys: `path`, `total_distance`, `nodes_explored`, `algorithm`, `walking_minutes`.

### `occupancy_model.py`
Neural network occupancy predictor:
- `generate_data(n_days)` — simulates occupancy data, returns `(X, y, DataFrame)`
- `OccupancyPredictor` class:
  - `.train(n_days, verbose)` — generates data, fits MLP + StandardScaler, prints metrics
  - `.predict(lot_name, hour, weekday, event)` — returns occupancy fraction for one lot
  - `.predict_all(hour, weekday, event)` — returns dict of all lot predictions

### `parking_agent.py`
The core AI agent:
- `_score(occ, distance, preference)` — weighted composite scoring function
- `ParkingAgent` class:
  - `.initialise(verbose)` — trains the neural network (call once before `recommend`)
  - `.recommend(destination, hour, weekday, ...)` — runs full pipeline, returns result dict
  - `.format_report(result)` — formats result dict as a printable string
  - `.reroute(result, full_lot_id)` — stretch feature: picks next best alternative

### `visualise.py`
Three matplotlib charts:
- `draw_campus_graph(result, filename)` — NetworkX graph with route highlighted
- `draw_occupancy_chart(occupancy, hour, filename)` — colour-coded bar chart per lot
- `draw_algo_comparison(source, target, filename)` — BFS vs DFS vs A\* side-by-side

All charts are saved to the `output/` subfolder (created automatically).

### `main.py`
Entry point only — no business logic:
- Parses CLI arguments with `argparse`
- Falls back to interactive prompts if arguments are missing
- Calls `agent.initialise()`, `agent.recommend()`, `agent.format_report()`
- Calls all three `visualise.py` functions
- Offers the live re-routing simulation at the end

---

## 13. Sample Output

Running:
```bash
python main.py --dest B_SCI --hour 9 --day 1 --pref available --algo astar
```

Produces the CLI report shown in Section 10, plus three charts in `output/`:

- `campus_graph.png` — shows the recommended lot in red with the A\* route highlighted
- `occupancy_chart.png` — shows all 5 lots at 09:00, Central Lot in orange/red (high occupancy)
- `algo_comparison.png` — shows A\* exploring 5 nodes vs BFS's 7 and DFS's 11 for the same route
