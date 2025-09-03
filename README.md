# Samudrapath

**Samudrapath** is a ship-routing toolkit and web application for route optimization in the Indian Ocean.
It produces multiple route alternatives (fuel-efficient, safe, and fastest) using a combination of grid-based pathfinding (Theta\*) to seed candidate routes and a multi-objective Genetic Algorithm (NSGA-II) to evolve a Pareto set of optimized routes while accounting for weather, waves, pirate-risk zones, ecological areas, and ship parameters.

---

# Table of contents

* [Quick start](#quick-start)
* [What this project does](#what-this-project-does)
* [How it works (high level)](#how-it-works-high-level)
* [Project layout](#project-layout-typical)
* [Requirements](#requirements)
* [Entire Algorithm notebook — `entire_algorithm.ipynb`](#entire-algorithm-notebook--entire_algorithmipynb)
* [Setup & run — Backend](#setup--run---backend)
* [Setup & run — Frontend](#setup--run---frontend)
* [Usage (API + UI)](#usage-api--ui)
* [Configuration & input formats](#configuration--input-formats)
* [Outputs & visualization](#outputs--visualization)

---

# Quick start

1. Clone the repository.
2. Start the backend (Flask) and then the frontend (npm).
3. Open the frontend UI (usually `http://localhost:3000`) and request routes by selecting start / goal and ship parameters.

---

# What this project does

* Generates candidate maritime routes between two points.
* Optimizes routes simultaneously for:

  * fuel consumption,
  * travel time,
  * safety/risk (wind, waves, piracy, ecological constraints).
* Returns a Pareto front of non-dominated solutions so users can choose trade-offs.

---

# How it works (high level)

1. Convert lat/lon to the project's grid coordinates and load environmental rasters (wind, waves, risk maps).
2. Use **Theta\*** on a navigability grid to generate initial candidate paths.
3. Seed an NSGA-II population by perturbing waypoints from Theta\*.
4. Evaluate each candidate for fuel, time, and safety using the ship model + environmental data.
5. Evolve the population with crossover and mutation, keeping the non-dominated solutions as the Pareto front.
6. Backend returns route geometries and metrics; frontend visualizes them and the Pareto trade-offs.

---

# Project layout

```
/backend
  ├─ server.py            # flask app entrypoint
  ├─ routes.py
  ├─ theta_star.py
  ├─ nsga2.py
  ├─ eval.py
  ├─ utils.py
  └─ requirements.txt
/frontend
  ├─ package.json
  ├─ public/
  └─ src/
      ├─ components/
      └─ pages/      # api client
/docs/
  └─ Samudrapath Documentation.pdf
/algorithm
  └─ entire_algorithm.ipynb
```

---

# Requirements

* Python 3.x
* Node.js + npm
* `requirements.txt`

---

## Entire Algorithm notebook — `entire_algorithm.ipynb`

> **Purpose:** The included Jupyter notebook is a runnable demo that demonstrates the core pipeline: generating baseline path(s) (Theta\*), seeding and running a multi-objective GA (NSGA-II), and visualizing the resulting Pareto front and routes. Use it to quickly validate algorithm behavior, test parameter changes, and visualize intermediate results (waypoints, fitness scores, Pareto sets).

### What you will find in the notebook

* A minimal, self-contained demo that:

  * sets up a small navigability grid or loads sample rasters/maps,
  * constructs an initial Theta\* path between start and goal,
  * seeds and runs NSGA-II with configurable GA parameters,
  * evaluates each route for fuel/time/risk and computes a Pareto front,
  * visualizes routes and the Pareto scatterplot (fuel vs time, colored by risk).
* Example configuration cells where you can edit start/goal, ship parameters, and GA settings.
* Helper visualization cells (matplotlib / folium / plotly) to inspect geometry and metrics.

### Prerequisites for running the notebook

1. Python virtual environment with project dependencies installed (see [Setup & run — Backend](#setup--run---backend)). Additionally install notebook tools if not present:

```bash
pip install jupyterlab jupyter matplotlib pandas notebook
```
<!-- 
2. Data files that the notebook expects (if any). The notebook may look for sample rasters or a small `data/` folder inside the repo. If you don't have those, the notebook often contains options to run with a synthetic/demo grid (check the top cells and set `use_synthetic_demo = True`).

> Tip: activate the same `.venv` used for the backend before launching Jupyter so notebook imports the same packages:

```bash
.venv\Scripts\activate   # or `source .venv/bin/activate` on macOS/Linux
jupyter lab              # or `jupyter notebook`
``` -->

### Quick steps to run the demo

1. Activate `.venv` (see above).
2. Launch Jupyter:

```bash
jupyter lab
# or
jupyter notebook
```

3. Open `entire_algorithm.ipynb`.
4. Run cells from top to bottom. The first cell(s) typically define configuration variables — set `start`, `goal`, `ship_params`, `ga_params`, and data paths as needed.
5. Inspect outputs:

   * a plotted baseline Theta\* route (grid visualization),
   * evolution progress values (it prints generation summaries),
   * final Pareto scatterplot and a list/table of candidate routes with metrics,
   * route geometries plotted on a map.

### What to look for (expected results)

* A valid Theta\* baseline path between the chosen points (no land/obstacle crossing).
* GA evolution logs or summarized metrics showing improvement or exploration across generations.
* A Pareto plot showing trade-offs (fuel vs time) and routes colored or labeled by risk.
* A small set of non-dominated routes (the Pareto front) from which you can pick different trade-offs.

### How to use the notebook to test changes

* **Change GA parameters** (`population`, `generations`, `crossover_prob`, `mutation_sigma`) and re-run the GA cell to see how solutions differ.
* **Modify ship parameters** (speed, efficiencies, SFOC) and observe effects on fuel/time metrics.
* **Instrument intermediate values**: insert print or plot statements in evaluation cells to inspect per-segment resistance, effective speed, or per-segment risk.
<!-- * **Swap environmental input**: point the notebook at different wind/wave rasters (if available) or toggle `use_synthetic_demo`. -->

<!-- ### Running the notebook headless (automated / CI)

You can run the notebook end-to-end (execute all cells) from the command line to produce HTML or to validate the demo automatically:

```bash
# execute notebook in place (writes output into the .ipynb)
jupyter nbconvert --to notebook --execute Demo_Final.ipynb --ExecutePreprocessor.timeout=600

# or produce an HTML report
jupyter nbconvert --to html --execute Demo_Final.ipynb --ExecutePreprocessor.timeout=600
# result: Demo_Final.html
```

This is useful for automated regression checks: after changes to `theta_star.py` or `nsga2.py`, run the notebook CI job to confirm demo still runs and produces expected metrics (you can add asserts in a final cell that fail the notebook if values stray outside expected ranges). -->

---
# Setup & run — Backend

From project root, run the following commands.

```bash
# create virtual environment
python3 -m venv .venv

# activate the virtualenv
# On Windows (PowerShell or cmd as in your instructions):
.venv\Scripts\activate

# On macOS / Linux:
# source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the Flask app
flask --app server run
```

<!-- ```bash
flask --app server --debug run --host=0.0.0.0 --port=5000
``` -->

---

# Setup & run — Frontend

From the frontend directory:

```bash
cd frontend
npm install
npm start
```

`npm start` typically runs a development server (often at `http://localhost:3000`). Make sure the frontend is configured to call your backend host/port (e.g., `http://localhost:5000`).

---

# Usage (API + UI)

## UI

1. Start backend and frontend.
2. In the web UI:

   * Set start and destination (lat/lon or pick on the map).
   * Choose ship type or enter ship parameters.
   * Optionally choose environmental data inputs or allow automatic forecast fetching.
   * Request route(s) and review the Pareto plot and map overlays.

<!-- ## Example API request (JSON)

POST `/api/route` (example payload):

```json
{
  "start": [9.0, 78.0],
  "goal": [10.5, 80.2],
  "ship": {
    "type": "cargo",
    "V0": 12.5,
    "D": 15000,
    "n_h": 0.95,
    "n_s": 0.6,
    "n_e": 0.38,
    "c_sfoc": 180
  },
  "env": {
    "wind_speed_raster": "path_or_id",
    "wind_dir_raster": "path_or_id",
    "wave_height_raster": "path_or_id",
    "pirate_risk_raster": "path_or_id"
  },
  "ga_params": {
    "population": 100,
    "generations": 100,
    "crossover_prob": 0.8,
    "mutation_sigma": 0.01
  }
}
```

A successful response typically includes:

* list of candidate routes (waypoint arrays)
* per-route metrics (fuel, time, risk)
* Pareto front summary

--- -->

# Configuration & input formats

* **Grid/binary map**: 2D raster/grid marking navigable vs blocked cells.
* **Environmental rasters**: wind speed/direction, wave height/direction aligned to the grid.
* **Piracy / risk maps**: raster with numeric risk levels.
* **Ship parameters**: JSON fields for speed, displacement, efficiencies, SFOC, etc.
* See `docs/Samudrapath Documentation.pdf` for precise formats and units used in the code.

---

# Outputs & visualization

* Route geometries (lat/lon waypoints).
* Metrics for each route: fuel (units), travel time (hours), aggregated risk score.
* Map overlays: routes, wind/wave fields, risk zones.
* Pareto visual: fuel vs time vs (color-coded) risk for non-dominated routes.

---

<!-- # Troubleshooting

* **Flask won't start**: ensure `.venv` is active and `requirements.txt` installed; confirm `hello.py` exists or set `FLASK_APP` correctly.
* **Missing GDAL/rasterio errors**: install the GDAL system package for your OS before installing `rasterio`.
* **Frontend can't reach backend**: check backend host/port and CORS settings in Flask.
* **Slow GA runs**: reduce population / generations for development, run heavy experiments offline or on a machine with more CPU.

--- -->

# Contributing

1. Fork the repo.
2. Create a branch: `git checkout -b feat/your-feature`.
3. Add tests and documentation updates.
4. Submit a pull request with description and reproducible examples.

---

Thank You :)
