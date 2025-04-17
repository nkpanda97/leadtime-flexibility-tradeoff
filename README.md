# Impact of Lead Time on Aggregate EV Flexibility for Congestion Management Services

This repository supplements the paper _"Impact of Lead Time on Aggregate EV Flexibility for Congestion Management Services."_ It contains the experimental setup, simulation scripts, and visualisation code used in the study.
Sure! Here’s the **"How to Cite"** section for your `README.md`:

---

## 📚 How to Cite

If you use this code or build upon it, please cite the following:

### 📄 Paper  
> **"Impact of Lead Time on Aggregate EV Flexibility for Congestion Management Services"**  
> *Nanda Kishor Panda, Peter Palensky, Simon H. Tindemans*, *IEEE PowerTech 2025, Kiel, Germany*, *2025*.
> *Preprint*: [arxiv](https://arxiv.org/pdf/2501.15946) 
> DOI: [https://doi.org/10.48550/arXiv.2501.15946](https://doi.org/10.48550/arXiv.2501.15946)#### BibTeX
```bibtex
@misc{panda2025impactleadtimeaggregate,
      title={Impact of Lead Time on Aggregate EV Flexibility for Congestion Management Services}, 
      author={Nanda Kishor Panda and Peter Palensky and Simon H. Tindemans},
      year={2025},
      eprint={2501.15946},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2501.15946}, 
}
```
### 💻 Code (via Zenodo)  
This code is archived with a DOI via [Zenodo](https://zenodo.org/). Please cite it as:

> Nanda Kishor Panda, Peter Palensky, Simon H. Tindemans. (2025). Impact of Lead Time on Aggregate EV Flexibility for Congestion Management Services.
> Software: Zenodo.  
> DOI:[https://doi.org/10.5281/zenodo.XXXXXXX ](https://doi.org/10.5281/zenodo.15236426)

#### BibTeX
```bibtex
@software{nanda_kishor_panda_2025_15236427,
  author       = {Nanda Kishor Panda},
  title        = {nkpanda97/leadtime-flexibility-tradeoff: Version
                   2025.1
                  },
  month        = apr,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {production},
  doi          = {10.5281/zenodo.15236427},
  url          = {https://doi.org/10.5281/zenodo.15236427},
  swhid        = {swh:1:dir:33854ff2eea0667b7db9349ad01f43f6a0c9bba6
                   ;origin=https://doi.org/10.5281/zenodo.15236426;vi
                   sit=swh:1:snp:f4ce790777ea5f2e3406b5cb04e1a21df67f
                   61a0;anchor=swh:1:rel:97386b1f46cb310a26a8c1857cf9
                   f8cbd8c353c0;path=nkpanda97-leadtime-flexibility-
                   tradeoff-a80bf74
                  },
}
```

--- 

Just replace `XXXXXXX` with the actual Zenodo DOI after upload. Let me know if you want help with the Zenodo setup too!
---
## 📦 Repository Structure

```
ev_products_tradeoff/
│
├── scripts/                   # Simulation orchestration scripts
│   └── run_flex_product.py    # Main entry-point for simulations
│
├── utils/                     # Utility functions and helpers
│   └── powertech_helper.py    # Data processing and Pyomo utilities
│
├── powertech_experiments.py   # Core logic for model generation, optimization, and data loading
├── powertech_viz_cleaned.ipynb # Cleaned notebook for visualizing final results
├── figures/                   # Final paper figures (generated or stored)
├── slurm/                     # SLURM output and error logs for HPC runs
├── slurm_run.sh               # SLURM batch script to run experiments
├── requirements.txt           # Python dependencies (to be added)
├── README.md                  # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ev_products_tradeoff.git
cd ev_products_tradeoff
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> A `requirements.txt` file can be generated with:
> ```bash
> pip freeze > requirements.txt
> ```

---

## 🚀 How to Run

### Run simulations:
```bash
python scripts/run_flex_product.py
```

This executes the complete pipeline for data loading, optimization, and results generation.

---

## 📊 Visualizations

The `powertech_viz_cleaned.ipynb` notebook contains cleaned and reproducible code for generating final figures. Outputs are saved in the `figures/` directory.

---

## 🧠 Function Overview

### `powertech_experiments.py`
- `remove_infeasibility_transactions(ev_data, delta_t)`
  > Filters out infeasible EV charging sessions.
- `process_for_optimization(ev_data, year_, month_, day_, delta_t, key_dict, model_type)`
  > Prepares data and optimization model inputs.
- `get_cost_data(cost_data, emisson_data, ev_processed_data)`
  > Extracts cost and emission profiles from raw data.
- `bau_profile_generator(...)`
  > (No docstring) Likely simulates baseline profiles.
- `update_model(...)`
  > Populates Pyomo model parameters with EV flexibility data.

### `utils/powertech_helper.py`
- Contains Pyomo modeling tools and flexibility profiling helpers.

### `scripts/run_flex_product.py`
- Orchestrates data processing, model creation, and runs simulations for different flexibility products.

---

## 📁 SLURM (Optional HPC)
Use `slurm_run.sh` to submit jobs to a SLURM cluster.

---

## 📬 Contact
For questions or collaboration, feel free to reach out to the author via GitHub or email (as noted in the paper).

---

## 📝 License
MIT License or as per your publication terms.
---

## 📂 Data Requirements

The code requires access to 5 key input files. Due to privacy or size, these files are not provided. However, the expected structure is described below so users can recreate or mock the necessary files.

| File Path | Variable | Description |
|-----------|----------|-------------|
| `inputs/cs_per_category.pkl` | `CS_PER_CATEGORY` | Dictionary of CS IDs per category: `{'residential':[], 'commercial':[], 'shared':[], 'all_categories':[]}` |
| `inputs/day_ahead_price.pkl` | `PATH_COST_DATA` | Pandas DataFrame with columns: `['date', 'Day-ahead Price [EUR/kWh]']` |
| `inputs/mef.pkl` | `PATH_EMISSION_DATA` | Pandas DataFrame with columns: `['date', 'MEF']` |
| `inputs/transactiondata_allstations_powerlog_cleaned_onlywithpowerlog.pkl` | `EV_DATA_RAW` | Real-world EV transaction data. See structure below. |
| `inputs/ev_data_for_bau_profile_generation_2023.pkl` | `EV_DATA_FOR_BAU_PROFILE_GENERATION` | Cleaned and enriched EV data used for profile generation. |

### 🧾 Structure: EV_DATA_FOR_BAU_PROFILE_GENERATION

| Column | Type | Example |
|--------|------|---------|
| START_UTC_rounded | `Timestamp` | 2023-01-01 00:00:00+00:00 |
| STOP_UTC_rounded | `Timestamp` | 2023-01-01 14:00:00+00:00 |
| START_int | `int` | 0 |
| STOP_int | `int` | 14 |
| VOL | `float` | 2.159 |
| P_MAX | `float` | 6.18 |
| DUR_int, DUR_int_adj | `int` | 14 |
| Connector_id | `str` | 1707451 |
| day_ahead_price | `np.ndarray` | 1D array of length 15 |
| mef | `np.ndarray` | 1D array of length 15 |

### 🧾 Structure: EV_DATA_RAW

| Column | Type | Example |
|--------|------|---------|
| START, STOP | `Timestamp` | 2019-01-08 23:29:04+00:00 |
| ID_VIS_NUM, ID_TOK_UID | `str` | rLQViOlftUdh |
| VOL | `float` | 42.01 |
| COST | `float` | 0.0 |
| STATION | `str` | 1702074*1 |
| STATION_ADDRESS | `str` | Jan Pieterszoon Coenstraat 5 |
| DUR | `float` | 32.69 |
| P_MAX | `float` | 16.44 |
| Shared EV | `str` | No |

### 🧾 Structure: all_iter_list['residential'][0]['individual_data']

| Column | Type | Example |
|--------|------|---------|
| Base_profiles | `dict` | e.g., `profile_Cost_bi_directional_True: [float, ...]` |
| objective_value_* | `float` | e.g., -0.3078 |
| Other columns similar to `EV_DATA_FOR_BAU_PROFILE_GENERATION` |

---


## 🧠 Optimization Model

The model is used to simulate aggregate EV fleet flexibility for **redispatch** and **capacity limitation** products. Each transaction is modeled as a continuous charging variable over a discrete time horizon.

### ⚡ BAU Optimization (Unidirectional / V2G)

**Objective:**
- (1a) Minimize cost: `min ∑ Π_day_ahead * p * Δt`
- (1b) Minimize emissions: `min ∑ Π_mef * p * Δt`
- (1c) Unoptimized (greedy): `min ∑ e`

**Constraints:**
1. Charging outside connection window is zero
2. Total energy delivered matches historical session
3. Power bounded by min/max (depends on directionality)
4. Energy evolves linearly over time
5. Session capped to 24h

### 🔁 Redispatch Optimization

Maximize downward deviation from BAU:
```
max cr + ε * f(p)

subject to:
  p_t ≤ p_BAU - cr
  p = p_BAU before activation
```

### ⛔ Capacity Limitation Optimization

Minimize max power during window:
```
min cl + ε * f(p)

subject to:
  max(p_t) ≤ cl
  p = p_BAU before activation
```

Assumes:
- Perfect foresight of arrivals/departures
- No additional energy beyond historical
- Time resolution: 15 mins

---

### 🧾 Structure: `all_iter_list`

```python
all_iter_list = {
    'residential': [dict1, dict2, ...],
    'commercial': [dict1, dict2, ...],
    'shared': [dict1, dict2, ...]
}
```

Each item in the list (e.g., `dict1`) contains:

#### ➤ `individual_data`: DataFrame

| Column                | Type                            | Example |
|-----------------------|----------------------------------|---------|
| START_UTC_rounded     | `pandas.Timestamp`               | 2023-01-01 00:00:00+00:00 |
| STOP_UTC_rounded      | `pandas.Timestamp`               | 2023-01-01 15:00:00+00:00 |
| START_int / STOP_int  | `int`                            | 0 / 15 |
| VOL                   | `float`                          | 6.861 |
| P_MAX                 | `float`                          | 6.66 |
| DUR_int / DUR_int_adj | `int`                            | 15 |
| STOP_int_adj          | `int`                            | 15 |
| Connector_id          | `str`                            | 1802558 |
| day_ahead_price       | `np.ndarray`                     | e.g., 15 values |
| mef                   | `np.ndarray`                     | e.g., 15 values |
| Base_profiles         | `dict[str, List[float]]`         | Profile name → float list |
| objective_value_*     | `float`                          | e.g., -0.3 |

> Fields are similar across all three keys (`residential`, `commercial`, `shared`), so the structure above applies to all.

---

### 🧾 Structure: `all_iter_list` (Expanded)

```python
all_iter_list = {
    'residential': [run_1, run_2, ...],
    'commercial': [run_1, run_2, ...],
    'shared': [run_1, run_2, ...]
}
```

Each element in the list (e.g., `run_1`) is a dictionary with:

---

#### 🔹 `individual_data`: DataFrame

| Column                | Type               | Example |
|-----------------------|--------------------|---------|
| START_UTC_rounded     | `Timestamp`        | 2023-01-01 00:00:00+00:00 |
| STOP_UTC_rounded      | `Timestamp`        | 2023-01-01 15:00:00+00:00 |
| START_int / STOP_int  | `int`              | 0 / 15 |
| VOL                   | `float`            | 6.861 |
| P_MAX                 | `float`            | 6.66 |
| DUR_int, DUR_int_adj  | `int`              | 15 |
| STOP_int_adj          | `int`              | 15 |
| Connector_id          | `str`              | 1802558 |
| day_ahead_price       | `np.ndarray`       | Hourly price (length matches duration) |
| mef                   | `np.ndarray`       | Hourly emissions |
| Base_profiles         | `dict[str, list]`  | Profile name → list of floats |
| objective_value_*     | `float`            | Optimization objective |

---

#### 🔹 `aggregate_profiles`: Dict[str, np.ndarray]

Aggregated profiles across all sessions in a run.

| Key Example                        | Type        | Shape |
|-----------------------------------|-------------|--------|
| profile_Cost_bi_directional_True  | np.ndarray  | 72 |
| profile_MEF_bi_directional_False  | np.ndarray  | 72 |
| profile_Dumb_bi_directional_False | np.ndarray  | 72 |

---

#### 🔹 `signals`: Dict[str, pd.DataFrame]

Time-series signals used in optimization.

| Key               | Columns                                | Shape   |
|------------------|----------------------------------------|----------|
| `day_ahead_price`| `['date', 'Day-ahead Price [EUR/kWh]']`| 72 × 2 |
| `mef`            | `['date', 'MEF']`                      | 72 × 2 |

---
