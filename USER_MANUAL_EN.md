# User Manual & Replication Guide: Cross-Faculty Shared Passenger Van Allocation MILP Engine (English Edition)

**System Name:** Cross-Faculty Shared Passenger Van Allocation MILP Engine (v6 Empirical Pipeline)  
**Primary Source Files:** `van_allocation_engine.py` and `testing_v6_Candidate.ipynb`  
**Regulatory Compliance Standards:** Thailand Land Transport Act B.E. 2522, Road Traffic Act B.E. 2522, and International Hours-of-Service (HOS 14h / 72h / 10h) Benchmarks

---

## 1. System Overview

This software package provides a Mixed-Integer Linear Programming (MILP) Decision Support System (DSS) and empirical benchmarking pipeline for allocating passenger vans and drivers across multiple university faculties over a rolling 14-day planning horizon.

### Key Capabilities
1. **Hierarchical Cross-Faculty Resource Pooling:** Prioritizes each faculty's own vehicles and primary drivers first, while automatically lending idle vans across faculties when a requester's home fleet is fully occupied.
2. **Multi-Van Request Splitting:** Automatically splits large passenger groups (e.g., 12–33 passengers) across up to $K_r = 3$ vans (standard capacity $C_v = 11$ seats per van).
3. **Hard Hours-of-Service (HOS) Safety Compliance:**
   - Maximum daily operational duty of **14 hours/day** ($H_{\max}^{\text{day}} = 14.0\text{ h}$)
   - Maximum cumulative workload of **72 hours per 14-day horizon** ($H_{\max}^{14\text{d}} = 72.0\text{ h}$, averaging 36 h/week)
   - Minimum mandatory inter-shift rest turnaround of **10 consecutive hours** ($\Delta_{\text{rest}} = 10.0\text{ h}$)
4. **Automated Multi-Format Data Cleaning Pipeline (`v6`):**
   - Automatically converts Thai Buddhist Era years (`B.E. > 2400`) to Gregorian years (`C.E.`) prior to timestamp parsing to prevent 64-bit nanosecond `OutOfBoundsDatetime` errors.
   - Normalizes dot-delimited time strings (e.g., `10.00`, `16.30`) into standard `HH:MM:SS` ISO timestamps.
   - Prunes trailing blank spreadsheet rows (`.dropna(how='all')`) and standardizes status labels (`approve` $\rightarrow$ `approved`, `delete` $\rightarrow$ `cancel`).
5. **Built-in Baselines & Independent Verification Oracle:**
   - `MILP` (Proposed global optimization model solved via COIN-OR CBC in PuLP)
   - `FCFS-A` (Greedy First-Come, First-Served without HOS fatigue checks)
   - `FCFS-B` (Safety-compliant Greedy First-Come, First-Served)
   - `Heuristic-SDF` (Safety-compliant Shortest-Duration-First greedy heuristic)
   - `Rule-Based Verification Oracle` (Deterministic checker validating all 864 parametric scenarios)

---

## 2. Repository & File Structure

| File / Directory | Type | Description |
| :--- | :--- | :--- |
| `van_allocation_engine.py` | Python Module / CLI | Standalone, modular Python optimization engine and CLI runner for production deployment and API integration |
| `testing_v6_Candidate.ipynb` | Jupyter Notebook | Complete 70-cell experimental notebook reproducing all tables, statistical weights, and figures in the research paper |
| `requirements.txt` | Configuration | Python package dependencies required to run the solver and notebook |
| `bookingN.csv`, `driversN.csv` | Input CSV | Real operational booking requests and driver work logs from the School of ICT |
| `sci_bookingN.csv`, `sci_driversN.csv` | Input CSV | Real operational booking requests and driver work logs from the School of Science (SCI) |
| `seen_bookingN.csv`, `seen_driversN.csv` | Input CSV | Synthetic stress-test booking requests and driver logs for the School of Energy and Environment (SEEN) |
| `raw_output/` | Output Directory | Generated CSV summary tables and PNG visualizations |
| `EASR_Submission/` | Submission Package | Contains the formatted EASR manuscript (`EASR_manuscript_EN.docx`) and separate 300-DPI figure files (`.jpg` and `.tiff`) |

---

## 3. Prerequisites & Installation

### 3.1 System Requirements
- **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **Python Version:** Python 3.9 – 3.12 (Python 3.11 recommended)
- **Memory (RAM):** Minimum 4 GB (8 GB recommended for full 900-scenario batch simulations)
- **MILP Solver:** **COIN-OR CBC** (bundled automatically with the open-source `pulp` library; no commercial license required)

### 3.2 Installation Steps
Open a terminal or PowerShell window in the project root directory (`d:\Pro`) and execute:

```powershell
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install all required packages
pip install -r requirements.txt
```

---

## 4. Input Data Schema

To deploy the model with data from another university or institution, prepare two CSV tables encoded in `UTF-8`:

### 4.1 Booking Requests Table (`*bookingN.csv`)
| Column Name | Data Type | Example Value | Description |
| :--- | :--- | :--- | :--- |
| `booking_id` | String | `B000001` | Unique booking identifier (automatically prefixed by faculty code during ingestion) |
| `requester_faculty` | String | `ICT`, `SCI` | Requesting faculty or department code |
| `passenger_count` | Integer | `8`, `15` | Number of passengers requiring transport ($P_r \ge 1$) |
| `trip_scope` | String | `in_province`, `out_of_province` | Trip scope (`out_of_province` mandates a dedicated driver per assigned van) |
| `start_date`, `end_date` | Date String | `2025-02-10` or `10/02/2568` | Trip departure and return dates (supports both Gregorian C.E. and Buddhist Era B.E.) |
| `start_time`, `end_time` | Time String | `08:00:00` or `08.00` | Departure and return times (supports both `:` and `.` delimiters) |
| `booking_status` | String | `approved`, `pending`, `cancel` | Reservation status (`approve` is normalized to `approved` automatically) |

### 4.2 Driver Work Logs Table (`*driversN.csv`)
| Column Name | Data Type | Example Value | Description |
| :--- | :--- | :--- | :--- |
| `log_id` | String | `DL000001` | Unique driver duty log identifier |
| `driver_id` | String | `ICT_DRIVER_NOW` | Driver identifier (`_NOW` = active driver blocking current horizon; `_OLD` = historical log) |
| `work_date` | Date String | `15/01/2568` | Date of duty (supports both C.E. and B.E. formats) |
| `start_time`, `end_time` | Time String | `08.30`, `17.00` | Duty start and end times |
| `work_hours` | Float | `8.5` | Total duty hours logged on that date |

---

## 5. Execution & Replication Guide

### Option 1: Command-Line Execution (`van_allocation_engine.py`)
Run the standalone engine from the command line to inspect verified v6 results or execute a sample allocation:

```powershell
# 1. Print all verified v6 empirical tables from raw_output/
python van_allocation_engine.py --mode summary

# 2. Run a self-contained 3-faculty cross-faculty allocation demonstration
python van_allocation_engine.py --mode demo
```

### Option 2: Programmatic Python API Integration
Import `solve_van_allocation_milp` and `AllocationConfig` directly into any Python web backend (e.g., FastAPI, Flask, or Django):

```python
import pandas as pd
from van_allocation_engine import AllocationConfig, solve_van_allocation_milp

# 1. Configure institutional parameters
config = AllocationConfig(
    faculties=["ENG", "MED", "BUS", "SCI"],
    van_capacity=11,
    max_hours_per_day=14.0,
    max_hours_per_14d=72.0,
    min_rest_hours=10.0,
)

# 2. Define fleet, drivers, and pending requests
vans = [
    {"van_id": "VAN_ENG_01", "faculty": "ENG", "capacity": 11, "primary_driver_id": "DRV_ENG_01"},
    {"van_id": "VAN_SCI_01", "faculty": "SCI", "capacity": 11, "primary_driver_id": "DRV_SCI_01"},
]
drivers = [
    {"driver_id": "DRV_ENG_01", "faculty": "ENG", "base_day_hours": {1: 4.0}},
    {"driver_id": "DRV_SCI_01", "faculty": "SCI", "base_day_hours": {1: 0.0}},
]
requests = [
    {
        "request_id": "REQ_101",
        "faculty": "ENG",
        "pax": 10,
        "max_vans": 1,
        "is_out_of_province": 1,
        "start_dt": pd.Timestamp("2025-03-01 08:00:00"),
        "end_dt": pd.Timestamp("2025-03-02 18:00:00"),
        "day_hours": {1: 10.0, 2: 10.0},
    }
]

# 3. Solve the MILP allocation model
result = solve_van_allocation_milp(requests, vans, drivers, config=config)
print("Solver Status:", result["solver_status"])
print("Assignments:", result["assignments"])
```

### Option 3: Full Notebook Replication (`testing_v6_Candidate.ipynb`)
To re-run all 70 cells of the experimental pipeline and regenerate every CSV table and figure from scratch:

```powershell
jupyter nbconvert --to notebook --execute --inplace testing_v6_Candidate.ipynb
```

---

## 6. Customizing the Model for Other Universities

Other institutions can adapt the model without altering its mathematical constraints by modifying `AllocationConfig` in `van_allocation_engine.py`:

1. **Vehicle Seating Capacity (`van_capacity`):** Defaults to `11` seats. Change `van_capacity` or specify per-vehicle `"capacity"` in the `vans` list to support 13-seat vans or 20-seat minibuses.
2. **HOS Safety Thresholds (`max_hours_per_day`, `max_hours_per_14d`, `min_rest_hours`):**
   - Defaults are `14.0` h/day, `72.0` h/14 days, and `10.0` h minimum rest.
   - Adjust these values directly to reflect local labor union agreements or national transport statutes.
3. **Centralized vs. Decentralized Ownership Policy (`w3_same_faculty_van` vs. `w3_cross_faculty_van`):**
   - By default, assigning a home-faculty van incurs a soft cost of `100.0`, while borrowing a van across faculties incurs `400.0`.
   - To operate a **fully centralized university motor pool** with no faculty ownership distinction, set both `w3_same_faculty_van = 100.0` and `w3_cross_faculty_van = 100.0`.

---

## 7. Guide to Generated Output Files (`raw_output/`)

| Output Filename | Description |
| :--- | :--- |
| `empirical_probability_comparison.csv` | Empirical demand and fleet availability probabilities from cleaned ICT + SCI records vs. synthetic priors |
| `scenario_empirical_vs_synthetic.csv` | Exhaustive 864-scenario summary (97.09% empirical full assignment; 100% Oracle match) |
| `baseline_comparison_real.csv` | Performance comparison of `MILP`, `FCFS-A`, `FCFS-B`, and `Heuristic-SDF` on the $N=10$ real pending requests |
| `baseline_comparison_synthetic.csv` | Multi-request benchmark results across 900 synthetic stress-test instances |
| `exp1_scalability.csv` | Solver runtime and optimality rates across request batch sizes $N \in \{5, \dots, 60\}$ |
| `exp2_fleet_sizing_and_sharing.csv` | Sensitivity analysis across fleet sizes (1, 2, 3 vans/faculty) and sharing policies |
| `exp3_hos_sensitivity.csv` | Sensitivity analysis across 14-day cumulative workload limits ($48\text{ h}$ to $96\text{ h}$) |
| `baseline_fulfillment.png` | Bar chart comparing request fulfillment rates and safety violations across all 4 algorithms |
| `sensitivity_and_scalability.png` | 4-panel figure illustrating fleet sizing, HOS quota sensitivity, and solver runtime scaling |

---

## 8. Troubleshooting

1. **`OutOfBoundsDatetime` when loading Thai CSV files:**
   - **Cause:** Buddhist Era years (`2566–2568`) exceed Pandas' 64-bit nanosecond timestamp limit (`2262 C.E.`).
   - **Solution:** Always pass raw date columns through `parse_flexible_date()`, which converts years `> 2400` to Gregorian before invoking `pd.to_datetime()`.
2. **Inflated row counts when loading Excel-exported CSVs:**
   - **Cause:** Trailing blank rows containing only commas (`,,,,,`) in Excel exports.
   - **Solution:** Call `df.dropna(how='all')` immediately after `pd.read_csv()` (built into the v6 pipeline).
3. **`PulpSolverError: Cannot execute cbc`:**
   - **Solution:** Reinstall PuLP with `pip install --upgrade --force-reinstall pulp` to restore the bundled COIN-OR CBC executable for your platform.
