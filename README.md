# RL-Based OS Scheduler with Real-Time Process Monitoring & Anomaly Detection

> **College OS Project — 2 Students · 13-Day Plan**
> Python · Q-Learning · psutil · Flask · Ubuntu 20.04 / 22.04

---

## 📁 Project Structure

```
os_scheduler_project/
├── monitor.py        # Real-time process & system stats (psutil)
├── scheduler.py      # Q-Learning agent (training loop + priority control)
├── anomaly.py        # CPU/memory spike detection (Z-score + hard threshold)
├── app.py            # Flask web dashboard (API + serves HTML)
├── templates/
│   └── index.html    # Live dashboard UI (Chart.js, auto-refresh)
├── q_table.pkl       # Saved Q-table (auto-created on first run)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## ⚙️ System Requirements

- Ubuntu 20.04 / 22.04 (or any modern Linux)
- Python 3.8+
- `sudo` access (needed to set negative nice values)

---

## 🚀 Setup Instructions

### Step 1 — Clone or create the project folder

```bash
mkdir os_scheduler_project
cd os_scheduler_project
# (copy all project files here)
```

### Step 2 — Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option A — Run the full dashboard (recommended)

```bash
sudo venv/bin/python app.py
```

> `sudo` is needed so the scheduler can adjust process nice values.
> Open your browser at: **http://127.0.0.1:5000**

### Option B — Test each module independently

```bash
# Test the process monitor
python monitor.py

# Test the Q-Learning scheduler (runs training loop in terminal)
sudo python scheduler.py

# Test the anomaly detector (30-second live test)
python anomaly.py
```

---

## 📦 requirements.txt

```
flask>=2.3.0
psutil>=5.9.0
```

Install with:
```bash
pip install flask psutil
```

---

## 🧠 How It Works

### 1. monitor.py
Uses `psutil` to collect:
- Per-process: PID, name, CPU %, memory %, nice value, status
- System-wide: total CPU %, RAM used/total

### 2. scheduler.py — Q-Learning Agent

| Component    | Details |
|-------------|---------|
| **State**   | (cpu_bucket, mem_bucket) → e.g. `"high_medium"` |
| **Actions** | `increase_priority` (nice −5), `decrease_priority` (nice +5), `keep` |
| **Reward**  | `0.6 × cpu_drop + 0.4 × mem_drop` after applying action |
| **Update**  | Bellman equation: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]` |
| **Policy**  | ε-greedy (ε decays from 0.2 → 0.05 over time) |

### 3. anomaly.py — Two Detection Methods

| Method | Trigger |
|--------|---------|
| **Hard threshold** | CPU > 85% or Memory > 90% |
| **Z-score spike** | Value > 2 standard deviations from rolling mean |

### 4. app.py + index.html
Flask serves a live dashboard. Browser polls 6 JSON endpoints every 5 seconds:
- `/api/stats` — CPU & memory gauges
- `/api/processes` — process table
- `/api/decisions` — latest scheduler actions
- `/api/anomalies` — alerts and log
- `/api/chart` — before/after CPU data
- `/api/qtable` — Q-table snapshot

---

## 📅 13-Day Work Plan

| Days  | Task | Student |
|-------|------|---------|
| 1–2   | Setup, `monitor.py` | Student 1 |
| 3–4   | `scheduler.py` (Q-table, actions) | Student 2 |
| 5–6   | `anomaly.py` (Z-score logic) | Student 1 |
| 7–8   | `app.py` Flask routes + threading | Student 2 |
| 9–10  | `index.html` dashboard + Chart.js | Both |
| 11    | Integration testing on Ubuntu | Both |
| 12    | Viva prep + comments review | Both |
| 13    | Final demo & documentation | Both |

---

## 🎤 Viva Talking Points

1. **Why Q-learning?** — Tabular, no neural networks, easy to explain and debug.
2. **State space** — Discretizing CPU/mem into 3 bins each = 9 possible states. Small Q-table.
3. **Reward function** — Penalises high CPU/memory, rewards reductions.
4. **Epsilon decay** — Balances exploration early on vs exploitation once Q-table matures.
5. **Nice values** — Linux scheduling priority. Range −20 (highest) to +19 (lowest). Requires root.
6. **Z-score anomaly** — Statistically rigorous, adapts to the baseline of the current machine.
7. **Thread safety** — `threading.Lock` prevents race conditions between the background monitor and Flask.

---

## ⚠️ Notes

- On Ubuntu, changing a process to negative nice requires `sudo`.
- The scheduler only touches the **top 5 CPU-consuming processes** to avoid affecting system stability.
- System processes (PID < 100) may raise `AccessDenied` — these are silently skipped.
- The Q-table is saved as `q_table.pkl` and reloaded on restart, so learning persists.

---

*Built with ❤️ for the OS course project.*
