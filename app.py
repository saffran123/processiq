"""
app.py - Flask Dashboard
------------------------
Serves a real-time web dashboard that shows:
  - Live system CPU and memory usage
  - Top processes table
  - Scheduler decisions (Q-learning actions)
  - Anomaly alerts
  - CPU usage chart (before vs after optimization)
  - Q-table snapshot

All data is served via JSON API endpoints so the browser
can poll them every few seconds without full page reloads.

Authors: Student 1 & Student 2
Run with: python app.py
"""

from flask import Flask, render_template, jsonify
import threading
import time
import os

from monitor import get_system_stats, get_top_processes
from scheduler import QLearningScheduler
from anomaly import AnomalyDetector

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE (updated by background thread, read by API endpoints)
# ─────────────────────────────────────────────────────────────────────────────

# We use a simple dict protected by a threading Lock to avoid race conditions.
state_lock = threading.Lock()
shared_state = {
    'system_stats':    {},
    'top_processes':   [],
    'decisions':       [],
    'anomalies':       [],
    'anomaly_log':     [],
    'q_table':         [],
    'cpu_history':     [],   # rolling list for chart
    'cpu_optimized':   [],   # after-optimization values for chart
    'episode':         0,
}

# Global instances
scheduler = QLearningScheduler()
detector  = AnomalyDetector()


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND MONITORING & TRAINING THREAD
# ─────────────────────────────────────────────────────────────────────────────

def background_loop():
    """
    Runs in a separate thread every 5 seconds:
      1. Collect system stats
      2. Check for anomalies
      3. Run one Q-learning training step
      4. Update shared_state
    """
    episode = 0
    while True:
        try:
            # Step 1: Collect stats
            stats = get_system_stats()
            procs = get_top_processes(15)

            # Step 2: Anomaly detection
            alerts = detector.check(stats['cpu_total'], stats['memory_total'])
            log    = detector.get_log()

            # Step 3: Q-learning training step
            decisions = scheduler.train_step()
            episode  += 1

            # Collect cpu_after values for the chart
            cpu_after_vals = [d['cpu_after'] for d in decisions if decisions]

            # Step 4: Update shared state safely
            with state_lock:
                shared_state['system_stats']  = stats
                shared_state['top_processes'] = procs
                shared_state['decisions']     = decisions
                shared_state['anomalies']     = alerts
                shared_state['anomaly_log']   = log
                shared_state['q_table']       = scheduler.get_q_table_summary()
                shared_state['episode']       = episode

                # Rolling CPU history (last 30 data points for chart)
                shared_state['cpu_history'].append(stats['cpu_total'])
                shared_state['cpu_history'] = shared_state['cpu_history'][-30:]

                # Average cpu_after for the episode
                avg_opt = (sum(cpu_after_vals) / len(cpu_after_vals)) if cpu_after_vals else stats['cpu_total']
                shared_state['cpu_optimized'].append(round(avg_opt, 1))
                shared_state['cpu_optimized'] = shared_state['cpu_optimized'][-30:]

        except Exception as e:
            print(f"[Background] Error: {e}")

        time.sleep(5)   # Poll every 5 seconds


# Start the background thread (daemon=True means it dies with the main process)
bg_thread = threading.Thread(target=background_loop, daemon=True)
bg_thread.start()


# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard HTML page."""
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    """Return system stats as JSON."""
    with state_lock:
        return jsonify(shared_state['system_stats'])


@app.route("/api/processes")
def api_processes():
    """Return top processes as JSON."""
    with state_lock:
        return jsonify(shared_state['top_processes'])


@app.route("/api/decisions")
def api_decisions():
    """Return latest scheduler decisions."""
    with state_lock:
        return jsonify({
            'episode':   shared_state['episode'],
            'epsilon':   round(scheduler.epsilon, 4),
            'decisions': shared_state['decisions']
        })


@app.route("/api/anomalies")
def api_anomalies():
    """Return latest anomaly alerts and log."""
    with state_lock:
        return jsonify({
            'active':  shared_state['anomalies'],
            'log':     shared_state['anomaly_log'],
            'stats':   detector.get_stats()
        })


@app.route("/api/chart")
def api_chart():
    """Return CPU history data for the before/after chart."""
    with state_lock:
        return jsonify({
            'labels':    list(range(1, len(shared_state['cpu_history']) + 1)),
            'before':    shared_state['cpu_history'],
            'after':     shared_state['cpu_optimized']
        })


@app.route("/api/qtable")
def api_qtable():
    """Return Q-table summary."""
    with state_lock:
        return jsonify(shared_state['q_table'])


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  RL-Based OS Scheduler Dashboard")
    print("  Open your browser at: http://127.0.0.1:5001")
    print("=" * 55)
    # debug=False so that Flask doesn't reload and spawn a second bg thread
    app.run(host="0.0.0.0", port=5001, debug=False)
