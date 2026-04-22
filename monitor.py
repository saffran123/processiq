"""
monitor.py - Real-time Process Monitor
--------------------------------------
Uses psutil to collect live CPU, memory, and process data.
This module is the "eyes" of the system — it gathers data for
the scheduler and anomaly detector to act on.

Authors: Student 1 & Student 2
"""

import psutil
import time
import datetime
import sys

# ── Windows priority class → display nice value mapping ──────────────────────
# Windows doesn't use -20..19 nice values. We map its constants to a readable int.
WIN_PRIORITY_MAP = {}

if sys.platform == "win32":
    WIN_PRIORITY_MAP = {
        psutil.IDLE_PRIORITY_CLASS: 19,
        psutil.BELOW_NORMAL_PRIORITY_CLASS: 10,
        psutil.NORMAL_PRIORITY_CLASS: 0,
        psutil.ABOVE_NORMAL_PRIORITY_CLASS: -5,
        psutil.HIGH_PRIORITY_CLASS: -10,
        psutil.REALTIME_PRIORITY_CLASS: -20,
    }


def _safe_nice(raw_nice):
    """
    Normalise the 'nice' value from psutil so it's always a small integer
    regardless of platform.
    - Linux/macOS : already an int in -20..19, return as-is
    - Windows     : psutil returns a priority CLASS constant (e.g. 32, 64).
                    Map it to an equivalent nice-style int for display.
    """
    if sys.platform == "win32":
        return WIN_PRIORITY_MAP.get(raw_nice, 0)
    return raw_nice if raw_nice is not None else 0


def get_all_processes():
    """
    Returns a list of dicts, one per running process.
    Each dict contains: pid, name, cpu_percent, memory_percent, status, priority (nice value).
    """
    processes = []

    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status', 'nice']):
        try:
            info = proc.info
            # Skip zombie/dead processes
            if info['status'] in ('zombie', 'dead'):
                continue
            processes.append({
                'pid': info['pid'],
                'name': info['name'] or 'unknown',
                'cpu_percent': round(info['cpu_percent'] or 0.0, 2),
                'memory_percent': round(info['memory_percent'] or 0.0, 2),
                'status': info['status'],
                'nice': _safe_nice(info['nice'])
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process may have ended or we lack permissions — skip it
            continue

    return processes


def get_system_stats():
    """
    Returns overall system CPU and memory usage.
    Used by the dashboard and anomaly detector.
    """
    cpu = psutil.cpu_percent(interval=0.5)          # 0.5s sample window
    mem = psutil.virtual_memory()
    return {
        'cpu_total': cpu,
        'memory_total': mem.percent,
        'memory_used_mb': round(mem.used / (1024 * 1024), 1),
        'memory_total_mb': round(mem.total / (1024 * 1024), 1),
        'timestamp': datetime.datetime.now().strftime("%H:%M:%S")
    }


def get_top_processes(n=10):
    """
    Returns top-N processes sorted by CPU usage.
    Used by the scheduler to decide which processes to adjust.
    """
    procs = get_all_processes()
    # Sort by CPU descending, then memory as tiebreaker
    sorted_procs = sorted(procs, key=lambda p: (p['cpu_percent'], p['memory_percent']), reverse=True)
    return sorted_procs[:n]


def get_process_by_pid(pid):
    """
    Returns info for a single process by PID, or None if not found.
    """
    try:
        proc = psutil.Process(pid)
        info = proc.as_dict(attrs=['pid', 'name', 'cpu_percent', 'memory_percent', 'status', 'nice'])
        return info
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


# ── Quick test: run this file directly ──────────────────────────────────────
if __name__ == "__main__":
    print("=== System Stats ===")
    stats = get_system_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== Top 5 Processes by CPU ===")
    for p in get_top_processes(5):
        print(f"  PID {p['pid']:6d}  CPU {p['cpu_percent']:5.1f}%  MEM {p['memory_percent']:5.1f}%  "
              f"Nice {p['nice']:3d}  {p['name']}")