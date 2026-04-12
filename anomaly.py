"""
anomaly.py - Anomaly Detection Module
--------------------------------------
Detects CPU and memory spikes using a rolling-window Z-score approach.

How it works:
  1. We keep a short history (window) of recent CPU/memory readings.
  2. We compute the mean and standard deviation of that window.
  3. If a new reading is more than THRESHOLD standard deviations from
     the mean, we flag it as an anomaly.

This is called "statistical process control" — simple but effective.

Authors: Student 1 & Student 2
"""

import collections
import math
import datetime


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_SIZE    = 20    # How many past readings to keep (rolling window)
Z_THRESHOLD    = 2.0   # Flag if reading is > 2 standard deviations from mean
CPU_ALERT_PCT  = 85.0  # Hard threshold: always alert above this CPU %
MEM_ALERT_PCT  = 90.0  # Hard threshold: always alert above this Memory %


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Maintains rolling windows of system stats and detects anomalies.
    Two detection methods:
      1. Hard threshold  — instant alert if CPU/mem exceeds a fixed limit
      2. Z-score         — statistical alert when a value is unusually high
    """

    def __init__(self):
        # deque acts like a list but automatically drops old items
        self.cpu_history  = collections.deque(maxlen=WINDOW_SIZE)
        self.mem_history  = collections.deque(maxlen=WINDOW_SIZE)
        self.anomaly_log  = []   # All detected anomalies (shown on dashboard)

    # ── Statistical helpers ──────────────────────────────────────────────────

    def _mean(self, data):
        """Compute arithmetic mean of a list."""
        if not data:
            return 0.0
        return sum(data) / len(data)

    def _std(self, data):
        """
        Compute population standard deviation.
        std = sqrt( mean( (xi - mean)^2 ) )
        """
        if len(data) < 2:
            return 0.0
        mean = self._mean(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return math.sqrt(variance)

    def _z_score(self, value, data):
        """
        Z-score = (value - mean) / std
        Tells us how many standard deviations the value is from the mean.
        """
        std = self._std(data)
        if std == 0:
            return 0.0
        return (value - self._mean(data)) / std

    # ── Main detection method ────────────────────────────────────────────────

    def check(self, cpu_pct, mem_pct):
        """
        Feed a new (cpu_pct, mem_pct) reading into the detector.
        Returns a list of anomaly dicts (empty if none detected).
        """
        anomalies = []
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # 1. Hard threshold check (immediate danger zone)
        if cpu_pct > CPU_ALERT_PCT:
            anomalies.append(self._make_anomaly(
                kind="CPU Hard Threshold",
                value=cpu_pct,
                threshold=CPU_ALERT_PCT,
                severity="critical",
                timestamp=timestamp
            ))

        if mem_pct > MEM_ALERT_PCT:
            anomalies.append(self._make_anomaly(
                kind="Memory Hard Threshold",
                value=mem_pct,
                threshold=MEM_ALERT_PCT,
                severity="critical",
                timestamp=timestamp
            ))

        # 2. Z-score check (statistical spike detection)
        # Only meaningful once we have enough history
        if len(self.cpu_history) >= 5:
            z_cpu = self._z_score(cpu_pct, list(self.cpu_history))
            if z_cpu > Z_THRESHOLD:
                anomalies.append(self._make_anomaly(
                    kind="CPU Spike (Z-score)",
                    value=cpu_pct,
                    threshold=round(self._mean(list(self.cpu_history)) +
                                    Z_THRESHOLD * self._std(list(self.cpu_history)), 1),
                    severity="warning",
                    timestamp=timestamp,
                    extra=f"z={z_cpu:.2f}"
                ))

        if len(self.mem_history) >= 5:
            z_mem = self._z_score(mem_pct, list(self.mem_history))
            if z_mem > Z_THRESHOLD:
                anomalies.append(self._make_anomaly(
                    kind="Memory Spike (Z-score)",
                    value=mem_pct,
                    threshold=round(self._mean(list(self.mem_history)) +
                                    Z_THRESHOLD * self._std(list(self.mem_history)), 1),
                    severity="warning",
                    timestamp=timestamp,
                    extra=f"z={z_mem:.2f}"
                ))

        # 3. Update rolling windows AFTER scoring (don't bias current check)
        self.cpu_history.append(cpu_pct)
        self.mem_history.append(mem_pct)

        # 4. Log anomalies (keep last 50)
        if anomalies:
            self.anomaly_log.extend(anomalies)
            self.anomaly_log = self.anomaly_log[-50:]

        return anomalies

    def _make_anomaly(self, kind, value, threshold, severity, timestamp, extra=""):
        """Helper to build a standardised anomaly record."""
        return {
            'type': kind,
            'value': round(value, 1),
            'threshold': threshold,
            'severity': severity,    # "critical" or "warning"
            'timestamp': timestamp,
            'detail': extra
        }

    # ── Convenience accessors ────────────────────────────────────────────────

    def get_log(self):
        """Return the most recent 20 anomalies for dashboard display."""
        return list(reversed(self.anomaly_log[-20:]))

    def get_stats(self):
        """Return current rolling-window statistics."""
        cpu_list = list(self.cpu_history)
        mem_list = list(self.mem_history)
        return {
            'cpu_mean':  round(self._mean(cpu_list), 1),
            'cpu_std':   round(self._std(cpu_list), 1),
            'mem_mean':  round(self._mean(mem_list), 1),
            'mem_std':   round(self._std(mem_list), 1),
            'samples':   len(cpu_list)
        }

    def clear_log(self):
        """Clear the anomaly log (useful for testing)."""
        self.anomaly_log = []


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time, random
    from monitor import get_system_stats

    detector = AnomalyDetector()
    print("Anomaly Detector running — monitoring system for 30 seconds...\n")

    for i in range(30):
        stats = get_system_stats()
        cpu = stats['cpu_total']
        mem = stats['memory_total']
        alerts = detector.check(cpu, mem)

        status = "  [OK]"
        if alerts:
            status = f"  ⚠  {', '.join(a['type'] for a in alerts)}"

        print(f"  t={i+1:2d}  CPU {cpu:5.1f}%  MEM {mem:5.1f}%{status}")
        time.sleep(1)

    print("\n── Anomaly Log ──")
    for a in detector.get_log():
        print(f"  [{a['severity'].upper():8s}] {a['timestamp']}  {a['type']}  "
              f"value={a['value']}%  threshold={a['threshold']}%  {a['detail']}")

    print("\n── Rolling Stats ──")
    s = detector.get_stats()
    print(f"  CPU  mean={s['cpu_mean']}%  std={s['cpu_std']}%")
    print(f"  MEM  mean={s['mem_mean']}%  std={s['mem_std']}%")
    print(f"  Samples in window: {s['samples']}")
