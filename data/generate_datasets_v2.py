"""
Smart Health Guardian - synthetic dataset v2 generator.

Why v2 (findings from the V29 device tests, 2026-10-14/15):
  * v1 rows were independent random values every 30 ms, so the 3 s window average of every class
    was an almost fixed point (e.g. Relaxed SpO2 97.8-98.2). Real readings fell in the gaps between
    classes and got arbitrary labels (anomaly score of real data 0.23-3.08 vs <= 0.13 for training).
  * v1 class ranges were narrow (Relaxed 70-80 bpm): a resting person at 88 bpm was "unknown".
  * v1 SignalQuality was ~1.0 for every healthy class, while the real sensor gives 0.6-0.85.

v2 design:
  * Each class file is made of segments of 6-15 s (~40-100 states per class). Every segment draws its own physiological state
    inside a realistic range, and values drift slowly inside the segment (like the device, whose
    values are smoothed). The 3 s window averages therefore cover the whole class range.
  * Inside a class, heart rate and HRV are linked through one "arousal" value per segment
    (higher BPM -> lower RMSSD / pNN50), as in real physiology.
  * Ranges are wider and overlap at the edges (e.g. Relaxed / Stressed around 85-95 bpm), so the
    validation accuracy is expected to drop below 100% - that is more honest than v1.
  * Several files per class (Class.NN.csv): Edge Impulse takes the label from the text before the
    first dot and splits train/test per file, so test windows do not overlap training windows.
  * Same columns and 30 ms timestamps as v1; 480 s per class (600 s Relaxed), twice v1, to cover the wider ranges.

Usage:  python generate_datasets_v2.py      -> writes ./datasets_v2/*.csv  (seeded, reproducible)
"""
import csv
import os
import random

SEED = 2026
ROW_MS = 30                      # same as v1 (~33 Hz)
FILES_PER_CLASS = 8
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets_v2")

# Per-class ranges for the segment state.
#   bpm / rmssd / pnn50 are given as (value at arousal=0, value at arousal=1): HR rises, HRV falls.
#   spo2 / temp / sigq are independent uniform ranges.
CLASSES = {
    "Relaxed": dict(seconds=600, bpm=(55, 92), rmssd=(85, 25), pnn50=(50, 8),
                    spo2=(96.0, 100.0), temp=(36.1, 37.2), sigq=(0.55, 1.00)),
    "Stressed": dict(seconds=480, bpm=(85, 125), rmssd=(40, 12), pnn50=(15, 0),
                     spo2=(95.0, 99.5), temp=(36.3, 37.3), sigq=(0.55, 1.00)),
    "Active": dict(seconds=480, bpm=(110, 175), rmssd=(22, 4), pnn50=(6, 0),
                   spo2=(94.0, 99.0), temp=(36.8, 37.9), sigq=(0.45, 0.95)),
    "Fever": dict(seconds=480, bpm=(88, 130), rmssd=(40, 10), pnn50=(15, 0),
                  spo2=(94.0, 98.5), temp=(38.0, 40.5), sigq=(0.55, 1.00)),
    "Hypoxia": dict(seconds=480, bpm=(80, 130), rmssd=(45, 10), pnn50=(18, 0),
                    spo2=(80.0, 91.5), temp=(36.1, 37.3), sigq=(0.50, 0.95)),
    # irregular rhythm (e.g. atrial fibrillation): very high beat-to-beat variability at any HR
    "Arrhythmia": dict(seconds=480, bpm=(60, 140), rmssd=(110, 220), pnn50=(55, 95),
                       spo2=(94.0, 99.0), temp=(36.1, 37.2), sigq=(0.55, 0.95)),
}

# Drift inside a segment (std of the slow random walk per second) and per-row noise.
DRIFT = dict(BPM=0.6, SpO2=0.08, Temp=0.01, RMSSD=1.0, pNN50=0.6, SignalQuality=0.01)
NOISE = dict(BPM=0.3, SpO2=0.05, Temp=0.01, RMSSD=0.4, pNN50=0.3, SignalQuality=0.005)
LIMITS = dict(BPM=(30, 220), SpO2=(70, 100), Temp=(20, 42), RMSSD=(0, 300), pNN50=(0, 100),
              SignalQuality=(0, 1))
COLUMNS = ["BPM", "SpO2", "Temp", "RMSSD", "pNN50", "SignalQuality"]


def lerp(pair, u):
    return pair[0] + (pair[1] - pair[0]) * u


def clamp(name, v):
    lo, hi = LIMITS[name]
    return max(lo, min(hi, v))


def segment_state(rng, cfg):
    """Physiological state of one segment."""
    u = rng.random()                                     # arousal inside the class
    jitter = lambda span: rng.gauss(0, span * 0.06)      # individual differences
    state = {
        "BPM": lerp(cfg["bpm"], u) + jitter(abs(cfg["bpm"][1] - cfg["bpm"][0])),
        "RMSSD": lerp(cfg["rmssd"], u) + jitter(abs(cfg["rmssd"][1] - cfg["rmssd"][0])),
        "pNN50": lerp(cfg["pnn50"], u) + jitter(abs(cfg["pnn50"][1] - cfg["pnn50"][0])),
        "SpO2": rng.uniform(*cfg["spo2"]),
        "Temp": rng.uniform(*cfg["temp"]),
        "SignalQuality": rng.uniform(*cfg["sigq"]),
    }
    return {k: clamp(k, v) for k, v in state.items()}


def poor_signal_rows(rng, n_rows):
    """Bad contact / motion: implausible, jumpy values, low quality, die temperature only."""
    rows, t = [], 0
    while len(rows) < n_rows:
        seg = int(rng.uniform(2, 8) * 1000 / ROW_MS)     # short bursts of garbage
        base = dict(BPM=rng.uniform(30, 200), SpO2=rng.uniform(70, 100), Temp=rng.uniform(24, 34),
                    RMSSD=rng.uniform(0, 250), pNN50=rng.uniform(0, 100),
                    SignalQuality=rng.uniform(0.0, 0.3))
        for _ in range(min(seg, n_rows - len(rows))):
            row = {k: clamp(k, base[k] + rng.gauss(0, (LIMITS[k][1] - LIMITS[k][0]) * 0.02))
                   for k in COLUMNS}
            row["SignalQuality"] = clamp("SignalQuality", min(row["SignalQuality"], 0.3))
            rows.append(row)
    return rows


def class_rows(rng, cfg, n_rows):
    rows = []
    while len(rows) < n_rows:
        state = segment_state(rng, cfg)
        seg_rows = int(rng.uniform(6, 15) * 1000 / ROW_MS)
        per_row = (ROW_MS / 1000.0) ** 0.5               # random-walk step per row from per-second std
        for _ in range(min(seg_rows, n_rows - len(rows))):
            for k in COLUMNS:
                state[k] = clamp(k, state[k] + rng.gauss(0, DRIFT[k] * per_row))
            rows.append({k: clamp(k, state[k] + rng.gauss(0, NOISE[k])) for k in COLUMNS})
    return rows


def write_files(name, rows):
    per_file = len(rows) // FILES_PER_CLASS
    for i in range(FILES_PER_CLASS):
        chunk = rows[i * per_file:(i + 1) * per_file]
        path = os.path.join(OUT_DIR, f"{name}.{i + 1:02d}.csv")
        with open(path, "w", newline="", encoding="ascii") as f:
            w = csv.writer(f)
            w.writerow(["timestamp"] + COLUMNS)
            for j, r in enumerate(chunk):
                w.writerow([j * ROW_MS, round(r["BPM"]), f"{r['SpO2']:.2f}", f"{r['Temp']:.2f}",
                            f"{r['RMSSD']:.2f}", f"{r['pNN50']:.2f}", f"{r['SignalQuality']:.2f}"])


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, cfg in CLASSES.items():
        n = cfg["seconds"] * 1000 // ROW_MS
        write_files(name, class_rows(rng, cfg, n))
    write_files("Poor_signal", poor_signal_rows(rng, 480 * 1000 // ROW_MS))
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
