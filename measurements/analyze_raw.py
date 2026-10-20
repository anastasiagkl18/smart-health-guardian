import re, sys, datetime
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = sys.argv[1]
VICKO = sys.argv[2]
OUT = sys.argv[3]

R, B, pc_of_t = [], [], {}
for line in open(LOG, encoding="utf-8-sig", errors="replace"):
    pc, s = line[:8], line[9:].strip().replace("\x00", "")
    m = re.match(r"R,(\d+),(-?\d+),(-?\d+)$", s)
    if m:
        t, ir, red = map(int, m.groups())
        R.append((t, ir, red)); pc_of_t[t] = pc
        continue
    m = re.match(r"B,([\d.]+)$", s)
    if m:
        B.append(float(m.group(1)))

R = np.array(R, dtype=float)
# keep the run after the last clock reset
reset = np.where(np.diff(R[:, 0]) < 0)[0]
if len(reset):
    R = R[reset[-1] + 1:]
t, ir, red = R[:, 0], R[:, 1], R[:, 2]
B = np.array([b for b in B if t[0] <= b <= t[-1]])
finger = ir > 7000
first = np.argmax(finger)
t, ir, red = t[first:], ir[first:], red[first:]
fs = 100.0

# ---- offline reference: zero-phase band-pass, systolic peaks, max-slope fiducial ----
b, a = butter(2, [0.5 / (fs / 2), 4.0 / (fs / 2)], btype="band")
y = filtfilt(b, a, -(ir - ir.mean()))
dy = np.gradient(y)
peaks, prop = find_peaks(y, distance=int(0.33 * fs), prominence=0)
prom = prop["prominences"]
# adaptive prominence: keep peaks with prominence > 35% of the local (10 s) median
keep = []
for i, p in enumerate(peaks):
    win = (peaks > p - 5 * fs) & (peaks < p + 5 * fs)
    if prom[i] > 0.35 * np.median(prom[win]):
        keep.append(p)
peaks = np.array(keep)
ref = []
for p in peaks:
    lo = max(1, p - int(0.25 * fs))
    k = lo + np.argmax(dy[lo:p + 1])
    if 1 <= k < len(dy) - 1:
        a_, b_, c_ = dy[k - 1], dy[k], dy[k + 1]
        den = a_ - 2 * b_ + c_
        off = 0.5 * (a_ - c_) / den if den != 0 else 0.0
        ref.append(t[k] + np.clip(off, -0.5, 0.5) * 10)
ref = np.array(ref)
ref_ibi = np.diff(ref)

# ---- compare device beats with reference ----
def match(dev, ref, tol=150):
    hits, errs = 0, []
    used = set()
    for d in dev:
        j = np.argmin(np.abs(ref - d))
        if abs(ref[j] - d) <= tol and j not in used:
            used.add(j); hits += 1; errs.append(d - ref[j])
    return hits, np.array(errs)

hits, errs = match(B, ref)
print(f"recording: {(t[-1]-t[0])/1000:.0f} s, IR DC ~{np.median(ir):.0f}, PI ~{100*np.std(y)*2.8/np.median(ir):.2f}%")
print(f"reference beats: {len(ref)}  mean HR {60000/np.median(ref_ibi):.1f} bpm, IBI range {ref_ibi.min():.0f}-{ref_ibi.max():.0f} ms")
good = ref_ibi[(ref_ibi > 300) & (ref_ibi < 1500)]
sd = np.diff(good)
print(f"reference HRV: RMSSD {np.sqrt(np.mean(sd**2)):.1f} ms, pNN50 {100*np.mean(np.abs(sd)>50):.1f}%  (all intervals)")
print(f"device beats: {len(B)}  matched {hits} ({100*hits/len(ref):.0f}% of reference), "
      f"missed {len(ref)-hits}, extra {len(B)-hits}")
if len(errs):
    print(f"device timing error vs reference: median {np.median(errs):.1f} ms, spread (std) {np.std(errs):.1f} ms")

# ---- Vicko comparison (PC time) ----
pcs = np.array([pc_of_t.get(int(x), None) for x in t])
def pc_to_sec(s):
    h, m, ss = map(int, s.split(":")); return h * 3600 + m * 60 + ss
tsec = np.array([pc_to_sec(p) if p else np.nan for p in pcs])
print("\nVicko vs reference HR (10 s around each reading):")
for line in open(VICKO):
    hh, spo2, hr = line.split()
    s = pc_to_sec(hh)
    idx = np.where((tsec >= s - 10) & (tsec <= s))[0]
    if len(idx) < 50:
        continue
    t0, t1 = t[idx[0]], t[idx[-1]]
    ib = ref_ibi[(ref[1:] >= t0) & (ref[1:] <= t1)]
    ib = ib[(ib > 300) & (ib < 1500)]
    print(f"  {hh}  Vicko {hr} bpm   reference {60000/np.median(ib):.1f} bpm")

# ---- plots ----
dev_ibi = np.diff(B)
fig, ax = plt.subplots(3, 1, figsize=(14, 10))
ax[0].plot(ref[1:] / 1000, ref_ibi, ".", ms=3, label="reference IBI")
ax[0].plot(B[1:] / 1000, dev_ibi, "x", ms=3, label="device IBI")
ax[0].set_ylim(0, 2000); ax[0].set_ylabel("IBI ms"); ax[0].legend(); ax[0].set_title("Beat intervals")
# segment with the worst device misses
long = np.where(dev_ibi > 1.6 * np.median(ref_ibi))[0]
centre = B[long[len(long) // 2]] if len(long) else t[len(t) // 2]
for k, (c, title) in enumerate([(t[len(t) // 3], "typical segment"), (centre, "segment with device misses")]):
    m = (t > c - 6000) & (t < c + 6000)
    axk = ax[k + 1]
    axk.plot(t[m] / 1000, y[m], lw=0.8, label="band-pass IR (offline)")
    rr = ref[(ref > c - 6000) & (ref < c + 6000)]
    bb = B[(B > c - 6000) & (B < c + 6000)]
    axk.plot(rr / 1000, np.interp(rr, t, y), "go", ms=5, label="reference beat")
    axk.plot(bb / 1000, np.interp(bb, t, y), "rx", ms=8, label="device beat")
    axk.set_title(title); axk.legend(loc="upper right")
plt.tight_layout(); plt.savefig(OUT, dpi=80)
np.savez(OUT.replace(".png", ".npz"), t=t, ir=ir, red=red, ref=ref, dev=B)
print("plot:", OUT)
