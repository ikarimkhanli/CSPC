import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, spectrogram
import os

os.makedirs("figures", exist_ok=True)

vP, vS = 6.2, 3.6

d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z = d["Z"].to_numpy()
N = d["N"].to_numpy()
E = d["E"].to_numpy()
fs = 1 / (t[1] - t[0])
n = len(t)
print(f"fs = {fs:.1f} Hz, N = {n} samples, duration = {t[-1]:.1f} s")


# step 1
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
for a, x, name in zip(ax, [Z, N, E], ["Z", "N", "E"]):
    a.plot(t, x * 1e3, lw=0.5)
    a.set_ylabel(f"{name} (mm/s)")
ax[-1].set_xlabel("time (s)")
ax[0].set_title("Seismogram A, raw")
plt.tight_layout()
plt.savefig("figures/01_raw.png", dpi=150)
plt.close()

q = t < 60
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
for a, x, name in zip(ax, [Z, N, E], ["Z", "N", "E"]):
    a.plot(t[q], x[q] * 1e3, lw=0.5)
    a.set_ylabel(f"{name} (mm/s)")
ax[-1].set_xlabel("time (s)")
ax[0].set_title("First 60 s")
plt.tight_layout()
plt.savefig("figures/01_quiet_zoom.png", dpi=150)
plt.close()
print(f"quiet part max |Z| = {np.abs(Z[q]).max()*1e3:.4f} mm/s, whole max |Z| = {np.abs(Z).max()*1e3:.3f} mm/s")


# step 2
X = np.fft.fft(Z)
f = np.fft.fftfreq(n, d=1 / fs)
amp = np.abs(X) / n * 2

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(amp, lw=0.5)
ax[0].set_xlabel("array index")
ax[0].set_ylabel("amplitude (m/s)")
ax[1].plot(f, amp, lw=0.5)
ax[1].set_xlabel("frequency (Hz)")
ax[1].set_ylabel("amplitude (m/s)")
plt.tight_layout()
plt.savefig("figures/02a_spectrum_index_vs_freq.png", dpi=150)
plt.close()


def spectrum(x):
    m = len(x)
    ff = np.fft.fftfreq(m, d=1 / fs)
    aa = np.abs(np.fft.fft(x)) / m * 2
    p = ff > 0
    return ff[p], aa[p]


fq, aq = spectrum(Z[q])
fw, aw = spectrum(Z)
plt.figure(figsize=(9, 5))
plt.loglog(fw, aw, lw=0.5, label="whole record")
plt.loglog(fq, aq, lw=0.5, label="quiet part (first 60 s)")
plt.xlabel("frequency (Hz)")
plt.ylabel("amplitude (m/s)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/02b_spectrum_loglog.png", dpi=150)
plt.close()
print(f"quiet part: strongest at {fq[np.argmax(aq)]:.3f} Hz")
ratio_f = np.logspace(-1.5, 1.6, 12)
for lo, hi in zip(ratio_f[:-1], ratio_f[1:]):
    m1 = (fq >= lo) & (fq < hi)
    m2 = (fw >= lo) & (fw < hi)
    if m1.any() and m2.any():
        print(f"  {lo:7.3f}-{hi:7.3f} Hz  quiet {aq[m1].mean():.2e}  whole {aw[m2].mean():.2e}")


# step 3
print("ifft(fft(Z)) == Z :", np.allclose(np.fft.ifft(np.fft.fft(Z)).real, Z))


def bandpass(x, fs, a, b):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), 1 / fs)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0
    return np.fft.ifft(X).real


Zf = bandpass(Z, fs, 0.5, 10)
Nf = bandpass(N, fs, 0.5, 10)
Ef = bandpass(E, fs, 0.5, 10)

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
ax[0].plot(t, Z * 1e3, lw=0.5)
ax[0].set_ylabel("raw Z (mm/s)")
ax[1].plot(t, Zf * 1e3, lw=0.5)
ax[1].set_ylabel("filtered Z (mm/s)")
ax[2].plot(t, (Z - Zf) * 1e3, lw=0.5)
ax[2].set_ylabel("Z - clean (mm/s)")
ax[2].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("figures/03_filter.png", dpi=150)
plt.close()


# step 4
def envelope(x, w):
    return np.convolve(np.abs(x), np.ones(w) / w, mode="same")


w = int(0.5 * fs)
envZ = envelope(Zf, w)
H = np.sqrt(Nf**2 + Ef**2)
envH = envelope(H, w)

plt.figure(figsize=(10, 5))
for sec in [0.05, 0.5, 5]:
    plt.plot(t, envelope(Zf, int(sec * fs)) * 1e3, lw=0.8, label=f"w = {sec} s")
plt.xlabel("time (s)")
plt.ylabel("envelope of filtered Z (mm/s)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/04_envelope_windows.png", dpi=150)
plt.close()


# step 5
tP = t[np.argmax(envZ > 0.05 * envZ.max())]
after = t > tP + 2
tS = t[after][np.argmax(envH[after] > 0.10 * envH.max())]
peaks, _ = find_peaks(envH, height=0.5 * envH.max(), distance=int(5 * fs))
t_max = t[peaks[np.argmax(envH[peaks])]]
print(f"tP = {tP:.2f} s, tS = {tS:.2f} s, horizontal envelope max at {t_max:.2f} s")

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 9))
for a, x, name in zip(ax, [Zf, Nf, Ef], ["Z", "N", "E"]):
    a.plot(t, x * 1e3, lw=0.5)
    a.axvline(tP, color="r", label="P")
    a.axvline(tS, color="g", label="S")
    a.set_ylabel(f"{name} filt. (mm/s)")
ax[0].legend(loc="upper right")
ax[3].plot(t, envZ * 1e3, label="env Z")
ax[3].plot(t, envH * 1e3, label="env horizontal")
ax[3].axvline(tP, color="r")
ax[3].axvline(tS, color="g")
ax[3].axvline(t_max, color="k", ls="--", label="max of env H")
ax[3].set_ylabel("envelope (mm/s)")
ax[3].set_xlabel("time (s)")
ax[3].legend(loc="upper right")
plt.tight_layout()
plt.savefig("figures/05_arrivals.png", dpi=150)
plt.close()


# step 6
k = 1 / vS - 1 / vP
R = (tS - tP) / k
dR = 1 / k
t0 = tP - R / vP
R_max = (t_max - tP) / k
print(f"R = {R:.1f} km, 1 s error in tS -> {dR:.1f} km, origin time = {t0:.2f} s")
print(f"(using the maximum instead: R = {R_max:.1f} km)")


# step 7
fsg, tsg, S = spectrogram(Z, fs=fs, nperseg=256)
m = fsg <= 10
plt.figure(figsize=(10, 5))
plt.pcolormesh(tsg, fsg[m], 10 * np.log10(S[m]), shading="auto")
plt.colorbar(label="power (dB)")
plt.axvline(tP, color="r", ls="--", lw=0.8)
plt.axvline(tS, color="w", ls="--", lw=0.8)
plt.xlabel("time (s)")
plt.ylabel("frequency (Hz)")
plt.title("Spectrogram of raw Z")
plt.tight_layout()
plt.savefig("figures/07_spectrogram.png", dpi=150)
plt.close()

for name, lo, hi in [("P", tP, tP + 5), ("S", tS, tS + 5), ("late", tS + 40, tS + 80)]:
    sel = (tsg >= lo) & (tsg < hi) & True
    if sel.any():
        Sm = S[m][:, sel].mean(axis=1)
        print(f"  {name} window: dominant frequency {fsg[m][np.argmax(Sm)]:.2f} Hz")


# step 8
PGV = np.max(np.sqrt(N**2 + E**2)) * 100
M = (np.log10(PGV) + np.log10(R) + 1.0) / 0.55
print(f"PGV = {PGV:.4f} cm/s, M = {M:.2f}")
print(f"if log10 R were ignored, at 2R the magnitude changes by {-np.log10(2)/0.55:.2f}")
print(f"one unit larger -> PGV x {10**0.55:.2f} = {PGV*10**0.55:.4f} cm/s")
