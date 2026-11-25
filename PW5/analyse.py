import os
import numpy as np
import pandas as pd

vP, vS = 6.2, 3.6
A_CUT, B_CUT = 0.5, 10.0
WINDOW = 0.5


def bandpass(x, fs, a, b):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), 1 / fs)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0
    return np.fft.ifft(X).real


def envelope(x, w):
    return np.convolve(np.abs(x), np.ones(w) / w, mode="same")


def analyse(csv_file):
    d = pd.read_csv(csv_file)
    t = d["time_s"].to_numpy()
    Z = d["Z"].to_numpy()
    N = d["N"].to_numpy()
    E = d["E"].to_numpy()
    fs = 1 / (t[1] - t[0])

    Zf = bandpass(Z, fs, A_CUT, B_CUT)
    Nf = bandpass(N, fs, A_CUT, B_CUT)
    Ef = bandpass(E, fs, A_CUT, B_CUT)

    w = int(WINDOW * fs)
    envZ = envelope(Zf, w)
    envH = envelope(np.sqrt(Nf**2 + Ef**2), w)

    tP = t[np.argmax(envZ > 0.05 * envZ.max())]
    after = t > tP + 2
    tS = t[after][np.argmax(envH[after] > 0.10 * envH.max())]

    R = (tS - tP) / (1 / vS - 1 / vP)
    t0 = tP - R / vP

    PGV = np.max(np.sqrt(N**2 + E**2)) * 100
    M = (np.log10(PGV) + np.log10(R) + 1.0) / 0.55

    return {"tP_s": tP, "tS_s": tS, "R_km": R, "origin_s": t0, "PGV_cm_s": PGV, "M": M}


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rows = []
    for name in ["A", "B"]:
        res = analyse(f"data/seismogram_{name}.csv")
        res = {"event": name, **res}
        rows.append(res)
    table = pd.DataFrame(rows).round(3)
    table.to_csv("results/events.csv", index=False)
    print(table.to_string(index=False))
