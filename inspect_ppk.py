#!/usr/bin/env python3
"""
inspect_ppk.py

Inspect an RTKLIB .pos solution (from rnx2rtkp) and dump a full set of
diagnostic plots + a text summary into an output folder.

Built for the Husky RTK workflow: collect SBP outdoors -> sbp2rinex ->
rnx2rtkp -> this. Handles BOTH regimes automatically:
  * stationary  -> scatter-about-mean, CEP/RMS accuracy, convergence
  * driving     -> track, speed, cumulative distance
and always emits the core health plots (fix rate, sats, stdev, ratio, age).

Usage
-----
    python3 inspect_ppk.py my_ppk.pos -o plots/
    python3 inspect_ppk.py my_ppk.pos --output plots/ --fixed-only

Expects the default RTKLIB lat/lon/height .pos format:
    week tow lat lon height Q ns sdn sde sdu sdne sdeu sdun age ratio
(i.e. run rnx2rtkp WITHOUT -e/-a; ENU is derived here.)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless: save files, no display needed
import matplotlib.pyplot as plt

# ---- WGS84 constants ---------------------------------------------------------
_A = 6378137.0                   # semi-major axis (m)
_F = 1.0 / 298.257223563         # flattening
_E2 = _F * (2 - _F)              # eccentricity^2

Q_LABEL = {1: "fix", 2: "float", 3: "sbas", 4: "dgps", 5: "single", 6: "ppp"}
Q_COLOR = {1: "#2ca02c", 2: "#ff7f0e", 3: "#9467bd", 4: "#1f77b4",
           5: "#d62728", 6: "#8c564b"}

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


# ---- parsing -----------------------------------------------------------------
def load_pos(path):
    """Parse an RTKLIB lat/lon/height .pos file into a DataFrame."""
    cols = ["week", "tow", "lat", "lon", "height", "Q", "ns",
            "sdn", "sde", "sdu", "sdne", "sdeu", "sdun", "age", "ratio"]
    df = pd.read_csv(path, comment="%", sep=r"\s+", names=cols, engine="python")
    # coerce numerics; drop any malformed rows
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "height", "Q"]).reset_index(drop=True)
    df["Q"] = df["Q"].astype(int)
    if len(df) == 0:
        sys.exit("No solution epochs parsed. Is this a lat/lon/height .pos file?")
    # GPST -> a real (approximate, leap-seconds ignored) datetime for the x-axis
    df["t"] = [GPS_EPOCH + timedelta(weeks=int(w), seconds=float(s))
               for w, s in zip(df["week"], df["tow"])]
    # elapsed seconds from start, for regime math & speed
    t0 = df["tow"].iloc[0] + df["week"].iloc[0] * 604800.0
    df["elapsed"] = (df["tow"] + df["week"] * 604800.0) - t0
    return df


# ---- geodetic -> local ENU ---------------------------------------------------
def lla_to_ecef(lat_deg, lon_deg, h):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sinlat, coslat = np.sin(lat), np.cos(lat)
    N = _A / np.sqrt(1 - _E2 * sinlat**2)
    x = (N + h) * coslat * np.cos(lon)
    y = (N + h) * coslat * np.sin(lon)
    z = (N * (1 - _E2) + h) * sinlat
    return np.vstack([x, y, z]).T


def ecef_to_enu(ecef, lat0_deg, lon0_deg, origin_ecef):
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    d = ecef - origin_ecef
    slat, clat = np.sin(lat0), np.cos(lat0)
    slon, clon = np.sin(lon0), np.cos(lon0)
    R = np.array([
        [-slon,          clon,         0.0],
        [-slat * clon,  -slat * slon,  clat],
        [ clat * clon,   clat * slon,  slat],
    ])
    enu = (R @ d.T).T
    return enu[:, 0], enu[:, 1], enu[:, 2]   # E, N, U


def add_enu(df):
    """Add E/N/U columns relative to the mean position (local tangent plane)."""
    lat0, lon0, h0 = df["lat"].mean(), df["lon"].mean(), df["height"].mean()
    origin = lla_to_ecef(np.array([lat0]), np.array([lon0]), np.array([h0]))[0]
    ecef = lla_to_ecef(df["lat"].values, df["lon"].values, df["height"].values)
    e, n, u = ecef_to_enu(ecef, lat0, lon0, origin)
    df["E"], df["N"], df["U"] = e, n, u
    return df, (lat0, lon0, h0)


# ---- regime detection --------------------------------------------------------
def detect_regime(df):
    """Return 'stationary' or 'driving' + a few motion metrics."""
    # horizontal distance of each epoch from the centroid
    r = np.hypot(df["E"] - df["E"].mean(), df["N"] - df["N"].mean())
    spread95 = np.percentile(r, 95)
    # total path length
    de = np.diff(df["E"].values)
    dn = np.diff(df["N"].values)
    path_len = np.nansum(np.hypot(de, dn))
    # heuristic: if the 95th-pct spread is small AND path is short, it's static
    regime = "stationary" if (spread95 < 1.0 and path_len < 5.0) else "driving"
    return regime, {"spread95_m": spread95, "path_len_m": path_len}


# ---- plotting helpers --------------------------------------------------------
def _qc(df):
    return df["Q"].map(Q_COLOR).fillna("#333333")


def plot_fix_rate(df, out):
    counts = df["Q"].value_counts().sort_index()
    labels = [f"{Q_LABEL.get(q, q)} ({q})" for q in counts.index]
    colors = [Q_COLOR.get(q, "#333") for q in counts.index]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts.values, color=colors)
    total = counts.sum()
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{100*c/total:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("epochs")
    ax.set_title(f"Solution quality distribution  (n={total})")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_track(df, out):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["E"], df["N"], s=6, c=_qc(df), linewidths=0)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("Horizontal track (colored by fix quality)")
    ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.3)
    _q_legend(ax, df)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_enu_time(df, out):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, comp, lbl in zip(axs, ["E", "N", "U"], ["East", "North", "Up"]):
        ax.scatter(df["t"], df[comp] - df[comp].mean(), s=4, c=_qc(df), linewidths=0)
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True, alpha=0.3)
    axs[0].set_title("ENU position vs time (mean-removed, colored by Q)")
    axs[-1].set_xlabel("time (GPST)")
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_health(df, out):
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axs[0].scatter(df["t"], df["ns"], s=4, c=_qc(df), linewidths=0)
    axs[0].set_ylabel("# sats"); axs[0].set_title("Satellites used")
    axs[1].plot(df["t"], df["sdn"], lw=0.6, label="sdN")
    axs[1].plot(df["t"], df["sde"], lw=0.6, label="sdE")
    axs[1].plot(df["t"], df["sdu"], lw=0.6, label="sdU")
    axs[1].set_yscale("log"); axs[1].set_ylabel("stdev (m)")
    axs[1].legend(loc="upper right", fontsize=8); axs[1].set_title("Position stdev")
    axs[2].scatter(df["t"], df["ratio"], s=4, c=_qc(df), linewidths=0)
    axs[2].axhline(3.0, color="k", ls="--", lw=0.8, label="AR thresh 3.0")
    axs[2].set_ylabel("AR ratio"); axs[2].legend(fontsize=8)
    axs[2].set_title("Ambiguity-resolution ratio")
    axs[3].plot(df["t"], df["age"], lw=0.6, color="#555")
    axs[3].set_ylabel("age (s)"); axs[3].set_title("Age of corrections")
    axs[3].set_xlabel("time (GPST)")
    for ax in axs:
        ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_stationary(df, out, fixed_only):
    d = df[df["Q"] == 1] if fixed_only else df
    if len(d) < 3:
        d = df
    e = (d["E"] - d["E"].mean()).values
    n = (d["N"] - d["N"].mean()).values
    u = (d["U"] - d["U"].mean()).values
    r = np.hypot(e, n)
    cep50 = np.percentile(r, 50)
    cep95 = np.percentile(r, 95)
    rms_h = np.sqrt(np.mean(e**2 + n**2))
    rms_u = np.sqrt(np.mean(u**2))

    fig = plt.figure(figsize=(12, 6))
    # scatter with CEP circles
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(e, n, s=8, c=_qc(d), linewidths=0, alpha=0.6)
    for rad, lbl, col in [(cep50, "CEP50", "#2ca02c"), (cep95, "CEP95", "#d62728")]:
        circ = plt.Circle((0, 0), rad, fill=False, color=col, ls="--", label=lbl)
        ax1.add_patch(circ)
    ax1.set_aspect("equal", "box"); ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("East err (m)"); ax1.set_ylabel("North err (m)")
    ax1.set_title("Horizontal scatter about mean"); ax1.legend(fontsize=8)
    # up histogram
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(u, bins=40, color="#1f77b4", alpha=0.8)
    ax2.set_xlabel("Up err (m)"); ax2.set_ylabel("epochs")
    ax2.set_title("Vertical error distribution")
    txt = (f"n={len(d)} ({'fixed only' if fixed_only else 'all Q'})\n"
           f"RMS horiz = {rms_h:.3f} m\nRMS up = {rms_u:.3f} m\n"
           f"CEP50 = {cep50:.3f} m\nCEP95 = {cep95:.3f} m\n"
           f"std E/N/U = {e.std():.3f}/{n.std():.3f}/{u.std():.3f} m")
    ax2.text(0.02, 0.98, txt, transform=ax2.transAxes, va="top", fontsize=9,
             family="monospace", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return dict(rms_h=rms_h, rms_u=rms_u, cep50=cep50, cep95=cep95,
                std_e=float(e.std()), std_n=float(n.std()), std_u=float(u.std()),
                n_used=len(d))


def plot_driving(df, out):
    # speed from consecutive ENU + dt
    de = np.diff(df["E"].values); dn = np.diff(df["N"].values)
    du = np.diff(df["U"].values); dt = np.diff(df["elapsed"].values)
    dt[dt == 0] = np.nan
    speed = np.hypot(de, dn) / dt
    dist = np.concatenate([[0], np.cumsum(np.hypot(de, dn))])
    tmid = df["t"].values[1:]

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axs[0].plot(tmid, speed, lw=0.6, color="#1f77b4")
    axs[0].set_ylabel("speed (m/s)"); axs[0].set_title("Horizontal speed")
    axs[0].grid(True, alpha=0.3)
    axs[1].plot(df["t"], dist, lw=0.8, color="#2ca02c")
    axs[1].set_ylabel("cumulative dist (m)"); axs[1].set_xlabel("time (GPST)")
    axs[1].set_title("Distance travelled"); axs[1].grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return dict(path_len_m=float(dist[-1]),
                speed_max=float(np.nanmax(speed)),
                speed_mean=float(np.nanmean(speed)))


def _q_legend(ax, df):
    present = sorted(df["Q"].unique())
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=Q_COLOR.get(q, "#333"),
               label=f"{Q_LABEL.get(q, q)} ({q})") for q in present]
    ax.legend(handles=handles, fontsize=8, loc="best")


# ---- report ------------------------------------------------------------------
def write_report(path, df, origin, regime, motion, stat, drive):
    total = len(df)
    qc = df["Q"].value_counts().sort_index()
    with open(path, "w") as f:
        f.write("PPK solution inspection report\n")
        f.write("=" * 50 + "\n")
        f.write(f"epochs           : {total}\n")
        f.write(f"time span        : {df['t'].iloc[0]} -> {df['t'].iloc[-1]} GPST\n")
        f.write(f"duration         : {df['elapsed'].iloc[-1]:.1f} s\n")
        f.write(f"origin (mean LLA): {origin[0]:.9f}, {origin[1]:.9f}, {origin[2]:.3f}\n")
        f.write(f"regime detected  : {regime}  "
                f"(spread95={motion['spread95_m']:.2f} m, "
                f"path={motion['path_len_m']:.1f} m)\n\n")
        f.write("fix quality:\n")
        for q, c in qc.items():
            f.write(f"   Q={q} {Q_LABEL.get(q,'?'):7s} {c:6d}  {100*c/total:5.1f}%\n")
        fixrate = 100 * qc.get(1, 0) / total
        f.write(f"   -> fixed rate  : {fixrate:.1f}%\n\n")
        f.write(f"satellites  mean/min/max : "
                f"{df['ns'].mean():.1f} / {df['ns'].min()} / {df['ns'].max()}\n")
        f.write(f"AR ratio    mean         : {df['ratio'].mean():.2f}\n\n")
        if stat:
            f.write("stationary accuracy (about mean):\n")
            for k, v in stat.items():
                f.write(f"   {k:8s} : {v:.4f}\n" if isinstance(v, float)
                        else f"   {k:8s} : {v}\n")
        if drive:
            f.write("driving metrics:\n")
            for k, v in drive.items():
                f.write(f"   {k:12s} : {v:.3f}\n")
    print(f"wrote {path}")


# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Inspect an RTKLIB .pos solution.")
    ap.add_argument("pos", help="input .pos file (lat/lon/height format)")
    ap.add_argument("-o", "--output", default="ppk_plots",
                    help="output folder for plots + report (default: ppk_plots)")
    ap.add_argument("--fixed-only", action="store_true",
                    help="compute stationary accuracy from fixed (Q=1) epochs only")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = load_pos(args.pos)
    df, origin = add_enu(df)
    regime, motion = detect_regime(df)
    print(f"loaded {len(df)} epochs; regime = {regime}")

    op = lambda name: os.path.join(args.output, name)

    # always-on health/overview plots
    plot_fix_rate(df, op("01_fix_rate.png"))
    plot_track(df, op("02_track.png"))
    plot_enu_time(df, op("03_enu_time.png"))
    plot_health(df, op("04_health.png"))

    stat = drive = None
    # both panels are cheap; emit the one that matches the regime, and always
    # emit the stationary accuracy panel too (useful even on slow drives).
    stat = plot_stationary(df, op("05_stationary_accuracy.png"), args.fixed_only)
    if regime == "driving":
        drive = plot_driving(df, op("06_driving.png"))

    write_report(op("report.txt"), df, origin, regime, motion, stat, drive)
    print(f"done -> {args.output}/")


if __name__ == "__main__":
    main()
