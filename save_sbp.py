#!/usr/bin/env python3
"""
sbp_raw_logger.py

Log raw SBP observation + ephemeris data from a Swift Navigation Duro/Piksi
receiver to a binary .sbp file, suitable for post-processing (sbp2rinex ->
RINEX -> RTKLIB) and for later tight-coupled GNSS/IMU work.

Usage examples
--------------
    # rover, default IP/port, log into current directory
    python3 sbp_raw_logger.py rover

    # base, save into a specific folder, show live output
    python3 sbp_raw_logger.py base --folder ~/rtk_logs --verbose

    # override host/port (e.g. the dedicated obs TCP server you enabled)
    python3 sbp_raw_logger.py rover --host 192.168.131.31 --port 55555

    # log EVERY message on the stream, not just the tight-coupling set
    python3 sbp_raw_logger.py base --all --verbose

Notes
-----
* Defaults follow the Clearpath layout: base Duro = 192.168.131.30,
  rover (UGV position) Duro = 192.168.131.31, SBP TCP server on port 55555.
  Override with --host/--port if you enabled the obs/ephemeris messages on a
  different TCP server port.
* The receiver only *sends* a message type if it is enabled in that port's
  `enabled_sbp_messages` setting in Swift Console. This script filters what it
  writes, but it cannot capture a message the Duro was never told to output.
  Enable at least: MSG_OBS, the MSG_EPHEMERIS_* family, MSG_BASE_POS_*,
  MSG_GLO_BIASES, MSG_IONO (and GPS time) on the port you log from.
* The Duro allows ONE TCP client per port. If you get "connection refused",
  the ROS driver (or someone's Swift Console) already holds that port. Log
  from a dedicated obs port, or stop the competing client.
* Output is byte-accurate framed SBP (0x55-preamble frames), which sbp2rinex
  reads directly:  sbp2rinex rover_raw_YYYYmmdd_HHMMSS.sbp
"""

import argparse
import os
import signal
import sys
import time
from collections import Counter
from datetime import datetime

# ---- libsbp imports (fail loudly with install hint) -------------------------
try:
    from sbp.client.drivers.network_drivers import TCPDriver
    from sbp.client import Handler, Framer
except ImportError:
    sys.stderr.write(
        "ERROR: libsbp is not installed.\n"
        "  pip3 install sbp   (or: pip3 install --break-system-packages sbp)\n"
    )
    sys.exit(1)

# ---- role -> default connection ---------------------------------------------
ROLE_DEFAULTS = {
    "base":  {"host": "192.168.131.30", "port": 55555},
    "rover": {"host": "192.168.131.31", "port": 55555},
}

# ---- the message types we keep for tight coupling ---------------------------
# Resolved by NAME from libsbp so we don't depend on hard-coded numeric IDs
# (they differ across constellations and SBP versions). Anything that fails to
# import is simply skipped; a warning is printed. If NONE resolve, we fall back
# to logging everything so we never silently drop needed data.
_WANTED_NAMES = [
    # raw observations
    "SBP_MSG_OBS",
    # broadcast ephemerides, per constellation
    "SBP_MSG_EPHEMERIS_GPS",
    "SBP_MSG_EPHEMERIS_GLO",
    "SBP_MSG_EPHEMERIS_BDS",
    "SBP_MSG_EPHEMERIS_GAL",
    "SBP_MSG_EPHEMERIS_QZSS",
    "SBP_MSG_EPHEMERIS_SBAS",
    # base station position (required for pseudo-absolute RTK / PPK)
    "SBP_MSG_BASE_POS_ECEF",
    "SBP_MSG_BASE_POS_LLH",
    # inter-constellation biases + ionosphere + time
    "SBP_MSG_GLO_BIASES",
    "SBP_MSG_IONO",
    "SBP_MSG_GPS_TIME",
]

# best-effort friendly names for the live/summary printout, keyed by type id
_NAME_BY_TYPE = {}


def build_wanted_type_set():
    """Return a set of numeric SBP message-type ids to keep, resolved by name."""
    import importlib

    wanted = set()
    obs_mod = importlib.import_module("sbp.observation")
    # SBP_MSG_OBS lives in sbp.observation; a couple of others too. Search a few
    # modules so we catch them regardless of where the constant is defined.
    search_mods = ["sbp.observation", "sbp.navigation", "sbp.system"]
    resolved = {}
    for name in _WANTED_NAMES:
        found = None
        for mod_name in search_mods:
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue
            if hasattr(mod, name):
                found = getattr(mod, name)
                break
        if found is None:
            sys.stderr.write(f"  (note) could not resolve {name}; skipping it\n")
            continue
        resolved[name] = found
        wanted.add(found)
        _NAME_BY_TYPE[found] = name.replace("SBP_MSG_", "")
    return wanted


def make_filename(role, folder):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{role}_raw_{stamp}.sbp"
    if folder:
        folder = os.path.expanduser(folder)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, fname)
    return fname


def main():
    ap = argparse.ArgumentParser(
        description="Log raw SBP obs/ephemeris from a Duro to a .sbp file."
    )
    ap.add_argument("role", choices=["base", "rover"],
                    help="which receiver you are logging (sets default host/port)")
    ap.add_argument("--folder", default="",
                    help="output folder (default: current directory)")
    ap.add_argument("--host", default=None,
                    help="override receiver IP (default depends on role)")
    ap.add_argument("--port", type=int, default=None,
                    help="override SBP TCP port (default 55555)")
    ap.add_argument("--all", action="store_true",
                    help="log EVERY message on the stream, not just the "
                         "tight-coupling set")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print live status as data arrives")
    ap.add_argument("--status-every", type=float, default=2.0,
                    help="seconds between live status lines in verbose mode")
    args = ap.parse_args()

    host = args.host or ROLE_DEFAULTS[args.role]["host"]
    port = args.port or ROLE_DEFAULTS[args.role]["port"]

    # decide what to keep
    if args.all:
        wanted = None  # keep everything
    else:
        wanted = build_wanted_type_set()
        if not wanted:
            sys.stderr.write(
                "WARNING: no wanted message names resolved from libsbp; "
                "falling back to logging ALL messages.\n"
            )
            wanted = None

    out_path = make_filename(args.role, args.folder)

    print(f"role      : {args.role}")
    print(f"receiver  : {host}:{port}")
    print(f"output    : {out_path}")
    print(f"filter    : {'ALL messages' if wanted is None else sorted(_NAME_BY_TYPE[t] for t in wanted)}")
    print("connecting... (Ctrl-C to stop)\n")

    counts = Counter()
    total_msgs = 0
    total_bytes = 0
    start = time.time()
    last_status = start
    stop = {"flag": False}

    def handle_sigint(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        with open(out_path, "wb") as logf:
            with TCPDriver(host, port) as driver:
                with Handler(Framer(driver.read, driver.write)) as source:
                    for msg, metadata in source:
                        if stop["flag"]:
                            break
                        mtype = msg.msg_type
                        if wanted is not None and mtype not in wanted:
                            continue
                        # byte-accurate framed SBP to disk
                        frame = msg.pack()
                        logf.write(frame)
                        total_bytes += len(frame)
                        total_msgs += 1
                        counts[mtype] += 1

                        if args.verbose:
                            now = time.time()
                            if now - last_status >= args.status_every:
                                last_status = now
                                elapsed = now - start
                                # show obs sat count if this was an OBS msg
                                sats = ""
                                try:
                                    if _NAME_BY_TYPE.get(mtype) == "OBS":
                                        sats = f"  sats_this_obs={len(msg.obs)}"
                                except Exception:
                                    pass
                                breakdown = ", ".join(
                                    f"{_NAME_BY_TYPE.get(t, hex(t))}={c}"
                                    for t, c in sorted(counts.items())
                                )
                                print(
                                    f"[{elapsed:6.1f}s] msgs={total_msgs} "
                                    f"bytes={total_bytes} rate={total_msgs/elapsed:5.1f}/s"
                                    f"{sats}\n           {breakdown}"
                                )
    except Exception as e:
        sys.stderr.write(f"\nERROR: {e}\n")
        if "refused" in str(e).lower():
            sys.stderr.write(
                "  -> port busy. The ROS driver or another Swift Console likely "
                "holds this port (one TCP client per port).\n"
                "     Log from a dedicated obs port, or stop the other client.\n"
            )
        # fall through to summary so partial logs are still reported

    elapsed = max(time.time() - start, 1e-6)
    print("\n---- summary ----------------------------------------------------")
    print(f"output file : {out_path}")
    print(f"duration    : {elapsed:.1f} s")
    print(f"messages    : {total_msgs}  ({total_bytes} bytes)")
    for t, c in sorted(counts.items()):
        print(f"   {_NAME_BY_TYPE.get(t, hex(t)):22s} {c}")
    if total_msgs == 0:
        print("NO messages captured. Check that the obs/ephemeris messages are")
        print("enabled on this port in Swift Console, and that the port is free.")
    else:
        print(f"\nnext: sbp2rinex {out_path}")
    print("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()
