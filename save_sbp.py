#!/usr/bin/env python3
"""
sbp_raw_logger.py  (fail-loud version)

Log raw SBP observation + ephemeris data from a Swift Navigation Duro/Piksi
to a binary .sbp file for sbp2rinex, and check the stream live so a bad log
is obvious while you are still in the field, not after conversion.

Usage
-----
    python3 sbp_raw_logger.py rover
    python3 sbp_raw_logger.py base --folder ~/rtk_logs --verbose
    python3 sbp_raw_logger.py rover --host 192.168.131.31 --port 55555 --strict

Live checks (printed to stderr when they start and when they clear)
-------------------------------------------------------------------
  critical:
    NO_STREAM   no SBP messages at all for --stall seconds
    NO_OBS      no MSG_OBS within --obs-timeout s of connecting
    EMPTY_OBS   MSG_OBS arrives but no epoch with a valid GPS week and
                >= --min-signals signals for --obs-timeout seconds
    NO_EPH      no ephemeris message within --eph-timeout seconds
  warning:
    NO_FIX      MSG_POS_LLH reports no solution for --fix-timeout seconds
    RX_ERROR    heartbeat reports a system / IO / SwiftNAP error
    ANT_SHORT   heartbeat reports an external antenna short
    NO_ANT      heartbeat reports no external antenna
  Receiver MSG_LOG text at WARN level or worse is echoed live.

All messages are monitored; only the obs/ephemeris set is written (unless
--all). POS_LLH, HEARTBEAT and LOG must be enabled on the port for the
NO_FIX / antenna / log checks to work.

The summary is printed and also saved to <output>.summary.txt.

Exit codes: 0 ok | 1 critical problem seen (only with --strict)
            2 could not connect | 3 connected but received nothing

Note: the reachability probe opens and immediately closes one TCP
connection to the port before the real connection.
"""

import argparse
import os
import socket
import sys
import threading
import time
from collections import Counter
from datetime import datetime

try:
    from sbp.client.drivers.network_drivers import TCPDriver
    from sbp.client import Handler, Framer
    import sbp.observation as sbp_obs
    import sbp.navigation as sbp_nav
    import sbp.logging as sbp_log
    import sbp.system as sbp_sys
except ImportError:
    sys.stderr.write(
        "ERROR: libsbp is not installed.\n"
        "  pip3 install sbp   (or: pip3 install --break-system-packages sbp)\n"
    )
    sys.exit(1)

ROLE_DEFAULTS = {
    "base": {"host": "192.168.131.30", "port": 55555},
    "rover": {"host": "192.168.131.31", "port": 55555},
}

# ---- message types ----------------------------------------------------------
MSG_OBS = sbp_obs.SBP_MSG_OBS
MSG_POS_LLH = sbp_nav.SBP_MSG_POS_LLH
MSG_LOG = sbp_log.SBP_MSG_LOG
MSG_HEARTBEAT = sbp_sys.SBP_MSG_HEARTBEAT

EPH_TYPES = {}
for _c in ("GPS", "GLO", "BDS", "GAL", "QZSS", "SBAS"):
    _t = getattr(sbp_obs, f"SBP_MSG_EPHEMERIS_{_c}", None)
    if _t is not None:
        EPH_TYPES[_t] = _c

EXTRA_KEEP = {}
for _mod, _name in (
    (sbp_obs, "SBP_MSG_BASE_POS_ECEF"),
    (sbp_obs, "SBP_MSG_BASE_POS_LLH"),
    (sbp_obs, "SBP_MSG_GLO_BIASES"),
    (sbp_obs, "SBP_MSG_IONO"),
    (sbp_nav, "SBP_MSG_GPS_TIME"),
):
    _t = getattr(_mod, _name, None)
    if _t is not None:
        EXTRA_KEEP[_t] = _name[len("SBP_MSG_") :]

NAMES = {
    MSG_OBS: "OBS",
    MSG_POS_LLH: "POS_LLH",
    MSG_LOG: "LOG",
    MSG_HEARTBEAT: "HEARTBEAT",
}
NAMES.update({t: f"EPH_{c}" for t, c in EPH_TYPES.items()})
NAMES.update(EXTRA_KEEP)

LOG_LEVELS = ["EMERG", "ALERT", "CRIT", "ERROR", "WARN", "NOTICE", "INFO", "DEBUG"]
FIX_MODES = ["none", "SPP", "DGNSS", "float", "fixed", "DR", "SBAS"]
CRITICAL = {"NO_STREAM", "NO_OBS", "EMPTY_OBS", "NO_EPH"}

_TTY = sys.stderr.isatty()


def loud(text, color="31"):
    line = f"[{datetime.now():%H:%M:%S}] {text}"
    if _TTY:
        line = f"\033[1;{color}m{line}\033[0m"
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def name_of(t):
    return NAMES.get(t, f"0x{t:04x}")


# ---- live state + checks ----------------------------------------------------
class Monitor:
    def __init__(self, args):
        self.a = args
        self.lock = threading.Lock()
        self.start = time.time()
        self.last_msg = None
        self.seen = Counter()
        self.written = Counter()
        self.bytes = 0
        # observations
        self.first_obs = None
        self.pending = 0
        self.epochs = 0
        self.good_epochs = 0
        self.empty_epochs = 0
        self.last_epoch_sig = None
        self.min_sig = None
        self.max_sig = 0
        self.last_good = None
        # ephemerides
        self.first_eph = None
        self.eph = Counter()
        # solution / receiver health
        self.fix = None
        self.no_fix_since = None
        self.hb = None
        # problems
        self.active = {}
        self.ever = {}
        self.events = []

    def problem(self, key, on, text=""):
        if on and key not in self.active:
            self.active[key] = text
            self.ever[key] = text
            tag = "PROBLEM" if key in CRITICAL else "warning"
            msg = f"{tag} [{key}] {text}"
            self.events.append(f"{datetime.now():%H:%M:%S} {msg}")
            loud(msg, "31" if key in CRITICAL else "33")
        elif not on and key in self.active:
            del self.active[key]
            msg = f"cleared [{key}]"
            self.events.append(f"{datetime.now():%H:%M:%S} {msg}")
            loud(msg, "32")

    def on_message(self, msg):
        now = time.time()
        t = msg.msg_type
        with self.lock:
            self.last_msg = now
            self.seen[t] += 1
            if t == MSG_OBS:
                self._on_obs(msg, now)
            elif t in EPH_TYPES:
                self.first_eph = self.first_eph or now
                self.eph[EPH_TYPES[t]] += 1
            elif t == MSG_POS_LLH:
                mode = msg.flags & 0x7
                self.fix = (mode, msg.n_sats)
                if mode == 0:
                    self.no_fix_since = self.no_fix_since or now
                else:
                    self.no_fix_since = None
            elif t == MSG_HEARTBEAT:
                self.hb = msg.flags
        if t == MSG_LOG and msg.level <= 4:
            text = msg.text
            if isinstance(text, (bytes, bytearray)):
                text = text.decode(errors="replace")
            loud(f"receiver log {LOG_LEVELS[msg.level]}: {text.strip()}", "35")

    def _on_obs(self, msg, now):
        self.first_obs = self.first_obs or now
        h = msg.header
        total, idx = h.n_obs >> 4, h.n_obs & 0xF
        if idx == 0:
            self.pending = 0
        self.pending += len(msg.obs)
        if idx < total - 1:
            return  # epoch continues in the next frame
        n = self.pending
        self.epochs += 1
        self.last_epoch_sig = n
        self.max_sig = max(self.max_sig, n)
        self.min_sig = n if self.min_sig is None else min(self.min_sig, n)
        if h.t.wn == 0 or n == 0:
            self.empty_epochs += 1
        if h.t.wn != 0 and n >= self.a.min_signals:
            self.good_epochs += 1
            self.last_good = now

    def evaluate(self):
        a = self.a
        now = time.time()
        el = now - self.start
        with self.lock:
            quiet = now - (self.last_msg or self.start)
            self.problem(
                "NO_STREAM",
                quiet > a.stall,
                f"no SBP messages for {quiet:.0f} s "
                "(receiver down, link lost, or port taken by another client)",
            )
            self.problem(
                "NO_OBS",
                self.first_obs is None and el > a.obs_timeout,
                f"no MSG_OBS in {a.obs_timeout:.0f} s: "
                "obs (74) is not being output on this port",
            )
            if self.first_obs is not None:
                since = now - (self.last_good or self.first_obs)
                self.problem(
                    "EMPTY_OBS",
                    since > a.obs_timeout,
                    f"MSG_OBS arriving but no epoch with valid GPS week and "
                    f">= {a.min_signals} signals for {since:.0f} s "
                    f"(last epoch: {self.last_epoch_sig} signals): "
                    "sky view or antenna problem",
                )
            self.problem(
                "NO_EPH",
                self.first_eph is None and el > a.eph_timeout,
                f"no ephemeris in {a.eph_timeout:.0f} s: ephemeris types not "
                "output on this port, or satellites not tracked long enough "
                "to decode",
            )
            nf = now - self.no_fix_since if self.no_fix_since else 0.0
            self.problem(
                "NO_FIX",
                nf > a.fix_timeout,
                f"POS_LLH reports no solution for {nf:.0f} s",
            )
            if self.hb is not None:
                f = self.hb
                errs = [
                    n
                    for b, n in ((0, "system"), (1, "IO"), (2, "SwiftNAP"))
                    if (f >> b) & 1
                ]
                self.problem(
                    "RX_ERROR", bool(errs), f"heartbeat reports {'/'.join(errs)} error"
                )
                self.problem(
                    "ANT_SHORT",
                    bool((f >> 30) & 1),
                    "heartbeat reports external antenna short circuit",
                )
                self.problem(
                    "NO_ANT",
                    not ((f >> 31) & 1),
                    "heartbeat reports no external antenna connected",
                )

    def status(self):
        with self.lock:
            el = time.time() - self.start
            if self.fix:
                m, ns = self.fix
                fix = f"{FIX_MODES[m] if m < len(FIX_MODES) else m}/{ns}sv"
            else:
                fix = "n/a"
            eph = ",".join(f"{c}={n}" for c, n in sorted(self.eph.items())) or "none"
            return (
                f"[{el:6.1f}s] {self.bytes} B  epochs {self.good_epochs}/{self.epochs} good"
                f"  last {self.last_epoch_sig} sig  fix {fix}  eph {eph}"
            )

    def summary(self, out_path, host, port, deleted):
        el = time.time() - self.start
        L = [
            "---- summary ----------------------------------------------------",
            f"receiver    : {host}:{port}",
            f"output file : {out_path}{'  (deleted: nothing written)' if deleted else ''}",
            f"duration    : {el:.1f} s",
            f"written     : {sum(self.written.values())} msgs, {self.bytes} bytes",
            f"obs epochs  : {self.epochs} total, {self.good_epochs} good, "
            f"{self.empty_epochs} empty/untimed",
            f"signals/ep  : min {self.min_sig}  max {self.max_sig}",
            "ephemerides : "
            + (", ".join(f"{c}={n}" for c, n in sorted(self.eph.items())) or "none"),
            "written by type:",
        ]
        for t, c in sorted(self.written.items()):
            L.append(f"   {name_of(t):18s} {c}")
        L.append("seen on stream (all types):")
        for t, c in sorted(self.seen.items()):
            L.append(f"   {name_of(t):18s} {c}")
        L.append("events:" if self.events else "events: none")
        L.extend(f"   {e}" for e in self.events)
        if self.active:
            L.append("still active at end: " + ", ".join(sorted(self.active)))

        if not self.seen:
            verdict = "FAIL: connected but received no messages"
        elif self.bytes == 0:
            verdict = "FAIL: nothing written (no obs/ephemeris on this port)"
        elif self.good_epochs == 0:
            verdict = (
                "FAIL: no valid observation epochs; sbp2rinex will produce nothing"
            )
        elif not self.eph:
            verdict = (
                "PARTIAL: observations OK, no ephemerides; sbp2rinex gives .obs "
                "only, get nav from IGS BRDC"
            )
        else:
            verdict = "OK: observations and ephemerides present"
        L.append(f"VERDICT     : {verdict}")
        if self.good_epochs:
            L.append(f"next        : sbp2rinex {out_path}")
        L.append("-----------------------------------------------------------------")
        return "\n".join(L), verdict.startswith("OK")


def watchdog(mon, args, stop):
    next_status = time.time() + args.status_every
    while not stop.wait(1.0):
        mon.evaluate()
        if args.verbose and time.time() >= next_status:
            print(mon.status(), flush=True)
            next_status += args.status_every


def make_filename(role, folder):
    fname = f"{role}_raw_{datetime.now():%Y%m%d_%H%M%S}.sbp"
    if folder:
        folder = os.path.expanduser(folder)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, fname)
    return fname


def main():
    ap = argparse.ArgumentParser(description="Log raw SBP obs/ephemeris, fail loudly.")
    ap.add_argument("role", choices=["base", "rover"])
    ap.add_argument("--folder", default="")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument(
        "--all", action="store_true", help="write every message, not just obs/ephemeris"
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--status-every", type=float, default=5.0)
    ap.add_argument(
        "--strict", action="store_true", help="exit 1 if any critical problem occurred"
    )
    ap.add_argument("--stall", type=float, default=5.0)
    ap.add_argument("--obs-timeout", type=float, default=15.0)
    ap.add_argument("--eph-timeout", type=float, default=180.0)
    ap.add_argument("--fix-timeout", type=float, default=60.0)
    ap.add_argument(
        "--min-signals",
        type=int,
        default=8,
        help="signals (not satellites) per epoch to count as good",
    )
    args = ap.parse_args()

    host = args.host or ROLE_DEFAULTS[args.role]["host"]
    port = args.port or ROLE_DEFAULTS[args.role]["port"]

    if not EPH_TYPES:
        loud(
            "libsbp exposes no SBP_MSG_EPHEMERIS_* constants; ephemerides will "
            "not be written. Upgrade libsbp or use --all."
        )
    wanted = None if args.all else ({MSG_OBS} | set(EPH_TYPES) | set(EXTRA_KEEP))

    # reachability probe, before any file is created
    try:
        socket.create_connection((host, port), timeout=3).close()
        time.sleep(0.5)
    except ConnectionRefusedError:
        loud(
            f"connection refused by {host}:{port}: TCP server disabled on that "
            "port, or another client (ROS driver, Swift Console) holds it"
        )
        sys.exit(2)
    except OSError as e:
        loud(f"cannot reach {host}:{port}: {e} (wrong IP, cable, or network)")
        sys.exit(2)

    out_path = make_filename(args.role, args.folder)
    print(f"role     : {args.role}")
    print(f"receiver : {host}:{port}")
    print(f"output   : {out_path}")
    print(
        "filter   : "
        + ("ALL" if wanted is None else ", ".join(sorted(name_of(t) for t in wanted)))
    )
    print("Ctrl-C to stop\n", flush=True)

    mon = Monitor(args)
    stop = threading.Event()
    th = threading.Thread(target=watchdog, args=(mon, args, stop), daemon=True)
    logf = None
    try:
        with TCPDriver(host, port) as driver:
            with Handler(Framer(driver.read, driver.write)) as source:
                logf = open(out_path, "wb")
                mon.start = time.time()
                th.start()
                for msg, _meta in source:
                    mon.on_message(msg)
                    t = msg.msg_type
                    if wanted is None or t in wanted:
                        frame = msg.pack()
                        logf.write(frame)
                        with mon.lock:
                            mon.bytes += len(frame)
                            mon.written[t] += 1
    except KeyboardInterrupt:
        pass
    except Exception as e:
        loud(f"stream error: {e}")
    finally:
        stop.set()
        if logf:
            logf.close()

    deleted = False
    if logf and mon.bytes == 0 and os.path.exists(out_path):
        os.remove(out_path)
        deleted = True

    text, ok = mon.summary(out_path, host, port, deleted)
    print("\n" + text)
    if logf:
        with open(out_path + ".summary.txt", "w") as f:
            f.write(text + "\n")

    if not mon.seen:
        sys.exit(3)
    if args.strict and (set(mon.ever) & CRITICAL):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
