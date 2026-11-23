# MIT License — see LICENSE in the project root.

"""
diagnostics.py — hang diagnostics for loading an image
=======================================================
While an image is loading (wait cursor showing), a small recorder writes a
log file that is **deleted when the load completes normally** and **kept when
it does not** (error, hang, crash, killed with Ctrl-C).  If euniverse hangs
with a spinning busy cursor, the leftover file shows where and why.

Location:  <cache dir>/euniverse/diagnostics/load_<date>_<time>_<pid>_<n>.log
           (Linux: ~/.cache/euniverse/diagnostics; the path is also printed
           to the console if a load takes unusually long.)

What it records
---------------
* Header: versions, Qt platform plugin, session type, file name and size.
* Stage marks with a time stamp, the thread, and process memory
  (RSS / available RAM / swap), written by the loader thread and by the
  GUI-thread slot that builds the display.
* Python stack traces of ALL threads every DUMP_EVERY seconds (faulthandler
  writes them from its own C thread, so it works even if the GUI thread is
  stuck in native code and holding the GIL), and on fatal signals (SIGSEGV,
  SIGABRT, ...).  On Linux/macOS ``kill -USR1 <pid>`` dumps them on demand.
* A heartbeat from the GUI thread.  When it stops for more than STALL_AFTER
  seconds, the log gets "GUI thread unresponsive", a per-thread table (CPU
  use, state, kernel wait channel: busy loop vs. deadlock vs. swapping), and,
  after GDB_AFTER seconds, native stack traces of all threads obtained with
  gdb if it is installed (Linux only; EUNIVERSE_DIAG_NO_GDB=1 to skip).
* Resize / move / show / hide / close / state-change events of windows, and
  counts of Resize events per widget class.

Environment variables
---------------------
  EUNIVERSE_NO_DIAG=1     switch the recorder off
  EUNIVERSE_KEEP_DIAG=1   keep the log even after a successful load
"""

import faulthandler
import glob
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

from PyQt5.QtCore import PYQT_VERSION_STR, QEvent, QObject, QT_VERSION_STR, QTimer
from PyQt5.QtWidgets import QApplication

DUMP_EVERY    = 10.0     # s between all-thread Python stack dumps while loading
STALL_AFTER   = 2.0      # s without a GUI heartbeat before it is called unresponsive
GDB_AFTER     = 20.0     # s of unresponsiveness before native stacks are collected
LONG_LOAD     = 20.0     # s after which the log location is printed to the console
TICK          = 1.0      # watchdog period (s)
BEAT_MS       = 200      # GUI heartbeat period (ms)
MAX_EVENT_LINES = 300
KEEP_DAYS     = 30       # old logs are removed after this many days
GDB_TIMEOUT   = 120

_active = None           # the current session, or None
_seq = 0                 # makes log file names unique within a process


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _enabled() -> bool:
    return not os.environ.get("EUNIVERSE_NO_DIAG")


def _diag_dir() -> str:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "euniverse", "diagnostics")


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def _memory() -> str:
    """'rss 812 MB, avail 3120 MB, swap used 150 MB' (Linux; empty elsewhere)."""
    try:
        rss = swap = None
        for line in _read("/proc/self/status").splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) // 1024
        mi = {l.split(":")[0]: int(l.split()[1]) for l in _read("/proc/meminfo").splitlines() if ":" in l and len(l.split()) > 1}
        if rss is None or "MemAvailable" not in mi:
            return ""
        swap = (mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) // 1024
        return f"rss {rss} MB, avail {mi['MemAvailable'] // 1024} MB, swap used {swap} MB"
    except Exception:
        return ""


def _file_info(path: str) -> str:
    try:
        return f"{os.path.getsize(path) / 1e6:.1f} MB"
    except Exception:
        return "size unknown"


class _EventLog(QObject):
    """Application event filter: records window-level events that could provoke a hang."""

    NAMES = {
        QEvent.Resize: "Resize", QEvent.Move: "Move", QEvent.Show: "Show", QEvent.Hide: "Hide",
        QEvent.Close: "Close", QEvent.WindowStateChange: "WindowStateChange",
        QEvent.WindowActivate: "WindowActivate", QEvent.WindowDeactivate: "WindowDeactivate",
    }

    def __init__(self, session):
        super().__init__()
        self._s = session
        self._lines = 0
        self.resizes = {}

    def eventFilter(self, obj, ev):
        try:
            name = self.NAMES.get(ev.type())
            if name:
                cls = type(obj).__name__
                if name == "Resize":
                    self.resizes[cls] = self.resizes.get(cls, 0) + 1
                if obj.isWidgetType() and obj.isWindow() and self._lines < MAX_EVENT_LINES:
                    extra = ""
                    if name == "Resize":
                        extra = f" {ev.oldSize().width()}x{ev.oldSize().height()} -> {ev.size().width()}x{ev.size().height()}"
                    self._lines += 1
                    self._s.mark(f"event: {name} on window {cls}{extra}", memory=False)
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# One recording session = one load
# ---------------------------------------------------------------------------

class _Session:
    def __init__(self, path: str):
        self.t0 = time.monotonic()
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.beat = time.monotonic()
        self.stalled_since = None
        self.gdb_done = False
        self.announced = False
        self.last_table = {}            # tid -> (cpu seconds, wall time) for CPU-use deltas
        self._fh_enabled_before = faulthandler.is_enabled()
        self._sig = False

        d = _diag_dir()
        os.makedirs(d, exist_ok=True)
        self._prune(d)
        global _seq
        _seq += 1                                   # two loads within one second must not share a file
        self.path = os.path.join(d, f"load_{datetime.now():%Y%m%d_%H%M%S}_{self.pid}_{_seq}.log")
        self.fh = open(self.path, "w", buffering=1)

        self._header(path)
        faulthandler.enable(file=self.fh, all_threads=True)
        faulthandler.dump_traceback_later(DUMP_EVERY, repeat=True, file=self.fh)
        if hasattr(signal, "SIGUSR1"):
            try:
                faulthandler.register(signal.SIGUSR1, file=self.fh, all_threads=True)
                self._sig = True
            except Exception:
                pass

        app = QApplication.instance()
        self.events = _EventLog(self)
        if app is not None:
            app.installEventFilter(self.events)
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_beat)
        self.timer.start(BEAT_MS)
        self.watchdog = threading.Thread(target=self._watch, name="euniverse-diag", daemon=True)
        self.watchdog.start()

    # -- file output --------------------------------------------------------
    def _write(self, text: str):
        with self.lock:
            if self.fh and not self.fh.closed:
                self.fh.write(text)
                self.fh.flush()

    def mark(self, text: str, memory: bool = True):
        t = time.monotonic() - self.t0
        m = _memory() if memory else ""
        mem = f"   | {m}" if m else ""
        self._write(f"[{t:8.3f}s] [{threading.current_thread().name}] {text}{mem}\n")

    def _header(self, path: str):
        app = QApplication.instance()
        lines = [
            f"euniverse load diagnostics — {datetime.now().isoformat(timespec='seconds')}",
            f"pid {self.pid}; argv {sys.argv}",
            f"file: {path} ({_file_info(path)})",
            f"python {platform.python_version()}; PyQt {PYQT_VERSION_STR}; Qt {QT_VERSION_STR}; {platform.platform()}",
            f"Qt platform: {app.platformName() if app else '?'}; QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM')}; "
            f"XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE')}; WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')}",
            f"DISPLAY={os.environ.get('DISPLAY')}; cpus {os.cpu_count()}",
            "Python stack dumps of all threads appear as 'Thread 0x… (most recent call first)'.",
            "A line with 'GUI thread unresponsive' means the main thread stopped processing events.",
            "-" * 78,
        ]
        self._write("\n".join(lines) + "\n")

    # -- GUI heartbeat (runs in the GUI thread) -----------------------------
    def _on_beat(self):
        self.beat = time.monotonic()

    # -- watchdog thread ------------------------------------------------------
    def _watch(self):
        while not self.done.wait(TICK):
            try:
                now = time.monotonic()
                elapsed = now - self.t0
                if elapsed > LONG_LOAD and not self.announced:
                    self.announced = True
                    extra = f"  (kill -USR1 {self.pid} dumps the stacks)" if hasattr(signal, "SIGUSR1") else ""
                    print(f"euniverse: still loading after {LONG_LOAD:.0f} s — diagnostics: {self.path}{extra}",
                          file=sys.stderr, flush=True)
                gap = now - self.beat
                if gap > STALL_AFTER:
                    first = self.stalled_since is None
                    if first:
                        self.stalled_since = now - gap
                    self.mark(f"GUI thread unresponsive for {gap:.1f} s", memory=True)
                    if first or int(gap) % 5 == 0:
                        self._thread_table()
                    if first:
                        self._dump_python()
                    if gap > GDB_AFTER and not self.gdb_done:
                        self.gdb_done = True
                        self._native_stacks()
                elif self.stalled_since is not None:
                    self.mark(f"GUI thread responsive again after {now - self.stalled_since:.1f} s")
                    self.stalled_since = None
                    self.gdb_done = False
                if self.events.resizes and int(elapsed) % 10 == 0:
                    self.mark("Resize events so far: " + ", ".join(f"{k} x{v}" for k, v in sorted(self.events.resizes.items())),
                              memory=False)
            except Exception as e:                       # never let the recorder kill anything
                self._write(f"(watchdog error: {e!r})\n")

    def _dump_python(self):
        with self.lock:
            if self.fh and not self.fh.closed:
                self.fh.write("--- Python stacks of all threads (watchdog) ---\n")
                faulthandler.dump_traceback(file=self.fh, all_threads=True)
                self.fh.flush()

    def _thread_table(self):
        """Per-thread state and CPU use from /proc (Linux)."""
        tdir = "/proc/self/task"
        if not os.path.isdir(tdir):
            return
        hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        now = time.monotonic()
        rows = ["--- threads (tid, name, state, CPU % since last table, kernel wait channel) ---"]
        for tid in sorted(os.listdir(tdir), key=int):
            try:
                stat = _read(f"{tdir}/{tid}/stat")
                rest = stat.rsplit(")", 1)[1].split()
                state, cpu = rest[0], (int(rest[11]) + int(rest[12])) / hz
                name = _read(f"{tdir}/{tid}/comm").strip()
                wchan = _read(f"{tdir}/{tid}/wchan").strip() or "-"
                prev = self.last_table.get(tid)
                pct = f"{100 * (cpu - prev[0]) / max(now - prev[1], 1e-3):5.0f}" if prev else "    ?"
                self.last_table[tid] = (cpu, now)
                rows.append(f"    {tid:>7} {name:<16} {state}  {pct}%  {wchan}")
            except Exception:
                continue
        self._write("\n".join(rows) + "\n")

    def _native_stacks(self):
        """Native stack traces of all threads via gdb (Linux only)."""
        if sys.platform != "linux" or os.environ.get("EUNIVERSE_DIAG_NO_GDB"):
            return
        gdb = shutil.which("gdb")
        if not gdb:
            self.mark("native stack traces skipped: gdb is not installed (e.g. 'sudo apt install gdb')", memory=False)
            return
        self.mark("collecting native stack traces with gdb (the process is paused while gdb runs) …", memory=False)
        try:                                           # allow our own child (gdb) to attach under Yama ptrace_scope=1
            import ctypes
            ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1 & 0xFFFFFFFFFFFFFFFF), 0, 0, 0)
        except Exception:
            pass
        try:
            out = subprocess.run(
                [gdb, "-p", str(self.pid), "-batch", "-nx",
                 "-iex", "set debuginfod enabled off",
                 "-ex", "set pagination off", "-ex", "thread apply all bt 40"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=GDB_TIMEOUT)
            text = out.stdout
        except Exception as e:
            text = f"gdb failed: {e!r}"
        if len(text) > 300_000:
            text = text[:300_000] + "\n… (truncated)"
        self._write("--- native stacks (gdb) ---\n" + text + "\n--- end of native stacks ---\n")

    # -- shutdown -------------------------------------------------------------------
    def end(self, ok: bool, note: str = "") -> bool:
        """Stop recording.  Deletes the log if the load succeeded.  Returns True if the log was kept."""
        self.done.set()
        try:
            faulthandler.cancel_dump_traceback_later()
            if self._sig:
                faulthandler.unregister(signal.SIGUSR1)
            if self._fh_enabled_before:
                faulthandler.enable(file=sys.stderr, all_threads=True)
            else:
                faulthandler.disable()
        except Exception:
            pass
        try:
            self.timer.stop()
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self.events)
        except Exception:
            pass
        if self.stalled_since is not None:
            self.mark(f"the GUI thread had been unresponsive for {time.monotonic() - self.stalled_since:.1f} s when the load ended")
        self.mark(f"load finished: {'OK' if ok else 'NOT OK'}{(' — ' + note) if note else ''}")
        with self.lock:
            try:
                self.fh.close()
            except Exception:
                pass
        keep = (not ok) or bool(os.environ.get("EUNIVERSE_KEEP_DIAG"))
        if not keep:
            try:
                os.remove(self.path)
            except Exception:
                pass
        return keep

    @staticmethod
    def _prune(d: str):
        cutoff = time.time() - KEEP_DAYS * 86400
        for f in glob.glob(os.path.join(d, "load_*.log")):
            try:
                if os.path.getmtime(f) < cutoff:
                    os.remove(f)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public API (all calls are safe no-ops if the recorder is off or has failed)
# ---------------------------------------------------------------------------

def begin(path: str):
    """Start recording a load of *path* (GUI thread).  Ends an unfinished earlier session first."""
    global _active
    if not _enabled():
        return None
    try:
        if _active is not None:
            _active.end(False, "superseded by a new load")
        _active = _Session(path)
        return _active.path
    except Exception as e:
        _active = None
        print(f"euniverse: load diagnostics unavailable: {e}", file=sys.stderr)
        return None


def mark(text: str):
    """Add a stage line (any thread)."""
    s = _active
    if s is not None:
        try:
            s.mark(text)
        except Exception:
            pass


def end(ok: bool, note: str = ""):
    """Finish: the log is deleted if *ok*, kept otherwise.  Returns the kept path or None."""
    global _active
    s, _active = _active, None
    if s is None:
        return None
    try:
        return s.path if s.end(ok, note) else None
    except Exception:
        return None


def active_path():
    return _active.path if _active is not None else None
