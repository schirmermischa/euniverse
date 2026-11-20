#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# euniverse.py - A program to display MER colour images created with eummy

# MIT License

# Copyright (c) [2026] [Mischa Schirmer]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
simbad_lookup.py — NED + SIMBAD overlay for the displayed field of view
=========================================================================
Implements the dock's "look-up" toggle:

  ON   Query NED and SIMBAD together (TAP/ADQL cone searches circumscribing
       the image, in parallel background threads) and keep the objects that
       fall inside the image.  NED objects get a green circle, SIMBAD objects
       a yellow square.  Hovering a marker shows the object's name, type
       (and NED redshift) in a tool-tip; left-clicking opens a dialog with
       the record of that service (plus a link to its full web page).  A
       table dialog with one tab per service opens when the first answer
       arrives.  One service failing or being slow does not affect the other.
  OFF  Remove all markers and discard the retrieved catalogs.

Pieces
------
SimbadQueryWorker   Does the network I/O (no Qt GUI calls).  Runs on a plain
                    daemon thread, deliberately NOT a QThread: a QThread that
                    is still blocked in a slow HTTP request when the app
                    exits aborts the process ("QThread: Destroyed while
                    thread is still running"), a daemon thread does not.
                    Results come back to the GUI thread via queued signals.
SimbadManager       Owns the overlay items and the retrieved catalogs; talks
                    to ImageViewer (WCS, scene) and ControlDock (button).
SimbadDetailsDialog Non-modal record viewer.

Only the standard library is used for HTTP (urllib) so no new dependency
is needed.
"""

import csv
import gzip
import hashlib
import html
import io
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

import numpy as np
from astropy.coordinates import SkyCoord

from PyQt5.QtCore import QEvent, QObject, QPointF, Qt, QRectF, QSettings, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QDesktopServices, QFont, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QGraphicsEllipseItem, QGraphicsRectItem, QTabWidget,
    QButtonGroup, QHBoxLayout, QHeaderView, QLabel, QMenu, QPlainTextEdit, QPushButton, QRadioButton, QToolTip,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Synchronous TAP endpoints, tried in this order.  The first is the CDS
# master; the others are SIMBAD's official mirrors, used when the master
# answers with 403/404/429/5xx or cannot be reached.
TAP_SYNC_URLS = [
    "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
    "https://simbad.harvard.edu/simbad/sim-tap/sync",
    "https://simbad.idia.ac.za/simbad/sim-tap/sync",
]
SIMBAD_ID_URL = "https://simbad.cds.unistra.fr/simbad/sim-id?Ident="

HTTP_TIMEOUT = 90          # seconds, per socket operation (last mirror)
FAILOVER_TIMEOUT = 30      # seconds, per socket operation on mirrors that still have a fallback
DEAD_MIRROR_SECS = 600     # a mirror that just failed is skipped for this long
MAX_SOURCES  = 20000       # safety cap on rows requested from SIMBAD

# Catalogue services.  NED (extragalactic: redshifts, galaxies, AGN) is the
# default; SIMBAD (all objects, strongest for stars) is the alternative.
SOURCES        = ("NED", "SIMBAD")      # both are always queried

# NED: field query through its TAP service (single table "objdir"); aliases and
# reference codes of one object through the ObjectLookup JSON service.
NED_TAP_URLS   = ["https://ned.ipac.caltech.edu/tap/sync"]
NED_LOOKUP_URL = "https://ned.ipac.caltech.edu/srs/ObjectLookup"
NED_OBJECT_URL = "https://ned.ipac.caltech.edu/byname?objname="
NED_TIMEOUT    = 180       # seconds per socket operation (NED's sync proxy gives up after 300 s)
# objdir columns requested.  Only prefname/ra/dec are essential; if NED rejects
# the query (HTTP 400) the schema is consulted and the request repeated with the
# columns that exist.
NED_COLUMNS    = ["prefname", "ra", "dec", "z", "pretype", "zflag", "n_crosref"]
_NED_REQUIRED  = ("prefname", "ra", "dec")
NED_FIELDS = [
    ("main_id",   "Preferred name"),
    ("otype",     "Object type"),
    ("rvz_redshift", "Redshift"),
    ("zflag",     "Redshift quality flag"),
    ("n_crosref", "Cross-references (n_crosref)"),
]
# NED object-type codes (written from memory; unknown codes are shown as they are).
NED_TYPES = {
    "G": "Galaxy", "GGroup": "Group of galaxies", "GPair": "Galaxy pair", "GTrpl": "Galaxy triplet",
    "GClstr": "Cluster of galaxies", "PofG": "Part of galaxy", "QSO": "Quasi-stellar object",
    "Q_Lens": "Lensed quasar", "RadioS": "Radio source", "IrS": "Infrared source",
    "UvS": "Ultraviolet source", "UvES": "UV-excess source", "XrayS": "X-ray source",
    "GammaS": "Gamma-ray source", "VisS": "Visual source", "AbLS": "Absorption-line system",
    "EmLS": "Emission-line source", "SNR": "Supernova remnant", "SN": "Supernova",
    "HII": "HII region", "PN": "Planetary nebula", "MCld": "Molecular cloud",
    "*": "Star", "**": "Double star", "!*": "Galactic star", "Other": "Other",
}

# Results of a field query are cached on disk (per field centre and size) so
# toggling the look-up off and on, or reloading a tile, is instant.  Set the
# environment variable EUNIVERSE_SIMBAD_NOCACHE=1 to bypass the cache, or
# delete ~/.cache/euniverse/simbad to force fresh data.
CACHE_MAX_AGE_DAYS = 7
SCHEMA_CACHE_DAYS  = 30     # column list / object-type definitions change rarely

# Overlay appearance
CIRCLE_COLOR     = QColor(0, 255, 0)
SQUARE_COLOR     = QColor(255, 215, 0)     # SIMBAD: yellow squares (NED: green circles)
SQUARE_SCALE     = 1.5     # half-side of a SIMBAD square = radius * SQUARE_SCALE, so that a
                           # circle for the same object sits inside its square
CIRCLE_WIDTH     = 1.5     # cosmetic: constant on screen at every zoom level
CIRCLE_WIDTH_SEL = 3.0
DEFAULT_RADIUS_ARCSEC = 1.5   # circle size for objects without a known extent
MIN_RADIUS_PX    = 8.0
MAX_RADIUS_PX    = 600.0

# (SIMBAD basic-table column, label shown in the details dialog).
# All of these are requested in a single query.  Only if SIMBAD rejects the
# query (HTTP 400, e.g. a column this release lacks) is the service's column
# list consulted -- a slow query that is then cached for 30 days -- and the
# request repeated with the columns that exist.
FIELDS = [
    ("main_id",        "Main identifier"),
    ("otype",          "Object type"),
    ("sp_type",        "Spectral type"),
    ("sp_bibcode",     "Spectral type reference"),
    ("rvz_redshift",   "Redshift"),
    ("rvz_radvel",     "Radial velocity (km/s)"),
    ("rvz_type",       "Redshift / velocity type"),
    ("rvz_bibcode",    "Redshift reference"),
    ("plx_value",      "Parallax (mas)"),
    ("plx_bibcode",    "Parallax reference"),
    ("pmra",           "Proper motion in RA (mas/yr)"),
    ("pmdec",          "Proper motion in Dec (mas/yr)"),
    ("pm_bibcode",     "Proper motion reference"),
    ("galdim_majaxis", "Galaxy major axis (arcmin)"),
    ("galdim_minaxis", "Galaxy minor axis (arcmin)"),
    ("galdim_angle",   "Galaxy position angle (deg)"),
    ("galdim_bibcode", "Galaxy size reference"),
    ("coo_err_maj",    "Position error, major axis (mas)"),
    ("coo_err_min",    "Position error, minor axis (mas)"),
    ("coo_err_angle",  "Position error angle (deg)"),
    ("coo_bibcode",    "Position reference"),
    ("nbref",          "Number of references"),
]
_REQUIRED = ("main_id", "ra", "dec")

# Which reference the table's "Reference" column shows when an object has
# several: the first non-empty one in this order (column, what it refers to).
REF_PRIORITY = [
    ("rvz_bibcode",    "redshift"),
    ("sp_bibcode",     "spectral type"),
    ("plx_bibcode",    "parallax"),
    ("pm_bibcode",     "proper motion"),
    ("galdim_bibcode", "galaxy size"),
    ("coo_bibcode",    "position"),
]


def ads_url(bibcode: str) -> str:
    """Link to the NASA/SAO ADS abstract page of a bibcode."""
    return "https://ui.adsabs.harvard.edu/abs/" + urllib.parse.quote(bibcode.strip(), safe="") + "/abstract"


def pick_reference(src: dict):
    """(bibcode, what it refers to) of the most relevant reference, or ("", "")."""
    for column, what in REF_PRIORITY:
        value = (src.get(column) or "").strip()
        if value:
            return value, what
    return "", ""


# ---------------------------------------------------------------------------
# TAP helpers (pure Python, no Qt)
# ---------------------------------------------------------------------------

class SimbadError(Exception):
    """Any failure talking to SIMBAD; the message is shown to the user."""


def _tap_error_text(body: str) -> str:
    """
    Short human-readable message from an error body: the QUERY_STATUS text
    of a TAP VOTable, the <title> of an HTML error page (never the page's
    CSS/markup), or the plain text, truncated.
    """
    m = re.search(r'<INFO[^>]*name="QUERY_STATUS"[^>]*>(.*?)</INFO>',
                  body, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    elif re.search(r"<\s*(!doctype\s+html|html)\b", body[:500], re.IGNORECASE):
        t = re.search(r"<title[^>]*>(.*?)</title>", body, re.DOTALL | re.IGNORECASE)
        text = t.group(1) if t else "HTML error page"
    else:
        text = re.sub(r"<[^>]+>", " ", body)
    return " ".join(html.unescape(text).split())[:200]


def _user_agent() -> str:
    try:
        from importlib.metadata import version
        v = version("euniverse")
    except Exception:
        v = "dev"
    return f"euniverse/{v} (+https://github.com/schirmermischa/euniverse)"


# HTTP statuses that indicate a problem with the *server*, not with our
# query, so the next mirror is worth trying.  (400 = bad ADQL: would fail
# identically everywhere, so it is reported straight away.)
_MIRROR_RETRY_CODES = {403, 404, 408, 429, 500, 502, 503, 504}

_preferred_mirror = 0     # index into TAP_SYNC_URLS of the last one that worked
_mirror_loaded    = False
_mirror_dead      = {}    # index -> time.monotonic() of its last failure


def _load_preferred_mirror():
    """Start with the mirror that worked last time (remembered for 24 h)."""
    global _preferred_mirror, _mirror_loaded
    _mirror_loaded = True
    got = _json_get("mirror.json.gz", 1.0)
    if got:
        for i, u in enumerate(TAP_SYNC_URLS):
            if urllib.parse.urlparse(u).netloc == got[0].get("host"):
                _preferred_mirror = i


def _post(url: str, payload: bytes, timeout: float, progress=None, stats=None) -> str:
    """
    POST *payload* and read the answer in chunks, so that the caller can show
    progress while the (possibly large) CSV streams in.  gzip is requested
    and transparently decoded if the server supports it.

    progress(rows, nbytes) is called a few times per second with the number
    of CSV lines decoded so far and the bytes received on the wire.
    stats (dict) is filled with wait/download seconds, bytes and gzip flag.
    """
    request = urllib.request.Request(url, data=payload, headers={
        "User-Agent": _user_agent(), "Accept-Encoding": "gzip"})
    t0 = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        t1 = time.monotonic()                 # headers received: the server has answered
        headers = getattr(resp, "headers", None)
        gz = bool(headers) and (headers.get("Content-Encoding") or "").lower() == "gzip"
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if gz else None
        chunks, nbytes, nlines, last = [], 0, 0, 0.0
        while True:
            block = resp.read(65536)
            if not block:
                break
            nbytes += len(block)
            data = decoder.decompress(block) if decoder else block
            chunks.append(data)
            nlines += data.count(b"\n")
            now = time.monotonic()
            if progress and now - last >= 0.2:
                progress(nlines, nbytes)
                last = now
        if decoder:
            chunks.append(decoder.flush())
        if progress:
            progress(nlines, nbytes)
    if stats is not None:
        stats.update(wait=t1 - t0, download=time.monotonic() - t1, bytes=nbytes, gzip=gz)
    return b"".join(chunks).decode("utf-8", errors="replace")


def tap_query(adql: str, timeout: float = HTTP_TIMEOUT, progress=None, stats=None,
              urls=None, extra=None, label: str = "SIMBAD") -> list:
    """
    Run a synchronous ADQL query; return the rows as a list of dicts of str.

    Tries the CDS master first and falls back to the mirrors on server-side
    failures.  The mirror that last worked is tried first next time.
    *progress* / *stats*: see _post(); stats additionally gets 'mirror'.
    *urls*: use this list of endpoints (tried in order, no mirror memory)
    instead of the SIMBAD mirrors -- used for NED.  *extra*: additional TAP
    parameters (e.g. MAXREC).  *label*: service name for messages.
    """
    global _preferred_mirror
    params = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}
    params.update(extra or {})
    payload = urllib.parse.urlencode(params).encode("utf-8")

    custom = urls is not None
    endpoints = list(urls) if custom else TAP_SYNC_URLS
    if custom:
        order = list(range(len(endpoints)))
    else:
        if not _mirror_loaded:
            _load_preferred_mirror()
        order = [(_preferred_mirror + i) % len(endpoints) for i in range(len(endpoints))]
        now = time.monotonic()
        alive = [i for i in order if now - _mirror_dead.get(i, -1e9) > DEAD_MIRROR_SECS]
        order = alive + [i for i in order if i not in alive]   # recently failed ones go last
    failures = []
    for pos, idx in enumerate(order):
        url  = endpoints[idx]
        host = urllib.parse.urlparse(url).netloc
        t_try = time.monotonic()
        # A hung mirror must not hold us for the full timeout while others are waiting.
        per_try = timeout if pos == len(order) - 1 else min(timeout, FAILOVER_TIMEOUT)
        try:
            text = _post(url, payload, per_try, progress, stats)
        except urllib.error.HTTPError as e:
            msg = f"HTTP {e.code} ({_tap_error_text(e.read().decode('utf-8', errors='replace'))})"
            if e.code in _MIRROR_RETRY_CODES:
                if not custom:
                    _mirror_dead[idx] = time.monotonic()
                failures.append(f"{host}: {msg} after {time.monotonic() - t_try:.0f} s")
                continue
            raise SimbadError(f"{label} ({host}) returned {msg}")
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            if not custom:
                _mirror_dead[idx] = time.monotonic()
            failures.append(f"{host}: {getattr(e, 'reason', e)} after {time.monotonic() - t_try:.0f} s")
            continue

        # A failed query can also arrive as HTTP 200 with a VOTable/HTML body.
        if text.lstrip().startswith("<"):
            raise SimbadError(f"{label} ({host}) query failed: {_tap_error_text(text)}")
        if not custom:
            if idx != _preferred_mirror:
                _json_put("mirror.json.gz", {"host": host})     # next session starts here
            _preferred_mirror = idx
            _mirror_dead.pop(idx, None)
        if stats is not None:
            stats["mirror"] = host
            stats["failovers"] = failures
        return list(csv.DictReader(io.StringIO(text)))

    raise SimbadError(f"Could not get an answer from any {label} server — "
                      + "; ".join(failures))


_basic_columns_cache = None   # set of column names of SIMBAD's `basic` table
_otype_cache         = None   # {otype or label: (label, description)}


def _basic_columns(force: bool = False):
    """
    Column names of SIMBAD's `basic` table, or None if unknown.

    force=False: from memory or the 30-day disk cache; never touches the
    network (returns None if not cached).  force=True: ask the service --
    this query can take a minute -- and cache the answer.
    """
    global _basic_columns_cache
    if not force:
        if _basic_columns_cache:
            return _basic_columns_cache
        got = _json_get("schema_basic.json.gz", SCHEMA_CACHE_DAYS)
        if got and all(c in got[0] for c in _REQUIRED):
            _basic_columns_cache = set(got[0])
            return _basic_columns_cache
        return None
    try:
        rows = tap_query("SELECT column_name FROM TAP_SCHEMA.columns "
                         "WHERE table_name = 'basic'")
        cols = {r["column_name"] for r in rows if r.get("column_name")}
        if all(c in cols for c in _REQUIRED):
            _basic_columns_cache = cols
            _json_put("schema_basic.json.gz", sorted(cols))
            return cols
    except SimbadError:
        pass
    return None


def _otype_definitions() -> dict:
    """Object-type code -> (label, description); empty if unavailable."""
    global _otype_cache
    if _otype_cache is None:
        got = _json_get("otypedef.json.gz", SCHEMA_CACHE_DAYS)
        if got:
            _otype_cache = {k: tuple(v) for k, v in got[0].items()}
        else:
            try:
                rows = tap_query("SELECT otype, label, description FROM otypedef")
                table = {}
                for r in rows:
                    entry = (r.get("label", ""), r.get("description", ""))
                    # Key by both fields: depending on the SIMBAD release,
                    # basic.otype holds either the short code or the label.
                    for key in (r.get("otype"), r.get("label")):
                        if key:
                            table.setdefault(key, entry)
                _otype_cache = table
                _json_put("otypedef.json.gz", table)
            except SimbadError:
                return {}
    return _otype_cache


def _to_float(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# On-disk cache of field queries
# ---------------------------------------------------------------------------

def _cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "euniverse", "simbad")


def _json_get(name: str, max_age_days: float):
    """(object, age_in_days) from a fresh cache file, else None."""
    if os.environ.get("EUNIVERSE_SIMBAD_NOCACHE"):
        return None
    path = os.path.join(_cache_dir(), name)
    try:
        if not os.path.exists(path):
            return None
        age = (time.time() - os.path.getmtime(path)) / 86400.0
        if age > max_age_days:
            return None
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f), age
    except Exception:                 # corrupt / unreadable cache: just refetch
        return None


def _json_put(name: str, obj):
    if os.environ.get("EUNIVERSE_SIMBAD_NOCACHE"):
        return
    path = os.path.join(_cache_dir(), name)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)         # atomic: never leaves a half-written file
    except OSError:
        pass                          # read-only home etc.: caching is optional


def _cache_path(ra: float, dec: float, radius: float, source: str = "SIMBAD"):
    """Cache file name of this field, or None if caching is disabled."""
    if os.environ.get("EUNIVERSE_SIMBAD_NOCACHE"):
        return None
    spec = [round(ra, 5), round(dec, 5), round(radius, 5), MAX_SOURCES,
            [c for c, _ in FIELDS] if source == "SIMBAD" else [source] + NED_COLUMNS]
    spec = json.dumps(spec)
    return f"field_{hashlib.sha1(spec.encode()).hexdigest()[:20]}.json.gz"


def _cache_load(name):
    """(rows, info, age_days) of a fresh cached field, else None."""
    got = _json_get(name, CACHE_MAX_AGE_DAYS) if name else None
    if not got:
        return None
    try:
        data, age = got
        info = {"truncated": bool(data["truncated"]),
                "otypes": {k: tuple(v) for k, v in data["otypes"].items()}}
        return data["rows"], info, age
    except Exception:
        return None


def _cache_save(name, rows: list, info: dict):
    if name:
        _json_put(name, {"rows": rows, "truncated": info["truncated"],
                         "otypes": info["otypes"]})


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _hide_tooltip_now():
    """QToolTip.hideText() only starts a ~300 ms hide timer; also hide the
    tip widget itself so it vanishes the instant the cursor leaves."""
    QToolTip.hideText()
    for w in QApplication.topLevelWidgets():
        if w.metaObject().className() == "QTipLabel" and w.isVisible():
            w.hide()


class _TipMixin:
    """Dismisses the tooltip the moment the cursor leaves the item (Qt
    otherwise keeps a scene item's tip on screen until it times out)."""

    def __init__(self, *args):
        super().__init__(*args)
        self.setAcceptHoverEvents(True)

    def hoverLeaveEvent(self, event):
        _hide_tooltip_now()
        super().hoverLeaveEvent(event)


class _SourceCircle(_TipMixin, QGraphicsEllipseItem):
    """NED object: green circle."""


class _SourceSquare(_TipMixin, QGraphicsRectItem):
    """SIMBAD object: yellow square."""


class SimbadQueryWorker(QObject):
    """
    Cone-searches SIMBAD.  Call run() from a background thread.

    Signals (all carry the generation number of the request so the
    manager can discard answers that arrive after the user moved on)
    --------------------------------------------------------------
    finished(int, list, dict)   rows (list of dict), info dict (incl. "timing")
    error(int, str)
    progress(int, str, int, int)   stage text, rows received, bytes received
    """

    finished = pyqtSignal(int, list, dict)
    error    = pyqtSignal(int, str)
    progress = pyqtSignal(int, str, int, int)

    def __init__(self, generation: int, ra: float, dec: float, radius_deg: float):
        super().__init__()
        self.generation = generation
        self.ra         = ra % 360.0
        self.dec        = dec
        self.radius     = radius_deg

    def _stage(self, text: str, rows: int = 0, nbytes: int = 0):
        self.progress.emit(self.generation, text, rows, nbytes)

    @staticmethod
    def _columns(avail) -> list:
        """Columns to request: everything we know of, or only those in *avail*."""
        cols = ["main_id", "ra", "dec"]
        if avail is None or "oid" in avail:
            cols.append("oid")        # not shown; key of the alternative-names lookup
        cols += [c for c, _ in FIELDS if c != "main_id" and (avail is None or c in avail)]
        return cols

    def _query(self, columns: list, stats: dict) -> list:
        adql = (
            f"SELECT TOP {MAX_SOURCES + 1} {', '.join(columns)} FROM basic "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {self.ra:.8f}, {self.dec:.8f}, {self.radius:.8f})) = 1 "
            f"AND ra IS NOT NULL AND dec IS NOT NULL"
        )
        return tap_query(
            adql, stats=stats,
            progress=lambda n, b: self._stage("receiving", max(n - 1, 0), b))   # n-1: CSV header

    def run(self):
        try:
            cache = _cache_path(self.ra, self.dec, self.radius)
            hit = _cache_load(cache)
            if hit:
                rows, info, age = hit
                info["timing"] = {"cached_days": age}
                self.finished.emit(self.generation, rows, info)
                return

            types = {}
            stats, extra = {}, {}
            columns = self._columns(_basic_columns())      # cached schema, never a network call
            self._stage("waiting for SIMBAD to answer")
            try:
                rows = self._query(columns, stats)
            except SimbadError as e:
                if "HTTP 400" not in str(e):
                    raise
                # SIMBAD rejected the query -- most likely a column this
                # release does not have.  Look at the real schema (slow, but
                # cached for 30 days) and retry once with what exists.
                self._stage("checking the SIMBAD schema (first time only)")
                t = time.monotonic()
                fresh = _basic_columns(force=True)
                extra["schema"] = time.monotonic() - t
                retry = self._columns(fresh) if fresh else None
                if not retry or retry == columns:
                    raise
                stats.clear()
                self._stage("waiting for SIMBAD to answer")
                rows = self._query(retry, stats)

            truncated = len(rows) > MAX_SOURCES
            rows = rows[:MAX_SOURCES]
            if rows:
                # Object-type names (display only; cached on disk for 30 days).
                # Fetched *after* the main query so that, if the master was down,
                # this request goes straight to the mirror that just answered
                # instead of repeating the failover.
                self._stage("reading object types")
                t = time.monotonic()
                types["defs"] = _otype_definitions()
                types["secs"] = time.monotonic() - t
            info = {"truncated": truncated,
                    "otypes": types.get("defs", {}) if rows else {}}
            _cache_save(cache, rows, info)
            info["timing"] = dict(stats, **extra, types=types.get("secs", 0.0))
            self.finished.emit(self.generation, rows, info)

        except SimbadError as e:
            self.error.emit(self.generation, str(e))
        except Exception as e:                      # never die silently
            self.error.emit(self.generation, f"Unexpected error during SIMBAD look-up: {e}")


class SimbadAliasWorker(QObject):
    """
    Fetches the alternative identifiers of one object (SIMBAD `ident`
    table).  Call run() from a background thread.

    finished(int, list)   request id, list of identifier strings
    error(int, str)       request id, message
    """

    finished = pyqtSignal(int, list)
    error    = pyqtSignal(int, str)

    def __init__(self, request_id: int, oid):
        super().__init__()
        self.request_id = request_id
        self.oid        = int(oid)        # int() also guards the ADQL string

    def run(self):
        try:
            rows = tap_query(f"SELECT id FROM ident WHERE oidref = {self.oid}")
            names = {" ".join((r.get("id") or "").split()) for r in rows}
            names.discard("")
            self.finished.emit(self.request_id, sorted(names, key=str.casefold))
        except SimbadError as e:
            self.error.emit(self.request_id, str(e))
        except Exception as e:
            self.error.emit(self.request_id, f"Unexpected error fetching names: {e}")



# ---------------------------------------------------------------------------
# NED
# ---------------------------------------------------------------------------

_ned_columns_cache = None     # set of column names of NED's objdir table


def _ned_columns(force: bool = False):
    """Column names of NED's objdir table, or None if unknown (see _basic_columns)."""
    global _ned_columns_cache
    if not force:
        if _ned_columns_cache:
            return _ned_columns_cache
        got = _json_get("schema_ned.json.gz", SCHEMA_CACHE_DAYS)
        if got and all(c in got[0] for c in _NED_REQUIRED):
            _ned_columns_cache = set(got[0])
            return _ned_columns_cache
        return None
    try:
        rows = tap_query("SELECT column_name FROM TAP_SCHEMA.columns "
                         "WHERE table_name = 'NEDTAP.objdir'",
                         timeout=NED_TIMEOUT, urls=NED_TAP_URLS, label="NED")
        cols = {(r.get("column_name") or "").strip() for r in rows} - {""}
        if all(c in cols for c in _NED_REQUIRED):
            _ned_columns_cache = cols
            _json_put("schema_ned.json.gz", sorted(cols))
            return cols
    except SimbadError:
        pass
    return None


def _ned_normalise(rows: list) -> list:
    """NED objdir rows -> the SIMBAD-style keys the rest of the module uses."""
    out = []
    for r in rows:
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        out.append({
            "main_id":      r.get("prefname", ""),
            "ra":           r.get("ra", ""),
            "dec":          r.get("dec", ""),
            "otype":        r.get("pretype", ""),
            "rvz_redshift": r.get("z", ""),
            "zflag":        r.get("zflag", ""),
            "n_crosref":    r.get("n_crosref", ""),
        })
    return out


def _ned_types() -> dict:
    return {code: (code, desc) for code, desc in NED_TYPES.items()}


class NedQueryWorker(SimbadQueryWorker):
    """Cone-searches NED's object directory.  Same signals as SimbadQueryWorker."""

    source = "NED"

    @staticmethod
    def _columns(avail) -> list:
        return [c for c in NED_COLUMNS if c in _NED_REQUIRED or avail is None or c in avail]

    def _query(self, columns: list, stats: dict) -> list:
        # Most cross-referenced objects first, so that if the field holds more
        # objects than we are willing to draw, the cut removes the least
        # well-established ones.
        order = " ORDER BY n_crosref DESC" if "n_crosref" in columns else ""
        adql = (
            f"SELECT {', '.join(columns)} FROM objdir "
            f"WHERE CONTAINS(POINT('J2000', ra, dec), "
            f"CIRCLE('J2000', {self.ra:.8f}, {self.dec:.8f}, {self.radius:.8f})) = 1{order}"
        )
        return tap_query(
            adql, timeout=NED_TIMEOUT, stats=stats, urls=NED_TAP_URLS, label="NED",
            extra={"MAXREC": MAX_SOURCES + 1},
            progress=lambda n, b: self._stage("receiving", max(n - 1, 0), b))

    def run(self):
        try:
            cache = _cache_path(self.ra, self.dec, self.radius, "NED")
            hit = _cache_load(cache)
            if hit:
                rows, info, age = hit
                info["timing"] = {"cached_days": age}
                self.finished.emit(self.generation, rows, info)
                return

            stats, extra = {}, {}
            columns = self._columns(_ned_columns())         # cached schema, never a network call
            self._stage("waiting for NED to answer")
            try:
                rows = self._query(columns, stats)
            except SimbadError as e:
                if "HTTP 400" not in str(e):
                    raise
                self._stage("checking the NED schema (first time only)")
                t = time.monotonic()
                fresh = _ned_columns(force=True)
                extra["schema"] = time.monotonic() - t
                retry = self._columns(fresh) if fresh else None
                if not retry or retry == columns:
                    raise
                stats.clear()
                self._stage("waiting for NED to answer")
                rows = self._query(retry, stats)

            rows = _ned_normalise(rows)
            truncated = len(rows) > MAX_SOURCES
            rows = rows[:MAX_SOURCES]
            info = {"truncated": truncated, "otypes": _ned_types()}
            _cache_save(cache, rows, info)
            info["timing"] = dict(stats, **extra)
            self.finished.emit(self.generation, rows, info)

        except SimbadError as e:
            self.error.emit(self.generation, str(e))
        except Exception as e:                      # never die silently
            self.error.emit(self.generation, f"Unexpected error during NED look-up: {e}")


class NedLookupWorker(QObject):
    """
    Fetches aliases and reference codes of one NED object (ObjectLookup
    service).  Call run() from a background thread.

    extras(int, dict)     request id, {"redshift_ref", "position_ref", "type_ref", "z_unc", ...}
    finished(int, list)   request id, alternative names
    error(int, str)
    """

    extras   = pyqtSignal(int, dict)
    finished = pyqtSignal(int, list)
    error    = pyqtSignal(int, str)

    def __init__(self, request_id: int, name: str):
        super().__init__()
        self.request_id = request_id
        self.name       = name

    def run(self):
        try:
            body = json.dumps({"name": {"v": self.name}, "aliases": {"v": True}},
                              separators=(",", ":"))
            try:
                text = _post(NED_LOOKUP_URL, urllib.parse.urlencode({"json": body}).encode(),
                             HTTP_TIMEOUT)
            except urllib.error.HTTPError as e:
                raise SimbadError(f"NED returned HTTP {e.code}")
            except (urllib.error.URLError, socket.timeout, OSError) as e:
                raise SimbadError(f"cannot reach NED: {getattr(e, 'reason', e)}")
            try:
                data = json.loads(text)
            except ValueError:
                raise SimbadError("NED sent an unreadable answer")
            if data.get("ResultCode") != 3:
                raise SimbadError(f"NED does not know the name '{self.name}'")

            pref = data.get("Preferred") or {}
            red  = pref.get("Redshift") or {}
            extras = {
                "redshift_ref": (red.get("RefCode") or "").strip(),
                "z_unc":        red.get("Uncertainty"),
                "z_quality":    red.get("QualityFlag"),
                "position_ref": ((pref.get("Position") or {}).get("RefCode") or "").strip(),
                "type_ref":     ((pref.get("ObjType") or {}).get("RefCode") or "").strip(),
            }
            names = {" ".join(str(n).split()) for n in (data.get("Aliases") or [])}
            names.discard("")
            self.extras.emit(self.request_id, extras)
            self.finished.emit(self.request_id, sorted(names, key=str.casefold))
        except SimbadError as e:
            self.error.emit(self.request_id, str(e))
        except Exception as e:
            self.error.emit(self.request_id, f"Unexpected error fetching NED details: {e}")


# ---------------------------------------------------------------------------
# Details dialog
# ---------------------------------------------------------------------------

class SimbadDetailsDialog(QDialog):
    """Non-modal viewer for one SIMBAD record."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SIMBAD")
        self.setModal(False)
        self.resize(480, 560)
        self._url = ""
        self._main_norm = ""

        layout = QVBoxLayout(self)
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.title_label)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Property", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setWordWrap(True)
        layout.addWidget(self.table, 3)

        self.alias_label = QLabel("Alternative names")
        self.alias_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.alias_label)
        self.alias_box = QPlainTextEdit()
        self.alias_box.setReadOnly(True)
        layout.addWidget(self.alias_box, 2)

        buttons = QHBoxLayout()
        self.open_button = QPushButton("Open in SIMBAD")
        self._rows = []
        self.open_button.clicked.connect(self._open_in_browser)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addStretch(1)
        buttons.addWidget(self.open_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def show_source(self, src: dict, otypes: dict):
        name = src.get("main_id", "?")
        ned  = src.get("_source") == "NED"
        service = "NED" if ned else "SIMBAD"
        self.setWindowTitle(f"{service} – {name}")
        self.open_button.setText(f"Open in {service}")
        self.title_label.setText(name)
        self._url = (NED_OBJECT_URL if ned else SIMBAD_ID_URL) + urllib.parse.quote(name)
        self._main_norm = " ".join(name.split())      # to drop it from the alias list

        rows = []                                     # (label, value, is_bibcode)
        for column, label in (NED_FIELDS if ned else FIELDS):
            value = (src.get(column) or "").strip()
            if not value:
                continue
            if column == "otype":
                label_txt, desc = otypes.get(value, ("", ""))
                if desc and desc != value:
                    value = f"{value} — {desc}"
            rows.append((label, value, column.endswith("_bibcode")))

        ra, dec = _to_float(src.get("ra")), _to_float(src.get("dec"))
        if ra is not None and dec is not None:
            sexa = SkyCoord(ra, dec, unit="deg").to_string("hmsdms", precision=2)
            # Placed right after the identifier/type rows.
            rows.insert(min(2, len(rows)), ("RA, Dec (deg)", f"{ra:.7f}, {dec:+.7f}", False))
            rows.insert(min(3, len(rows)), ("RA, Dec (hms dms)", sexa, False))
        if src.get("_x") is not None:
            rows.append(("Image position (x, y)", f"{src['_x']:.1f}, {src['_yfits']:.1f}", False))

        self._rows = rows
        self._fill_table()
        self.show()
        self.raise_()
        self.activateWindow()

    def _fill_table(self):
        rows = self._rows
        self.table.setRowCount(0)                     # also drops old cell widgets
        self.table.setRowCount(len(rows))
        for i, (k, v, is_ref) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(k))
            if is_ref:
                # References link to their ADS abstract page.
                link = QLabel(f'<a href="{html.escape(ads_url(v), quote=True)}">{html.escape(v)}</a>')
                link.setTextInteractionFlags(Qt.TextBrowserInteraction)
                link.setContentsMargins(4, 0, 4, 0)
                link.setToolTip("Open the abstract in ADS")
                link.linkActivated.connect(lambda url: QDesktopServices.openUrl(QUrl(url)))
                self.table.setCellWidget(i, 1, link)
            else:
                self.table.setItem(i, 1, QTableWidgetItem(v))
        self.table.resizeRowsToContents()

    def add_ned_extras(self, extras: dict):
        """Rows that only NED's ObjectLookup knows (references, redshift uncertainty)."""
        more = []
        z_unc = _to_float(extras.get("z_unc"))
        if z_unc is not None:
            more.append(("Redshift uncertainty", f"{z_unc:g}", False))
        if extras.get("z_quality"):
            more.append(("Redshift quality (lookup)", str(extras["z_quality"]), False))
        for key, label in (("redshift_ref", "Redshift reference"),
                           ("position_ref", "Position reference"),
                           ("type_ref",     "Object type reference")):
            if extras.get(key):
                more.append((label, extras[key], True))
        if more:
            self._rows = self._rows + more
            self._fill_table()

    # -- alternative names (filled in asynchronously by the manager) -------
    def set_aliases_loading(self):
        self.alias_label.setText("Alternative names")
        self.alias_box.setPlainText("Loading …")

    def set_aliases(self, names: list):
        names = [n for n in names if n != self._main_norm]    # main id is shown above
        self.alias_label.setText(f"Alternative names ({len(names)})")
        self.alias_box.setPlainText("\n".join(names) if names else "(none listed)")

    def set_aliases_error(self, message: str):
        self.alias_label.setText("Alternative names")
        self.alias_box.setPlainText(f"Not available: {message}")

    def _open_in_browser(self):
        if self._url:
            QDesktopServices.openUrl(QUrl(self._url))


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

class _SortItem(QTableWidgetItem):
    """Table cell that sorts by a key (numbers numerically, empty cells last)."""

    def __init__(self, text: str, key):
        super().__init__(text)
        # (is_empty, value): empties compare greater than any value.  Plain
        # Python floats only: a numpy scalar would make __lt__ return a
        # numpy.bool_, which PyQt rejects (and aborts on).
        if isinstance(key, (int, float, np.integer, np.floating)):
            key = float(key)
        self.key = (0, key) if key is not None else (1, 0)

    def __lt__(self, other):
        other_key = getattr(other, "key", None)
        if other_key is None:
            return super().__lt__(other)
        return bool(self.key < other_key)


class _ServicePage(QWidget):
    """
    One tab of the table dialog: the objects one service found in the field of
    view (name, type, coordinates, redshift and, for SIMBAD, a reference), or a
    note while that service is still being queried / has failed.

    Signals (int = index into SimbadManager.sources)
    -------
    row_clicked(int)     click on a row -> centre the view
    row_activated(int)   double click -> open the details dialog
    """

    row_clicked   = pyqtSignal(int)
    row_activated = pyqtSignal(int)

    REF_COL = 5
    COLUMNS = ["Name", "Type", "RA", "Dec", "Redshift", "Reference"]
    COORD_HEADERS = {"sexagesimal": ("RA (h:m:s)", "Dec (d:m:s)"), "decimal": ("RA (deg)", "Dec (deg)")}

    def __init__(self, service: str, fmt: str, parent=None):
        super().__init__(parent)
        self.service  = service
        self._fmt     = fmt
        self._text    = {}                 # fmt -> {source index: (ra_text, dec_text)}
        self._link_font  = QFont(); self._link_font.setUnderline(True)
        self._link_brush = QBrush(QColor(26, 95, 180))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.note, 1)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setMouseTracking(True)              # hand cursor over links
        self.table.viewport().installEventFilter(self)
        hdr = self.table.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        self._set_headers()
        # Explicit mouse clicks only: currentCellChanged would also fire when
        # the table is merely re-sorted, and itemActivated fires on a single
        # click under KDE's "single-click to open" setting.
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(lambda r, c: self._emit(self.row_activated, r))
        self.table.cellEntered.connect(self._on_cell_entered)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.table, 1)
        # NED's object directory carries no per-object bibliographic reference
        # (those are in the details dialog), so its column would stay empty.
        self.table.setColumnHidden(self.REF_COL, service == "NED")
        self.show_note("waiting for the query …")

    # -- note / table switch ---------------------------------------------------
    def show_note(self, text: str):
        self.note.setText(text)
        self.note.show()
        self.table.hide()

    # -- coordinate format ---------------------------------------------------
    def set_format(self, fmt: str):
        self._fmt = fmt
        self._refresh_coordinates()

    def _set_headers(self):
        ra_h, dec_h = self.COORD_HEADERS[self._fmt]
        labels = list(self.COLUMNS)
        labels[2], labels[3] = ra_h, dec_h
        self.table.setHorizontalHeaderLabels(labels)

    def _coord_cells(self, idx: int):
        """(ra_text, dec_text, ra_tooltip, dec_tooltip) of source idx in the current format."""
        other = "decimal" if self._fmt == "sexagesimal" else "sexagesimal"
        ra_t, dec_t = self._text[self._fmt][idx]
        ra_o, dec_o = self._text[other][idx]
        suffix = "°" if other == "decimal" else ""
        return ra_t, dec_t, ra_o + suffix, dec_o + suffix

    def _refresh_coordinates(self):
        """Re-render the RA/Dec columns after a format change (sort order is unaffected)."""
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self._set_headers()
        for row in range(self.table.rowCount()):
            idx = int(self.table.item(row, 0).data(Qt.UserRole))
            ra_t, dec_t, ra_tip, dec_tip = self._coord_cells(idx)
            for col, text, tip in ((2, ra_t, ra_tip), (3, dec_t, dec_tip)):
                item = self.table.item(row, col)
                item.setText(text)
                item.setToolTip(tip)
        self.table.setSortingEnabled(sorting)
        self.table.resizeColumnToContents(2)
        self.table.resizeColumnToContents(3)

    # -- filling ---------------------------------------------------------------
    def populate(self, entries: list, otypes: dict):
        """*entries*: list of (index into SimbadManager.sources, source dict)."""
        n = len(entries)
        ra  = np.array([float(s["ra"])  for _, s in entries])
        dec = np.array([float(s["dec"]) for _, s in entries])
        if n:
            sexa = np.atleast_1d(SkyCoord(ra, dec, unit="deg")
                                 .to_string("hmsdms", sep=":", precision=2))   # one vectorised call
            sexa = [x.split(" ", 1) + [""] for x in sexa]
        else:
            sexa = []
        self._text["sexagesimal"] = {idx: (sexa[k][0], sexa[k][1]) for k, (idx, _) in enumerate(entries)}
        self._text["decimal"]     = {idx: (f"{ra[k]:.6f}", f"{dec[k]:+.6f}") for k, (idx, _) in enumerate(entries)}

        self.table.setSortingEnabled(False)
        self.table.setRowCount(n)
        for row, (idx, src) in enumerate(entries):
            name  = src.get("main_id", "")
            otype = (src.get("otype") or "").strip()
            tdesc = otypes.get(otype, ("", ""))[1] or otype
            z_txt = (src.get("rvz_redshift") or "").strip()
            bib, what = pick_reference(src)

            first = _SortItem(name, name.casefold())
            first.setData(Qt.UserRole, idx)              # index into manager.sources
            self.table.setItem(row, 0, first)
            self.table.setItem(row, 1, _SortItem(tdesc, tdesc.casefold()))
            ra_t, dec_t, ra_tip, dec_tip = self._coord_cells(idx)
            ra_item, dec_item = _SortItem(ra_t, ra[row]), _SortItem(dec_t, dec[row])
            ra_item.setToolTip(ra_tip)
            dec_item.setToolTip(dec_tip)
            self.table.setItem(row, 2, ra_item)
            self.table.setItem(row, 3, dec_item)
            self.table.setItem(row, 4, _SortItem(z_txt, _to_float(z_txt)))
            ref = _SortItem(bib, bib.casefold() if bib else None)
            if bib:
                ref.setData(Qt.UserRole + 1, bib)        # marks the cell as a link
                ref.setFont(self._link_font)
                ref.setForeground(self._link_brush)
                ref.setToolTip(f"{what.capitalize()} reference — click to open the ADS abstract\n{ads_url(bib)}")
            self.table.setItem(row, 5, ref)

        self.table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(0, min(max(self.table.columnWidth(0), 140), 320))
        self.note.hide()
        self.table.show()

    # -- interaction -----------------------------------------------------------------
    def select_source(self, idx: int):
        """Highlight (and scroll to) the row of source *idx*, without emitting."""
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.UserRole) == idx:
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.setCurrentCell(row, 0)
                self.table.blockSignals(False)
                self.table.scrollToItem(self.table.item(row, 0))
                return

    def _bibcode_at(self, row: int, col: int) -> str:
        item = self.table.item(row, col) if col == self.REF_COL else None
        return (item.data(Qt.UserRole + 1) or "") if item is not None else ""

    def _on_cell_clicked(self, row: int, col: int):
        bib = self._bibcode_at(row, col)
        if bib:                                   # a reference: open ADS, do not move the view
            QDesktopServices.openUrl(QUrl(ads_url(bib)))
            return
        self._emit(self.row_clicked, row)

    def _on_context_menu(self, pos):
        """Right-click on a reference cell: open / copy the ADS link or bibcode."""
        item = self.table.itemAt(pos)
        bib = self._bibcode_at(item.row(), item.column()) if item is not None else ""
        if not bib:
            return
        menu = QMenu(self.table)
        menu.addAction("Open in ADS", lambda: QDesktopServices.openUrl(QUrl(ads_url(bib))))
        menu.addAction("Copy ADS link", lambda: QApplication.clipboard().setText(ads_url(bib)))
        menu.addAction("Copy bibcode", lambda: QApplication.clipboard().setText(bib))
        self._menu = menu                         # kept for inspection/tests
        menu.exec_(self.table.viewport().mapToGlobal(pos))

    def _on_cell_entered(self, row: int, col: int):
        if self._bibcode_at(row, col):
            self.table.viewport().setCursor(Qt.PointingHandCursor)
        else:
            self.table.viewport().unsetCursor()

    def eventFilter(self, obj, event):
        if obj is self.table.viewport() and event.type() == QEvent.Leave:
            obj.unsetCursor()
        return super().eventFilter(obj, event)

    def _emit(self, signal, row: int):
        if row < 0:
            return
        item = self.table.item(row, 0)
        if item is not None:
            signal.emit(int(item.data(Qt.UserRole)))


class SimbadTableDialog(QDialog):
    """
    Non-modal dialog listing the objects found in the field of view, one tab
    per service (NED, SIMBAD).  A tab shows a note while its service is still
    being queried or if it failed.

    The coordinate format (DD.ddd or h:m:s) applies to all tabs and is
    remembered between sessions.

    Signals (int = index into SimbadManager.sources)
    -------
    row_clicked(int)     click on a row -> centre the view
    row_activated(int)   double click -> open the details dialog
    """

    row_clicked   = pyqtSignal(int)
    row_activated = pyqtSignal(int)

    REF_COL = _ServicePage.REF_COL
    FORMATS = [("decimal", "DD.ddd"), ("sexagesimal", "h:m:s")]
    COORD_HEADERS = _ServicePage.COORD_HEADERS
    COLUMNS = _ServicePage.COLUMNS

    def __init__(self, parent=None, services=SOURCES):
        super().__init__(parent)
        self.setWindowTitle("Objects in the field of view")
        self.setModal(False)
        self.resize(860, 460)
        self._fmt = self._load_format()

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.hint = QLabel("Double-click for full record.")
        self.hint.setToolTip("Click a row to centre the view on the object. Click a reference to open it "
                             "in ADS (right-click to copy the link). Click a column header to sort.")
        top.addWidget(self.hint, 1)
        self.format_group = QButtonGroup(self)
        self.format_buttons = {}
        for key, label in self.FORMATS:
            rb = QRadioButton(label)
            rb.setChecked(key == self._fmt)
            self.format_group.addButton(rb)
            self.format_buttons[key] = rb
            top.addWidget(rb)
            rb.toggled.connect(lambda on, k=key: on and self._on_format_changed(k))
        layout.addLayout(top)

        self.tabs = QTabWidget()
        self.pages = {}
        for name in services:
            page = _ServicePage(name, self._fmt)
            page.row_clicked.connect(self.row_clicked)
            page.row_activated.connect(self.row_activated)
            self.pages[name] = page
            self.tabs.addTab(page, f"{name} (querying …)")
        layout.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    # -- coordinate format ---------------------------------------------------
    @staticmethod
    def _load_format() -> str:
        try:
            v = QSettings("euniverse", "euniverse").value("simbad/coord_format", "sexagesimal")
            return v if v in ("sexagesimal", "decimal") else "sexagesimal"
        except Exception:
            return "sexagesimal"

    def _on_format_changed(self, fmt: str):
        self._fmt = fmt
        try:
            QSettings("euniverse", "euniverse").setValue("simbad/coord_format", self._fmt)
        except Exception:
            pass
        for page in self.pages.values():
            page.set_format(fmt)

    # -- access ----------------------------------------------------------------
    def page(self, service: str) -> "_ServicePage":
        return self.pages[service]

    @property
    def table(self) -> QTableWidget:
        """Table of the tab that is currently shown."""
        return self.tabs.currentWidget().table

    def _set_tab(self, service: str, label: str, tip: str = ""):
        i = self.tabs.indexOf(self.pages[service])
        self.tabs.setTabText(i, f"{service} ({label})")
        self.tabs.setTabToolTip(i, tip)

    # -- filling (called by the manager as the answers arrive) ---------------------
    def populate_service(self, service: str, entries: list, otypes: dict, tip: str = ""):
        self.pages[service].populate(entries, otypes)
        self._set_tab(service, f"{len(entries):,}", tip)

    def set_note(self, service: str, label: str, text: str, tip: str = ""):
        """Tab text and a message in place of the table (querying, failed, empty)."""
        self.pages[service].show_note(text)
        self._set_tab(service, label, tip)

    def select_source(self, service: str, idx: int):
        """Show the tab of *service* and highlight source *idx* in it."""
        page = self.pages.get(service)
        if page is not None:
            self.tabs.setCurrentWidget(page)
            page.select_source(idx)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class _Relay(QObject):
    """Routes one worker's signals to the manager, tagged with the service.

    A QObject living in the GUI thread, so signals emitted from the worker's
    plain thread are delivered as queued calls on the GUI thread."""

    def __init__(self, manager, service: str):
        super().__init__(manager)
        self._m, self._service = manager, service

    def finished(self, gen: int, rows: list, info: dict):
        self._m._on_finished(self._service, gen, rows, info)

    def error(self, gen: int, message: str):
        self._m._on_error(self._service, gen, message)

    def progress(self, gen: int, stage: str, rows: int, nbytes: int):
        self._m._on_progress(self._service, gen, stage, rows, nbytes)


class SimbadManager(QObject):
    """
    Owns everything belonging to the look-up overlay of one ImageViewer.

    NED and SIMBAD are always queried together, in parallel.  NED objects get
    green circles, SIMBAD objects yellow squares.  One service failing or
    being slow does not hold up the other.

    Public API (called by ImageViewer / ControlDock)
    ------------------------------------------------
    start_lookup()                 begin asynchronous queries for the image
    clear()                        remove overlay, drop the catalogs, cancel
    handle_click(scene_pos, r)     True if a click hit an object (opens details)
    show_details(idx)              details dialog incl. alternative names
    center_on(idx)                 centre the view on an object (table row click)

    ``sources`` / ``items`` hold the objects of both services in arrival
    order; each source dict carries ``_source`` ("NED" or "SIMBAD").
    """

    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer   = viewer
        self.sources  = []          # list of dict (catalogue row + _x, _y, _r, _source)
        self.items    = []          # QGraphicsItem (circle or square), parallel to sources
        self._xy      = np.empty((0, 2))
        self._radius  = np.empty(0)
        self._square  = np.empty(0, dtype=bool)     # True for SIMBAD squares
        self.otypes   = {}          # service -> {code: (label, description)}
        self._gen     = 0           # bumped by start_lookup() and clear()
        self._workers = {}          # (gen, service) -> worker, keeps them alive while running
        self._relays  = {s: _Relay(self, s) for s in SOURCES}
        self._st      = {s: self._new_state() for s in SOURCES}
        self._selected = None       # index of the highlighted object
        self._dialog  = None        # SimbadDetailsDialog (one, reused)
        self._table   = None        # SimbadTableDialog
        self._table_opened = False  # opened once per look-up; not re-opened after the user closes it
        # Alternative names: fetched on demand and cached.
        self._alias_cache   = {}    # SIMBAD oid -> list of names
        self._alias_pending = {}    # request id -> (worker, key) (keeps it alive)
        self._alias_req     = 0
        self._alias_current = None  # request id the details dialog is waiting for
        self._ned_cache     = {}    # NED name -> (aliases, extras)
        # Progress counter shown in the status bar while queries are running
        self._timer     = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

    @staticmethod
    def _new_state() -> dict:
        return dict(state="idle", stage="", rows=0, nbytes=0, t0=0.0, n=0, secs=0.0,
                    truncated=False, text="", timing="")

    # ------------------------------------------------------------------ query

    def start_lookup(self) -> bool:
        v = self.viewer
        if v.original_image is None or v.wcs is None:
            self._say("Look-up needs an image with a valid WCS.", 5000)
            return False

        self.clear()
        h, w = v.original_image.shape[:2]
        try:
            centre = v.wcs.pixel_to_world(w / 2.0, h / 2.0)
            corners = [v.wcs.pixel_to_world(x, y)
                       for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
            c0 = SkyCoord(centre[0], centre[1], unit="deg")
            cc = SkyCoord([c[0] for c in corners], [c[1] for c in corners], unit="deg")
            radius = float(c0.separation(cc).deg.max()) * 1.01
        except Exception as e:
            self._say(f"Look-up: cannot determine the field of view ({e}).", 6000)
            return False

        self._gen += 1
        now = time.monotonic()
        for service, worker_class in (("NED", NedQueryWorker), ("SIMBAD", SimbadQueryWorker)):
            relay  = self._relays[service]
            worker = worker_class(self._gen, centre[0], centre[1], radius)
            worker.finished.connect(relay.finished)
            worker.error.connect(relay.error)
            worker.progress.connect(relay.progress)
            self._workers[(self._gen, service)] = worker
            self._st[service] = dict(self._new_state(), state="pending", stage=f"contacting {service}", t0=now)
        self._timer.start()
        self._publish()
        for service in SOURCES:
            worker = self._workers[(self._gen, service)]
            threading.Thread(target=worker.run, name=f"{service.lower()}-lookup", daemon=True).start()
        return True

    # ------------------------------------------------------------- progress / status

    def _on_progress(self, service: str, gen: int, stage: str, rows: int, nbytes: int):
        if gen == self._gen:
            st = self._st[service]
            st["stage"], st["rows"], st["nbytes"] = stage, rows, nbytes
            self._publish()

    def _tick(self):
        """Fired by the timer so the seconds keep running while a query is silent."""
        self._publish()

    def _all_done(self) -> bool:
        return all(st["state"] in ("ok", "error") for st in self._st.values())

    def _status_part(self, service: str) -> str:
        st = self._st[service]
        if st["state"] == "pending":
            elapsed = time.monotonic() - st["t0"]
            if st["rows"]:
                return (f"{service}: {st['stage']} … {st['rows']:,} sources, "
                        f"{st['nbytes'] / 1e6:.1f} MB — {elapsed:.0f} s")
            return f"{service}: {st['stage']} … {elapsed:.0f} s"
        return st["text"] or f"{service}: —"

    def _publish(self):
        """One status-bar line with the state of both services."""
        msg = "  |  ".join(self._status_part(s) for s in SOURCES)
        self._say(msg, 0 if not self._all_done() else 20000)
        if self._all_done():
            self._timer.stop()

    @staticmethod
    def _timing_text(t: dict, total: float) -> str:
        """Human-readable timing summary, to see where the time went."""
        if t.get("cached_days") is not None:
            return f"from cache, {t['cached_days']:.1f} days old"
        parts = []
        if "wait" in t:
            parts.append(f"server {t['wait']:.1f} s")
            parts.append(f"download {t['download']:.1f} s")
            parts.append(f"{t['bytes'] / 1e6:.1f} MB" + (", gzip" if t.get("gzip") else ", uncompressed"))
        if t.get("mirror"):
            parts.append(f"via {t['mirror']}")
        for f in t.get("failovers", []):
            parts.append(f"failed over from {f}")
        if t.get("schema", 0) >= 0.5:
            parts.append(f"schema check {t['schema']:.1f} s")
        if t.get("types", 0) >= 0.5:
            parts.append(f"object types {t['types']:.1f} s")
        parts.append(f"total {total:.1f} s")
        return ", ".join(parts)

    # --------------------------------------------------------------- results

    def _on_finished(self, service: str, gen: int, rows: list, info: dict):
        self._workers.pop((gen, service), None)
        if gen != self._gen:
            return                      # user untoggled / loaded another image
        st = self._st[service]
        secs = time.monotonic() - st["t0"]

        v = self.viewer
        if v.original_image is None or v.wcs is None:
            return
        h, w = v.original_image.shape[:2]

        ra  = np.array([_to_float(r.get("ra"))  if _to_float(r.get("ra"))  is not None else np.nan for r in rows])
        dec = np.array([_to_float(r.get("dec")) if _to_float(r.get("dec")) is not None else np.nan for r in rows])
        good = np.isfinite(ra) & np.isfinite(dec)
        rows, ra, dec = [r for r, g in zip(rows, good) if g], ra[good], dec[good]
        for r in rows:
            r["_source"] = service

        if len(rows):
            x, y = v.wcs.world_to_pixel(SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs"))
            y = h - y                                   # FITS (bottom-up) -> Qt (top-down)
            inside = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w) & (y >= 0) & (y <= h)
        else:
            x = y = np.empty(0)
            inside = np.zeros(0, dtype=bool)

        pix_arcsec = self._pixel_scale_arcsec(w, h)
        default_r  = max(MIN_RADIUS_PX, DEFAULT_RADIUS_ARCSEC / pix_arcsec) if pix_arcsec else 12.0
        square     = service == "SIMBAD"
        colour     = SQUARE_COLOR if square else CIRCLE_COLOR

        self.otypes[service] = info.get("otypes", {})
        base = len(self.sources)
        mine, radii = [], []
        v.setUpdatesEnabled(False)
        try:
            for r, xi, yi, ok in zip(rows, x, y, inside):
                if not ok:
                    continue
                radius = default_r
                maj = _to_float(r.get("galdim_majaxis"))        # arcmin (SIMBAD only)
                if maj and pix_arcsec:
                    radius = min(MAX_RADIUS_PX, max(default_r, 0.5 * maj * 60.0 / pix_arcsec))
                r["_x"], r["_y"] = float(xi), float(yi)
                r["_yfits"] = float(h - yi)           # same convention as the dock's x/y readout

                if square:
                    half = radius * SQUARE_SCALE
                    item = _SourceSquare(QRectF(xi - half, yi - half, 2 * half, 2 * half))
                    item.setZValue(1000.0 / radius - 0.5)        # squares below circles of equal size
                else:
                    item = _SourceCircle(QRectF(xi - radius, yi - radius, 2 * radius, 2 * radius))
                    item.setZValue(1000.0 / radius)              # small objects on top of big ones
                pen = QPen(colour, CIRCLE_WIDTH)
                pen.setCosmetic(True)
                item.setPen(pen)
                item.setToolTip(self._tooltip(r))
                item.setData(0, base + len(mine))
                v.scene.addItem(item)

                mine.append((base + len(mine), r))
                self.items.append(item)
                radii.append(radius)
        finally:
            v.setUpdatesEnabled(True)
            v.scene.update()

        self.sources.extend(r for _, r in mine)
        if mine:
            self._xy = np.vstack([self._xy, np.array([[r["_x"], r["_y"]] for _, r in mine])])
        self._radius = np.concatenate([self._radius, np.array(radii)])
        self._square = np.concatenate([self._square, np.full(len(mine), square, dtype=bool)])

        n = len(mine)
        timing = self._timing_text(info.get("timing", {}), secs)
        full = f"{service}: {n:,} source{'s' if n != 1 else ''} in the field of view ({timing})"
        short = f"{service}: {n:,} source{'s' if n != 1 else ''}"
        cached = info.get("timing", {}).get("cached_days")
        short += f" ({'cache' if cached is not None else f'{secs:.1f} s'})"
        if info.get("truncated"):
            full  += f" — query capped at {MAX_SOURCES:,}; list is incomplete"
            short += " — capped, list incomplete"
        elif n == 0:
            full  = f"{service}: no known sources in the field of view ({timing})"
            short = f"{service}: none ({secs:.1f} s)"
        st.update(state="ok", n=n, secs=secs, text=short, timing=timing, truncated=bool(info.get("truncated")))
        print(full)                     # full detail on the console, for timing comparisons

        if n:
            self._ensure_table()
            if self._table is not None:
                self._table.populate_service(service, mine, self.otypes[service], timing)
        elif self._table is not None:
            self._table.set_note(service, "none", f"No {service} objects in the field of view.", timing)
        self._publish()
        self._release_if_failed()

    def _on_error(self, service: str, gen: int, message: str):
        self._workers.pop((gen, service), None)
        if gen != self._gen:
            return
        st = self._st[service]
        st.update(state="error", text=f"{service}: {message}", timing=message)
        print(message)
        if self._table is not None:
            self._table.set_note(service, "failed", message, message)
        self._publish()
        self._release_if_failed()

    def _release_if_failed(self):
        """All answers are in, none drew anything and some failed: release the button."""
        if self._all_done() and not self.sources and any(s["state"] == "error" for s in self._st.values()):
            dock = self.viewer.control_dock
            if dock is not None:
                dock.set_lookup_checked(False)

    # ------------------------------------------------------------- interaction

    def handle_click(self, scene_pos, hit_radius: float) -> bool:
        """
        Left-click at *scene_pos*; open the details of the object hit.

        Inside a NED circle -> that NED object; in the margin between circle and
        square (or inside a SIMBAD square with no circle there) -> the SIMBAD
        object.  Failing an exact hit, the nearest object within *hit_radius*.
        """
        if not len(self.items):
            return False
        dx = self._xy[:, 0] - scene_pos.x()
        dy = self._xy[:, 1] - scene_pos.y()
        d  = np.hypot(dx, dy)                                   # distance to the centre
        dsq = np.maximum(np.abs(dx), np.abs(dy))                # "square distance"
        circle, square = ~self._square, self._square

        def nearest(mask):
            hit = np.flatnonzero(mask)
            return int(hit[np.argmin(d[hit])]) if len(hit) else None

        idx = nearest(circle & (d <= self._radius))
        if idx is None:
            idx = nearest(square & (dsq <= self._radius * SQUARE_SCALE))
        if idx is None:                                         # forgiving slop for tiny objects
            idx = nearest((circle & (d <= hit_radius)) | (square & (dsq <= hit_radius)))
        if idx is None:
            return False
        self.show_details(idx)
        if self._table is not None:
            self._table.select_source(self.sources[idx]["_source"], idx)   # keep the table in sync
        return True

    def show_details(self, idx: int):
        """Open the details dialog for source *idx* and start fetching its aliases."""
        if not 0 <= idx < len(self.sources):
            return
        self._select(idx)
        if self._dialog is None:
            self._dialog = SimbadDetailsDialog(self.viewer)
        src = self.sources[idx]
        self._dialog.show_source(src, self.otypes.get(src["_source"], {}))

        if src.get("_source") == "NED":
            self._show_ned_details(src)
            return

        oid = (src.get("oid") or "").strip()
        if not oid:
            self._alias_current = None
            self._dialog.set_aliases_error("SIMBAD did not provide an object id")
        elif oid in self._alias_cache:
            self._alias_current = None
            self._dialog.set_aliases(self._alias_cache[oid])
        else:
            self._dialog.set_aliases_loading()
            self._alias_req += 1
            rid = self._alias_current = self._alias_req
            try:
                worker = SimbadAliasWorker(rid, oid)
            except ValueError:
                self._dialog.set_aliases_error(f"invalid object id {oid!r}")
                return
            worker.finished.connect(self._on_aliases)
            worker.error.connect(self._on_alias_error)
            self._alias_pending[rid] = (worker, oid)
            threading.Thread(target=worker.run, name="simbad-aliases", daemon=True).start()

    def _show_ned_details(self, src: dict):
        """Aliases and reference codes of a NED object (ObjectLookup service)."""
        name = src.get("main_id", "")
        if name in self._ned_cache:
            names, extras = self._ned_cache[name]
            self._alias_current = None
            self._dialog.add_ned_extras(extras)
            self._dialog.set_aliases(names)
            return
        self._dialog.set_aliases_loading()
        self._alias_req += 1
        rid = self._alias_current = self._alias_req
        worker = NedLookupWorker(rid, name)
        worker.extras.connect(self._on_ned_extras)
        worker.finished.connect(self._on_aliases)
        worker.error.connect(self._on_alias_error)
        self._alias_pending[rid] = (worker, ("NED", name))
        threading.Thread(target=worker.run, name="ned-details", daemon=True).start()

    def _on_ned_extras(self, rid: int, extras: dict):
        _, key = self._alias_pending.get(rid, (None, None))
        if key:
            self._ned_extras_pending = getattr(self, "_ned_extras_pending", {})
            self._ned_extras_pending[rid] = extras
        if rid == self._alias_current and self._dialog is not None:
            self._dialog.add_ned_extras(extras)

    def _on_aliases(self, rid: int, names: list):
        _, oid = self._alias_pending.pop(rid, (None, None))
        if isinstance(oid, tuple):                       # NED: ("NED", name)
            extras = getattr(self, "_ned_extras_pending", {}).pop(rid, {})
            self._ned_cache[oid[1]] = (names, extras)
        elif oid is not None:
            self._alias_cache[oid] = names
        if rid == self._alias_current and self._dialog is not None:
            self._dialog.set_aliases(names)

    def _on_alias_error(self, rid: int, message: str):
        self._alias_pending.pop(rid, None)
        if rid == self._alias_current and self._dialog is not None:
            self._dialog.set_aliases_error(message)

    def center_on(self, idx: int):
        """Centre the view on source *idx* and highlight its marker."""
        if not 0 <= idx < len(self.sources):
            return
        src = self.sources[idx]
        self._select(idx)
        self.viewer.centerOn(QPointF(src["_x"], src["_y"]))
        self.viewer.refresh_preview()            # keep the dock navigator in step

    def _ensure_table(self):
        """Open the table dialog when the first service has something to show."""
        if self._table is not None or self._table_opened:
            return
        self._table_opened = True
        self._table = SimbadTableDialog(self.viewer)
        self._table.row_clicked.connect(self.center_on)
        self._table.row_activated.connect(self.show_details)
        for service in SOURCES:                          # services that are still out / already empty
            st = self._st[service]
            if st["state"] == "pending":
                self._table.set_note(service, "querying …", f"Waiting for {service} …")
            elif st["state"] == "error":
                self._table.set_note(service, "failed", st["timing"], st["timing"])
            elif st["state"] == "ok" and st["n"] == 0:
                self._table.set_note(service, "none", f"No {service} objects in the field of view.", st["timing"])
        self._table.show()

    def _open_table(self):
        """(Re)build the table dialog from the catalogs currently held."""
        self._close_table()
        self._table_opened = False
        self._ensure_table()
        if self._table is None:
            return
        for service in SOURCES:
            entries = [(i, r) for i, r in enumerate(self.sources) if r.get("_source") == service]
            if entries:
                self._table.populate_service(service, entries, self.otypes.get(service, {}),
                                             self._st[service]["timing"])

    def _close_table(self):
        if self._table is not None:
            self._table.close()
            self._table.deleteLater()
            self._table = None

    def _select(self, idx: int):
        import sip
        if self._selected is not None and self._selected < len(self.items):
            old = self.items[self._selected]
            if not sip.isdeleted(old):
                pen = old.pen(); pen.setWidthF(CIRCLE_WIDTH); old.setPen(pen)
        item = self.items[idx]
        pen = item.pen(); pen.setWidthF(CIRCLE_WIDTH_SEL); item.setPen(pen)
        self._selected = idx
        self.viewer.scene.update()

    # ------------------------------------------------------------------ cleanup

    def clear(self):
        """Remove the overlay and the catalogs, and cancel any pending query."""
        import sip
        self._gen += 1                       # makes any in-flight answer stale
        self._timer.stop()
        for item in self.items:
            try:
                if not sip.isdeleted(item) and item.scene() is self.viewer.scene:
                    self.viewer.scene.removeItem(item)
            except RuntimeError:
                pass
        if self.items:
            self.viewer.scene.update()
        self.items, self.sources, self.otypes = [], [], {}
        self._xy, self._radius = np.empty((0, 2)), np.empty(0)
        self._square = np.empty(0, dtype=bool)
        self._st = {s: self._new_state() for s in SOURCES}
        self._selected = None
        self._alias_cache, self._alias_current = {}, None
        self._ned_cache = {}
        self._table_opened = False
        self._close_table()
        if self._dialog is not None:
            self._dialog.close()

    # ------------------------------------------------------------------ helpers

    def _pixel_scale_arcsec(self, w: int, h: int):
        try:
            a = self.viewer.wcs.pixel_to_world(w / 2.0,       h / 2.0)
            b = self.viewer.wcs.pixel_to_world(w / 2.0 + 1.0, h / 2.0)
            return float(SkyCoord(a[0], a[1], unit="deg")
                         .separation(SkyCoord(b[0], b[1], unit="deg")).arcsec)
        except Exception:
            return None

    def _tooltip(self, src: dict) -> str:
        name  = html.escape(src.get("main_id", "?"))
        service = src.get("_source", "")
        otype = (src.get("otype") or "").strip()
        label = self.otypes.get(service, {}).get(otype, ("", ""))[1] or otype
        extra = f" &nbsp;<i>{html.escape(label)}</i>" if label else ""
        z = (src.get("rvz_redshift") or "").strip() if service == "NED" else ""
        if z:
            extra += f" &nbsp;z = {html.escape(z)}"
        return f"<b>{name}</b>{extra} &nbsp;<small>({service})</small>"

    def _say(self, message: str, timeout: int = 5000):
        try:
            self.viewer.update_status(message, timeout)
        except Exception:
            pass
