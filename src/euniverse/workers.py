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
workers.py — Background QThread workers for Euniverse Explorer
===============================================================
All non-UI work that must not block the Qt event loop lives here.
Each worker follows the standard Qt pattern:

    worker = SomeWorker(args)
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.start()

Two workers are defined:

  TiffLoader   — loads a TIFF image and its JSON metadata from disk.
                 Used by ImageViewer when the user opens a file.

  CsvUploader  — POSTs a CSV file to the Euclid target-receiver endpoint.
                 Used by ControlDock when the user submits annotations.
"""

import json

import numpy as np
import tifffile

from . import diagnostics

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# TiffLoader
# ---------------------------------------------------------------------------

class TiffLoader(QObject):
    """
    Loads a Euclid MER tile TIFF from disk in a background thread.

    The TIFF must have:
      - a single page (page 0) containing the image array
      - an 'ImageDescription' tag containing JSON-encoded WCS metadata

    Signals
    -------
    finished(np.ndarray, dict, str)
        Emitted on success with (image_array, metadata_dict, file_path).
    error(str)
        Emitted on failure with a human-readable error message.
    """

    finished = pyqtSignal(np.ndarray, dict, str)
    error    = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        """
        Entry point.  Runs in a plain Python thread started by
        ImageViewer.load_image (the signals are queued to the GUI thread).
        """
        try:
            # NOTE: no cursor handling here.  QApplication.setOverrideCursor()
            # and restoreOverrideCursor() are GUI-thread-only; calling them
            # from this worker thread made Qt start timers off-thread
            # ("QObject::startTimer: Timers cannot be started from another
            # thread", seen on Wayland).  ImageViewer.load_image() sets the
            # wait cursor and the finished/error slots restore it.
            diagnostics.mark("loader: opening the TIFF")
            with tifffile.TiffFile(self.path) as tif:
                diagnostics.mark("loader: reading the pixel data")
                image = tif.pages[0].asarray()
                diagnostics.mark(f"loader: pixel data read, shape {image.shape}, dtype {image.dtype}, {image.nbytes / 1e6:.0f} MB")

                if 'ImageDescription' not in tif.pages[0].tags:
                    raise ValueError("No ImageDescription tag found in TIFF metadata")

                desc = tif.pages[0].tags['ImageDescription'].value
                try:
                    metadata = json.loads(desc)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON in ImageDescription tag")

            diagnostics.mark("loader: metadata parsed, emitting finished")
            self.finished.emit(image, metadata, self.path)
            diagnostics.mark("loader: finished emitted (loader thread done)")

        except ValueError as ve:
            diagnostics.mark(f"loader: error {ve}")
            self.error.emit(str(ve))

        except Exception as e:
            diagnostics.mark(f"loader: unexpected error {e!r}")
            self.error.emit(f"Unexpected error loading TIFF: {e}")


# ---------------------------------------------------------------------------
# CsvUploader
# ---------------------------------------------------------------------------

class CsvUploader(QObject):
    """
    Uploads a CSV file to the Euclid target-receiver endpoint in a background
    thread so the UI stays responsive during the network call.

    The endpoint URL is expected to eventually be
    https://www.euclid-ec.org/target_receiver — it does not yet exist at the
    time of writing (see the TODO comment in ControlDock.on_submit_targets).

    Signals
    -------
    done(str)
        Emitted when the upload attempt completes (success or failure) with a
        human-readable status message suitable for display in the status bar.
    """

    done = pyqtSignal(str)

    # Centralise the endpoint URL so it only needs to change in one place
    ENDPOINT = "https://www.euclid-ec.org/target_receiver"

    def __init__(self, csv_path: str):
        super().__init__()
        self._csv_path = csv_path

    def run(self):
        """Entry point — runs in a plain Python thread started by ControlDock.on_submit_targets."""
        import requests  # imported here so the rest of the app has no hard dep

        try:
            with open(self._csv_path, 'rb') as f:
                files    = {'file': (self._csv_path, f, 'text/csv')}
                response = requests.post(self.ENDPOINT, files=files, timeout=10)

            if response.status_code == 200:
                self.done.emit("Successfully uploaded targets.")
            else:
                self.done.emit(f"Upload failed (HTTP {response.status_code})")

        except requests.exceptions.RequestException as e:
            self.done.emit(f"Network error: Could not reach server. ({e})")
