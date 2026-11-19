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
curves_dialog.py — GIMP-style contrast-curve tool
==================================================
Two pieces:

  CurveWidget   A self-contained QWidget that draws and lets the user edit
                a tone curve: a set of (input, output) control points in
                normalized [0,1] x [0,1] space, connected by a monotone
                cubic Hermite spline (Fritsch-Carlson limiter — no
                ringing/overshoot, and it degenerates to an exact straight
                line for the default 2-point case). It knows nothing about
                ImageViewer, TIFFs, or dtypes; it only deals with curve
                shape and mouse interaction, and exposes it losslessly via
                get_control_points() (the raw (xs, ys) points) as well as
                a quick-and-dirty sample_lut() for its own on-screen use.

  CurvesDialog  A thin QDialog that owns a CurveWidget, computes a
                background histogram, and runs an edit "session": on
                open, it hides the MER overlay and captures a preview crop
                once; every curve change updates that crop-only preview
                live (apply_curve_preview); the full-resolution image is
                only touched when the user clicks "Apply and Close"
                (apply_curve_full) — "Cancel" (or closing the dialog any
                other way) instead reverts the display to whatever was
                last committed (restore_full_display) and discards the
                edit.

Curves are always edited in the *current 8-bit display domain* (0-255) —
i.e. against whatever the image looks like right now, after the min/max
sliders and any previously-applied curve — not against the raw 16-bit
pixel data. That's what makes curves composable, and correctly so: each
applied session is stored by ImageViewer as its *raw control points*
(see CurveWidget.get_control_points), never pre-sampled into a rounded
table, so composing several curves in a row never compounds 8-bit
rounding error — see ImageViewer.current_display_lut for why that
distinction matters. Reopening the dialog always starts from a fresh
identity curve — "no further change yet" on top of whatever's already
accumulated — since there's nothing meaningful to seed it from otherwise.
"Restore Original Scaling" clears that entire accumulated stack back to
identity (but leaves the min/max slider stretch untouched).

The curve applies identically to every channel (single LUT indexed against
R, G, and B alike) — there is no per-channel mode, by design.
"""

import bisect

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QPainterPath, QPen, QColor, QCursor
from PyQt5.QtWidgets import (
    QApplication, QDialog, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout,
)

from .curve_math import eval_curve


# ---------------------------------------------------------------------------
# CurveWidget
# ---------------------------------------------------------------------------

class CurveWidget(QWidget):
    """
    Interactive tone-curve editor.

    Points are stored as a list of QPointF in normalized [0,1] x [0,1]
    space (x = input fraction, y = output fraction; (0,0) = black point
    maps to black, (1,1) = white point maps to white). The list is always
    kept sorted by x with at least two points; dragging a point clamps its
    x between its immediate neighbours so points can never cross (which
    would make the curve ambiguous), but the two extreme points can still
    be dragged inward — exactly how GIMP's curve endpoints double as the
    black/white point controls.

    Signals
    -------
    dragStarted()
        A point grab (or a new-point insertion, or a double-click removal)
        has begun. Connect this to whatever needs to snapshot state before
        the edit (e.g. capturing a preview crop).
    curveChanged()
        The curve shape changed (during a drag, or after a reset/removal).
        Connect this to a fast preview update. Call get_control_points()
        or sample_lut() to read the current shape.
    dragFinished()
        The edit is complete (mouse released, or a reset/removal finished).
    pointReadout(float, float)
        Normalized (x, y) of the point currently being dragged or hovered,
        for an optional numeric display.
    """

    dragStarted  = pyqtSignal()
    curveChanged = pyqtSignal()
    dragFinished = pyqtSignal()
    pointReadout = pyqtSignal(float, float)

    MARGIN     = 24   # px inset between the widget edge and the plot area
    HIT_RADIUS = 9    # px — mouse must be within this of a point to grab it
    _EPS       = 1e-4  # minimum x-separation enforced between points

    def __init__(self, histogram: np.ndarray = None, parent=None):
        super().__init__(parent)
        self.points = [QPointF(0.0, 0.0), QPointF(1.0, 1.0)]
        self._histogram      = histogram
        self._dragging_index = None
        self._hover_index    = None

        self.setMinimumSize(320, 320)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Restore the default identity (no-op) curve."""
        self.points = [QPointF(0.0, 0.0), QPointF(1.0, 1.0)]
        self._dragging_index = None
        self._hover_index    = None
        self.update()
        self.dragStarted.emit()
        self.curveChanged.emit()
        self.dragFinished.emit()

    def get_control_points(self):
        """
        Return (xs, ys) as numpy float64 arrays, sorted by x — the raw
        control points of the current curve, exactly as drawn, with no
        sampling/rounding applied.

        This is the representation ImageViewer stores for each applied
        session (see current_display_lut in image_viewer.py): keeping the
        lossless control points, rather than a pre-sampled fixed-resolution
        table, is what lets repeated curve applications compose without
        compounding rounding error.
        """
        xs = np.array([p.x() for p in self.points], dtype=np.float64)
        ys = np.array([p.y() for p in self.points], dtype=np.float64)
        return xs, ys

    def sample_lut(self, dtype_max: int = 255, out_max: int = 255) -> np.ndarray:
        """
        Render the current curve as a (dtype_max + 1,)-entry uint8 lookup
        table — a convenience for callers that just want a quick table
        (e.g. this widget's own on-screen preview). ImageViewer does NOT
        use this for composing sessions together — see get_control_points.
        """
        xs, ys = self.get_control_points()
        input_norm = np.arange(dtype_max + 1, dtype=np.float64) / dtype_max
        out_norm   = np.clip(eval_curve(xs, ys, input_norm), 0.0, 1.0)
        return np.round(out_norm * out_max).astype(np.uint8)

    # ------------------------------------------------------------------
    # Internal geometry / spline helpers
    # ------------------------------------------------------------------

    def _plot_rect(self) -> QRectF:
        return QRectF(self.MARGIN, self.MARGIN,
                       max(1, self.width()  - 2 * self.MARGIN),
                       max(1, self.height() - 2 * self.MARGIN))

    def _to_widget(self, x_norm, y_norm, rect=None) -> QPointF:
        rect = rect or self._plot_rect()
        return QPointF(rect.left() + x_norm * rect.width(),
                        rect.bottom() - y_norm * rect.height())  # y flipped

    def _to_norm(self, px, py, rect=None):
        rect = rect or self._plot_rect()
        x_norm = (px - rect.left())   / rect.width()
        y_norm = (rect.bottom() - py) / rect.height()
        return max(0.0, min(1.0, x_norm)), max(0.0, min(1.0, y_norm))

    def _hit_test(self, pos):
        rect = self._plot_rect()
        r2 = self.HIT_RADIUS ** 2
        for i, p in enumerate(self.points):
            wp = self._to_widget(p.x(), p.y(), rect)
            dx, dy = wp.x() - pos.x(), wp.y() - pos.y()
            if dx * dx + dy * dy <= r2:
                return i
        return None

    def _insert_point(self, x_norm, y_norm) -> int:
        xs  = [p.x() for p in self.points]
        idx = bisect.bisect_left(xs, x_norm)
        if idx > 0:
            x_norm = max(x_norm, xs[idx - 1] + self._EPS)
        if idx < len(xs):
            x_norm = min(x_norm, xs[idx] - self._EPS)
        x_norm = max(0.0, min(1.0, x_norm))
        self.points.insert(idx, QPointF(x_norm, y_norm))
        return idx

    def _move_point(self, idx, x_norm, y_norm):
        lo = self.points[idx - 1].x() + self._EPS if idx > 0                     else 0.0
        hi = self.points[idx + 1].x() - self._EPS if idx < len(self.points) - 1  else 1.0
        x_norm = max(lo, min(hi, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        self.points[idx] = QPointF(x_norm, y_norm)

    def _emit_readout(self, idx):
        p = self.points[idx]
        self.pointReadout.emit(p.x(), p.y())

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        idx = self._hit_test(event.pos())
        if idx is None:
            x_norm, y_norm = self._to_norm(event.pos().x(), event.pos().y())
            idx = self._insert_point(x_norm, y_norm)
        self._dragging_index = idx
        self.dragStarted.emit()
        self._emit_readout(idx)
        self.update()

    def mouseMoveEvent(self, event):
        if self._dragging_index is not None:
            x_norm, y_norm = self._to_norm(event.pos().x(), event.pos().y())
            self._move_point(self._dragging_index, x_norm, y_norm)
            self._emit_readout(self._dragging_index)
            self.curveChanged.emit()
            self.update()
        else:
            hover = self._hit_test(event.pos())
            if hover != self._hover_index:
                self._hover_index = hover
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging_index is not None:
            self._dragging_index = None
            self.dragFinished.emit()
            self.update()

    def mouseDoubleClickEvent(self, event):
        """Double-click a control point to remove it (endpoints excepted —
        at least two points must always remain so the curve stays defined)."""
        idx = self._hit_test(event.pos())
        if idx is not None and len(self.points) > 2:
            del self.points[idx]
            self._dragging_index = None
            self.update()
            self.dragStarted.emit()
            self.curveChanged.emit()
            self.dragFinished.emit()

    def leaveEvent(self, event):
        if self._hover_index is not None:
            self._hover_index = None
            self.update()
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self._plot_rect()

        painter.fillRect(rect, QColor(250, 250, 250))

        if self._histogram is not None and self._histogram.size and self._histogram.max() > 0:
            bins = len(self._histogram)
            log_counts = np.log1p(self._histogram.astype(np.float64))
            max_log = log_counts.max() or 1.0
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(140, 150, 220, 120))
            bar_w = rect.width() / bins
            for i, v in enumerate(log_counts):
                bar_h = (v / max_log) * rect.height()
                painter.drawRect(QRectF(rect.left() + i * bar_w,
                                         rect.bottom() - bar_h,
                                         bar_w + 1, bar_h))

        painter.setPen(QPen(QColor(210, 210, 210), 1))
        for frac in (0.25, 0.5, 0.75):
            y = rect.top() + frac * rect.height()
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            x = rect.left() + frac * rect.width()
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))

        painter.setPen(QPen(QColor(180, 180, 180), 1, Qt.DashLine))
        painter.drawLine(self._to_widget(0, 0, rect), self._to_widget(1, 1, rect))

        xs, ys    = self.get_control_points()
        sample_x  = np.linspace(0.0, 1.0, 200)
        sample_y  = np.clip(eval_curve(xs, ys, sample_x), 0.0, 1.0)

        path = QPainterPath()
        path.moveTo(self._to_widget(sample_x[0], sample_y[0], rect))
        for x, y in zip(sample_x[1:], sample_y[1:]):
            path.lineTo(self._to_widget(x, y, rect))
        painter.setPen(QPen(QColor(30, 100, 200), 2))
        painter.drawPath(path)

        for i, p in enumerate(self.points):
            wp = self._to_widget(p.x(), p.y(), rect)
            if i == self._dragging_index:
                painter.setBrush(QColor(255, 140, 0))
                radius = 6
            elif i == self._hover_index:
                painter.setBrush(QColor(60, 140, 230))
                radius = 6
            else:
                painter.setBrush(QColor(30, 100, 200))
                radius = 5
            painter.setPen(QPen(Qt.white, 1.2))
            painter.drawEllipse(wp, radius, radius)

        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        painter.end()


# ---------------------------------------------------------------------------
# CurvesDialog
# ---------------------------------------------------------------------------

class CurvesDialog(QDialog):
    """
    Non-modal dialog wrapping a CurveWidget and wiring it to ImageViewer.

    Always edits a fresh identity curve (see module docstring for why),
    against the current 8-bit display domain (0-255), and runs an
    explicit edit "session" rather than committing on every drag release:

      - On open: hide the MER overlay (if shown), remember the current
        scroll position, and capture a preview crop once — mirroring
        ControlDock.slider_pressed / ImageViewer.capture_preview_crop, but
        done once for the whole dialog rather than per-drag, since a
        curve edit is typically many small drags across one sitting.
      - While editing: every curve change updates that crop-only preview
        live (apply_curve_preview) — cheap, so it can fire continuously.
      - "Apply and Close": appends this session's curve (its raw control
        points) to ImageViewer's curve_sessions and runs the full-
        resolution pass (apply_curve_full), then tears the session down
        (discard the crop, restore MER/scroll position) and closes.
      - "Restore Original Scaling": clears the *entire* accumulated
        curve stack (every previously-applied curve, not just this
        session's) back to identity — the min/max slider stretch is left
        alone. Like any other edit, this only takes effect on Apply —
        clicking it and then Cancel leaves previously-applied curves
        exactly as they were.
      - "Cancel", or closing the dialog any other way (window X, Esc, or
        the curvesPushButton being toggled off) reverts the display to
        whatever was last committed (restore_full_display) without ever
        touching original_image/qimage/last_pixmap/curve_sessions, then
        tears the session down identically.

    Follows the same "dialog holds a direct viewer reference" pattern as
    TableDialog/PlotDialog elsewhere in this codebase, rather than routing
    through ControlDock's callback-injection layer (which exists to keep
    ImageViewer decoupled from the *slider* widgets specifically).
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Curves")
        self._viewcenter              = None
        self._mer_was_visible         = False
        self._session_active          = False
        # Set by "Restore Original Scaling"; see current_display_lut() /
        # apply_curve_full()'s ignore_cumulative / clear_previous args.
        self._clear_cumulative_pending = False

        histogram = self._compute_histogram()
        self.curve_widget = CurveWidget(histogram=histogram)

        self.readout_label = QLabel(" ")
        self.readout_label.setAlignment(Qt.AlignCenter)

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Undo this session's not-yet-applied curve edits.")
        reset_btn.clicked.connect(self._on_reset)

        restore_btn = QPushButton("Restore Original Scaling")
        restore_btn.setToolTip(
            "Clear every previously-applied curve (not just this session's), "
            "back to the plain min/max slider stretch. Takes effect on Apply."
        )
        restore_btn.clicked.connect(self._on_restore_original)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel)
        apply_btn = QPushButton("Apply and Close")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._on_apply_and_close)

        view_row = QHBoxLayout()
        view_row.addWidget(reset_btn)
        view_row.addStretch()
        view_row.addWidget(restore_btn)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)

        hint = QLabel(
            "Drag the curve or its endpoints to set the black/white points "
            "and contrast. Double-click a point to remove it. Curves stack: "
            "each Apply adds to whatever was applied before. Changes are "
            "only committed to the full image on Apply."
        )
        hint.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addLayout(view_row)
        layout.addWidget(self.curve_widget)
        layout.addWidget(self.readout_label)
        layout.addLayout(btn_row)

        self.curve_widget.pointReadout.connect(self._on_point_readout)
        self.curve_widget.curveChanged.connect(self._on_curve_changed)

        self._begin_session()

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def _compute_histogram(self, bins: int = 256):
        """
        Histogram of the image *as currently displayed* (post min/max
        stretch and any already-applied curves), shown as a backdrop
        behind the curve — so its 0-255 x-axis lines up with the curve
        widget's own 0-255 domain regardless of what stretch/curves are
        already in effect, the same way GIMP's curves histogram reflects
        the image state at the point you opened the tool.

        Subsamples large tiles so this stays fast regardless of image size
        (a 20000x20000x3 array would otherwise mean a ~1.2e9-element pass).
        """
        img = self.viewer.original_image
        if img is None:
            return None
        h, w = img.shape[:2]
        target_pixels = 2_000_000
        stride = max(1, int(np.sqrt((h * w) / target_pixels)))
        sample = img[::stride, ::stride]
        displayed = self.viewer.current_display_lut()[sample]   # uint8, 0-255
        counts, _ = np.histogram(displayed.ravel(), bins=bins, range=(0, 255))
        return counts

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _begin_session(self):
        """
        Snapshot view state once for the whole dialog session (not per
        drag — hiding/showing the MER overlay on every single point drag
        would flicker it back on mid-edit the moment a second drag began,
        since toggle_MER() is a toggle, not an idempotent hide).
        """
        dock = getattr(self.viewer, 'control_dock', None)
        self._viewcenter = self.viewer.get_current_view_center()
        self._mer_was_visible = bool(dock is not None and dock.MER_PushButton.isChecked())
        if self._mer_was_visible:
            self.viewer.toggle_MER()
        self.viewer.capture_preview_crop()
        self._session_active = True

    def _end_session(self):
        """Common teardown for both Apply and Cancel: discard the preview
        crop and restore whatever _begin_session() changed."""
        if not self._session_active:
            return
        self.viewer.discard_preview_crop()
        if self._mer_was_visible:
            self.viewer.toggle_MER()
        if self._viewcenter is not None:
            self.viewer.restore_view_center(self._viewcenter)
        self._session_active = False

    def _on_apply_and_close(self):
        points = self.curve_widget.get_control_points()
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        QApplication.processEvents()
        self.viewer.apply_curve_full(points, clear_previous=self._clear_cumulative_pending)
        QApplication.restoreOverrideCursor()
        self._end_session()
        self.accept()

    def _on_cancel(self):
        self.viewer.restore_full_display()
        self._end_session()
        self.reject()

    def closeEvent(self, event):
        # Closing via the window's own X button, Esc, or the
        # curvesPushButton being toggled off all bypass the Apply/Cancel
        # buttons — treat all of them as Cancel, so an accidental close
        # never silently commits a half-drawn curve. accept()/reject()
        # (the two paths above) already ran _end_session() and cleared
        # _session_active, so this is a no-op when reached via either of
        # those buttons.
        if self._session_active:
            self.viewer.restore_full_display()
            self._end_session()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # CurveWidget signal handlers / button actions
    # ------------------------------------------------------------------

    def _on_curve_changed(self):
        points = self.curve_widget.get_control_points()
        self.viewer.apply_curve_preview(points, ignore_cumulative=self._clear_cumulative_pending)

    def _on_reset(self):
        """Undo this session's not-yet-applied curve only — leaves any
        previously-applied (already-committed) curves untouched."""
        self.curve_widget.reset()

    def _on_restore_original(self):
        """
        Clear the entire accumulated curve stack, not just this session's
        edit. Only takes effect when the user actually commits via Apply
        (see _clear_cumulative_pending / apply_curve_full's clear_previous)
        — clicking this and then Cancel leaves prior curves untouched.

        Sets the flag *before* calling reset(): CurveWidget.reset() emits
        curveChanged synchronously as part of resetting, which would
        otherwise trigger one preview update with the stale (pre-restore)
        flag value before this method got a chance to set it.
        """
        self._clear_cumulative_pending = True
        self.curve_widget.reset()

    def _on_point_readout(self, x_norm, y_norm):
        self.readout_label.setText(
            f"In: {int(round(x_norm * 255))}   Out: {int(round(y_norm * 255))}"
        )
