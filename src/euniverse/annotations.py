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
annotations.py — Data model for user-created object annotations
================================================================
Defines the Annotation dataclass that replaces the raw 5-tuple formerly
stored in ImageViewer.circles:

    Old:  (QGraphicsEllipseItem, ra, dec, classifier, normal_thickness)
    New:  Annotation(item, ra, dec, classifier, normal_thickness)

Using a dataclass makes the fields named and self-documenting, eliminates
positional unpacking errors, and makes it easy to add new fields (e.g. a
note, a confidence flag) without touching every call site.

The classifier string uses the category prefixes understood by the rest of
the application:
    "GL: ..."   — gravitational lens candidate
    "AGN: ..."  — active galactic nucleus
    "Gx: ..."   — galaxy feature (ring, merger, stream, …)
"""

from dataclasses import dataclass, field
from PyQt5.QtWidgets import QGraphicsEllipseItem


@dataclass
class Annotation:
    """
    A single user-created annotation circle drawn on the image.

    Attributes
    ----------
    item : QGraphicsEllipseItem
        The live Qt scene item.  May become a dangling C++ reference if the
        scene is cleared without first calling scene.removeItem(item).
        Always guard access with ``sip.isdeleted(ann.item)`` before use.
    ra : float
        Right ascension of the annotated object in decimal degrees (ICRS).
    dec : float
        Declination of the annotated object in decimal degrees (ICRS).
    classifier : str
        Human-readable label, e.g. "GL: lens", "AGN: Seyfert 1", "Gx: Ring".
    normal_thickness : float
        Pen width used when the circle is *not* selected.  The selected state
        uses ``normal_thickness * 1.5`` so the highlight is always proportional.
    """

    item:             QGraphicsEllipseItem
    ra:               float
    dec:              float
    classifier:       str
    normal_thickness: float = field(default=2.0)

    # ------------------------------------------------------------------
    # Convenience helpers used at multiple call sites
    # ------------------------------------------------------------------

    @property
    def category(self) -> str:
        """
        Returns the broad category prefix ('GL', 'AGN', or 'Gx'), derived
        from the classifier string.  Returns '' if the format is unexpected.

        Example
        -------
        >>> ann.classifier
        'GL: Einstein ring'
        >>> ann.category
        'GL'
        """
        return self.classifier.split(":")[0].strip() if ":" in self.classifier else ""
