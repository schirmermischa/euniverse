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

import os
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QDialog, QPushButton, QToolButton, QComboBox, QLabel, QCheckBox, QFileDialog, QMessageBox, QLineEdit, QAction
from PyQt5.QtCore import Qt, QSize, pyqtSignal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from matplotlib.colors import LogNorm, to_rgba
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from .generate_icons import create_crosshair_icon, create_lasso_icon

class LassoSelectorTool(ToolToggleBase):
    """Custom Matplotlib tool for enabling/disabling the lasso selector."""
    default_keymap = 'L'  # Keyboard shortcut
    description = 'Lasso Selector'
    name = 'Lasso Selector'

    def __init__(self, plot_dialog, *args, **kwargs):
        super().__init__(None, name=self.name, *args, **kwargs)  # Pass name explicitly
        self.plot_dialog = plot_dialog  # Reference to PlotDialog instance

    def enable(self, *args):
        """Called when the lasso tool is activated."""
        self.plot_dialog.activate_lasso_selector()

    def disable(self, *args):
        """Called when the lasso tool is deactivated."""
        self.plot_dialog.deactivate_lasso_selector()

    def set_active(self, state):
        """Toggle the tool's active state."""
        if state:
            self.enable()
        else:
            self.disable()

            
class PlotDialog(QDialog):
    # New signal to emit the OBJECT_ID when a data point is clicked on the plot
    plot_point_clicked = pyqtSignal(int)
    lasso_points_selected = pyqtSignal(list)

    def __init__(self, catalog_manager, image_viewer, parent=None): # Added image_viewer
        super().__init__(parent)
        self.setWindowTitle("Catalog Plotter")
        self.catalog_manager = catalog_manager
        self.catalog = catalog_manager.catalog
        self.image_viewer = image_viewer # Store the image_viewer instance
        self.setGeometry(100, 100, 800, 600)

         # Variables to store plotted data and object IDs for interactivity
        self.plotted_x_data = None
        self.plotted_y_data = None
        self.plotted_object_ids = None
        self.artist = None # To store the scatter plot artist for hit testing

        self.original_toolbar_set_message = None
        self.original_toolbar_active_mode = None

        # Connection ID for the pick_event handler registered in activate_lasso_selector.
        # Kept here so deactivate_lasso_selector can disconnect it cleanly.
        self._pick_cid = None

        # Lasso selector variables
        self.lasso_selector = None
        self.select_action = None
        self.selected_object_ids = np.array([], dtype=int) # Store OBJECT_IDs of selected points
        self.selected_indices = np.array([], dtype=int) # Temporary indices for current plot
        self.original_colors = None  # Store original colors for resetting
        self._selection_overlay = None  # Overlay scatter for z-axis selection highlight
        self.lasso_tool = None  # Store the lasso tool instance
        
        # Load UI from .ui file
        from euniverse import get_resource
        ui_file = get_resource('catalog_plotter.ui')
        # ui_file = os.path.join(os.path.dirname(__file__), "catalog_plotter.ui")
        uic.loadUi(ui_file, self)

        # Set the background color for the plotWidget (overrides tooltip background)
        self.frame.setStyleSheet("background-color: #dddddd;")

        # Apply QSS for tooltips
        self.setStyleSheet("""
            QToolTip {
                color: black;
                background: #f0f0cc;
                border: 1px solid grey;
            }
        """)

        # Button stylesheet
        button_style = """
            QPushButton, QPushButton:checked, QPushButton:pressed {
                border-width: 0px !important;
                outline: none !important;
                padding: 0px !important;
                margin: 0px !important;
                background: none !important;
            }
            QPushButton:checked {
                background: #d0d0d0;
            }
            QPushButton:hover {
                background: #e0e0e0;
            }
        """

        # For conventional sky plots
        self.mirror_x_axis = False
        
        # Create a Matplotlib figure and canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Set up a layout for the plotWidget to embed the canvas
        plotWidget_layout = self.plotWidget.layout()
        if plotWidget_layout is None:
            plotWidget_layout = QVBoxLayout(self.plotWidget)
            plotWidget_layout.setContentsMargins(0, 0, 0, 0)
        plotWidget_layout.addWidget(self.canvas)

        # Add Navigation Toolbar
        self.add_navigation_toolbar()
        self.lasso_tool = LassoSelectorTool(plot_dialog=self)

        self.populate_comboboxes()

        # Set the default selected item for cmapComboBox to the first item
        self.zaxisComboBox.setCurrentIndex(0)
        self.xaxisComboBox.setCurrentIndex(self.xaxisComboBox.findText("RIGHT_ASCENSION"))
        self.yaxisComboBox.setCurrentIndex(self.yaxisComboBox.findText("DECLINATION"))

        self.xaxisComboBox.currentIndexChanged.connect(self.update_plot)
        self.yaxisComboBox.currentIndexChanged.connect(self.update_plot)
        self.zaxisComboBox.currentIndexChanged.connect(self.update_plot)
        self.cmapComboBox.currentIndexChanged.connect(self.update_plot)
        self.invertCmapCheckBox.stateChanged.connect(self.update_plot)
        self.xlogCheckBox.stateChanged.connect(self.update_plot)
        self.ylogCheckBox.stateChanged.connect(self.update_plot)
        self.zlogCheckBox.stateChanged.connect(self.update_plot)
        self.sizeSpinBox.valueChanged.connect(self.update_plot)
        self.redrawPushButton.clicked.connect(self.update_plot)
        self.equalAspectCheckBox.stateChanged.connect(self.update_plot)
        self.gridCheckBox.stateChanged.connect(self.update_plot)
        self.closePushButton.clicked.connect(self.accept)
        self.clearPushButton.clicked.connect(self.clear_plot)

        # Connect other buttons to untoggle lasso
        self.redrawPushButton.clicked.connect(lambda: self.untoggle_lasso_if_other_button_clicked(self.redrawPushButton))
        self.clearPushButton.clicked.connect(lambda: self.untoggle_lasso_if_other_button_clicked(self.clearPushButton))
        self.closePushButton.clicked.connect(lambda: self.untoggle_lasso_if_other_button_clicked(self.closePushButton))
        # Connect toolbar actions to untoggle lasso, excluding the lasso action itself
        for action in self.toolbar.actions():
            if action != self.select_action and action.isCheckable(): # Only connect checkable actions
                action.toggled.connect(lambda checked, a=action: self.untoggle_lasso_if_other_action_toggled(a, checked))
        
        self.update_plot()
        
    # ------------------------------------------------------------------
    # Selection overlay helpers (z-axis / colormap mode)
    # ------------------------------------------------------------------
    # When a z-axis colormap is active the main scatter artist stores a 1-D
    # array of scalar z-values, NOT RGBA data.  We cannot write RGBA tuples
    # into it.  Instead we maintain a *separate* overlay scatter (_selection_overlay)
    # that is drawn on top and contains only the selected points in red.
    #
    # _remove_selection_overlay  — removes and forgets the overlay
    # _apply_selection_overlay   — (re)creates it for the current selection
    #
    # All three selection-state methods (clear_lasso_selection,
    # on_lasso_select, activate_lasso_selector / deactivate_lasso_selector)
    # call these helpers so the behaviour is consistent.
    # ------------------------------------------------------------------

    def _remove_selection_overlay(self):
        """Remove the z-axis selection overlay scatter from the axes, if present."""
        if self._selection_overlay is not None:
            try:
                self._selection_overlay.remove()
            except ValueError:
                pass   # already detached (e.g. by figure.clear)
            self._selection_overlay = None

    def _apply_selection_overlay(self, x, y, symsize):
        """
        (Re)create the red overlay scatter for the currently selected points.

        Parameters
        ----------
        x, y    : full plotted data arrays (already filtered / masked)
        symsize : marker size used for the main scatter (overlay uses 2x)
        """
        self._remove_selection_overlay()
        if self.selected_indices.size == 0:
            return
        self._selection_overlay = self.ax.scatter(
            x[self.selected_indices],
            y[self.selected_indices],
            s=symsize * 2,        # slightly larger so selected points stand out
            c='tab:red',
            zorder=self.artist.get_zorder() + 1,  # always on top of the colormap layer
            label='_nolegend_',
        )

    def clear_lasso_selection(self):
        """Clears the lasso selection and resets point colors."""
        self.selected_object_ids = np.array([], dtype=int)
        self.selected_indices = np.array([], dtype=int)

        # Reset point colours.
        # z-axis active: the colormap scalar array is already correct — just
        # remove the red overlay scatter.
        # No z-axis: restore every point to the default RGBA colour.
        if self.artist is not None and self.plotted_x_data is not None:
            if self.zaxisComboBox.currentText():
                self._remove_selection_overlay()
            else:
                n_points = len(self.plotted_x_data)
                default_color = (self.artist.get_facecolors()[0]
                                 if len(self.artist.get_facecolors()) > 0
                                 else to_rgba('tab:blue'))
                colors = np.ones((n_points, 4))
                colors[:] = default_color
                self.artist.set_facecolors(colors)
                self.original_colors = colors.copy()
            self.canvas.draw()

        # Ensure canvas is interactive
        self.canvas.setFocus()
        self.canvas.draw_idle()

        # Clear any overlays in image_viewer
        if self.image_viewer:
            self.catalog_manager.clear_selected_MER()
            self.image_viewer.update_status("Selection cleared.", 2000)

        
    def untoggle_lasso_if_other_button_clicked(self, clicked_button):
        """Untoggles the lasso button if a different button is clicked."""
        if self.select_action.isChecked() and clicked_button.objectName() != self.select_action.objectName():
            self.select_action.setChecked(False)

    def untoggle_lasso_if_other_action_toggled(self, toggled_action, checked):
        """Untoggles the lasso button if a different toolbar action is toggled on."""
        # Only untoggle lasso if another action is being checked (activated)
        if checked and self.select_action.isChecked() and toggled_action != self.select_action:
            self.select_action.setChecked(False)
            
    def clear_plot(self):
        """
        Initializes an empty Matplotlib plot ranging from 0 to 1 on both axes.
        This is called initially and when invalid data selections are made.
        """
        # Deactivate the lasso before clearing the figure so LassoSelector
        # never holds a reference to the about-to-be-destroyed axes.
        # We do this unconditionally; deactivate_lasso_selector is a no-op
        # when no lasso is active.
        if self.lasso_selector is not None:
            self.deactivate_lasso_selector()
            # Also uncheck the toolbar button without re-entering the toggle chain
            self.select_action.blockSignals(True)
            self.select_action.setChecked(False)
            self.select_action.blockSignals(False)
        # Null the overlay reference — figure.clear() destroys it silently
        self._selection_overlay = None
        self.figure.clear() # Clear the entire figure
        self.ax = self.figure.add_subplot(111) # Add a new subplot
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.canvas.draw() # Redraw the canvas to show the changes

    def populate_comboboxes(self):
        """Populate the x, y, and z axis combo boxes with column names and colormaps."""
        # Populate cmapComboBox with common colormaps first, as it doesn't depend on catalog
        base_colormaps = ['viridis', 'plasma', 'inferno', 'coolwarm', 'magma', 'cividis',
                          'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuBuGn', 'PuBu', 'BuGn', 'GnBu', 'PuRd',
                          'RdPu', 'BuPu', 'YlGnBu', 'YlGn']
        for cmap in base_colormaps:
            self.cmapComboBox.addItem(cmap)

        if self.catalog_manager.catalog is not None:
            column_names = self.catalog_manager.get_all_column_names()
            for col in column_names:
                self.xaxisComboBox.addItem(col)
                self.yaxisComboBox.addItem(col)
                self.zaxisComboBox.addItem(col)
            self.sort_combobox(self.xaxisComboBox)
            self.sort_combobox(self.yaxisComboBox)
            self.sort_combobox(self.zaxisComboBox)
            self.zaxisComboBox.insertItem(0, "")  # needed if no color-coding is desired.
            self.xaxisComboBox.adjustSize()
            self.yaxisComboBox.adjustSize()
            self.zaxisComboBox.adjustSize()
            self.xaxisComboBox.setMaxVisibleItems(20)
            self.yaxisComboBox.setMaxVisibleItems(20)
            self.zaxisComboBox.setMaxVisibleItems(20)
        else:
            print("Catalog is not loaded. Combo boxes not populated.")

            
    def update_plot(self):
        """
        Updates the plot based on the current selections in the x, y, z axis, and colormap combo boxes.
        Applies log scaling to axes and colorbar if the respective checkboxes are checked.
        Preserves lasso selections using OBJECT_IDs.
        """
        # Do NOT reset lasso selector unless the plot is invalid
        # Note: figure.clear() and add_subplot are handled inside make_scatterplot,
        # so we must not call them here too (that would discard any state set before the call).
        x_label = self.xaxisComboBox.currentText() if self.xaxisComboBox else ""
        y_label = self.yaxisComboBox.currentText() if self.yaxisComboBox else ""
        z_label = self.zaxisComboBox.currentText() if self.zaxisComboBox else ""
        colormap_name = self.cmapComboBox.currentText() if self.cmapComboBox else "viridis"

        # Apply inversion if checkbox is checked
        if self.invertCmapCheckBox.isChecked():
            colormap_name += '_r'

        x_data = None
        y_data = None
        z_data = None
        object_ids = None

        # Retrieve data
        if x_label and self.catalog and x_label in self.catalog.columns:
            x_data = np.array(self.catalog[x_label].data)
        else:
            self.clear_plot()
            self.selected_indices = np.array([], dtype=int)
            if self.lasso_selector is not None:
                self.deactivate_lasso_selector()
                self.select_action.setChecked(False)
            return

        if y_label and self.catalog and y_label in self.catalog.columns:
            y_data = np.array(self.catalog[y_label].data)
        else:
            self.clear_plot()
            self.selected_indices = np.array([], dtype=int)
            if self.lasso_selector is not None:
                self.deactivate_lasso_selector()
                self.select_action.setChecked(False)
            return

        if 'OBJECT_ID' in self.catalog.columns:
            object_ids = np.array(self.catalog['OBJECT_ID'].data)
        else:
            object_ids = None
            print("Warning: 'OBJECT_ID' column not found. Selections may not persist.")

        # Store original data before filtering
        self.plotted_x_data = x_data.copy()
        self.plotted_y_data = y_data.copy()
        self.plotted_object_ids = object_ids.copy() if object_ids is not None else None

        # Apply log scale filters
        combined_mask = np.ones(len(x_data), dtype=bool)
        if self.xlogCheckBox.isChecked():
            positive_x_mask = x_data > 0
            combined_mask &= positive_x_mask
            self.ax.set_xscale('log')
        if self.ylogCheckBox.isChecked():
            positive_y_mask = y_data > 0
            combined_mask &= positive_y_mask
            self.ax.set_yscale('log')

        # Apply mask to data
        x_data = x_data[combined_mask]
        y_data = y_data[combined_mask]
        if object_ids is not None:
            object_ids = object_ids[combined_mask]

        # Update plotted data after filtering
        self.plotted_x_data = x_data
        self.plotted_y_data = y_data
        self.plotted_object_ids = object_ids

        # Map selected OBJECT_IDs to current indices
        self.selected_indices = np.array([], dtype=int)
        if self.selected_object_ids.size > 0 and object_ids is not None:
            self.selected_indices = np.where(np.isin(object_ids, self.selected_object_ids))[0]

        # Symbol size
        symsize = self.sizeSpinBox.value()

        norm_param = None
        # Retrieve z-axis data for color-coding
        if z_label and z_label != "":
            z_data = np.array(self.catalog[z_label].data)
            z_data = z_data[combined_mask]

            if self.zlogCheckBox.isChecked():
                positive_z_data = z_data[z_data > 0]
                if positive_z_data.size > 0:
                    vmin_log = positive_z_data.min()
                    vmax_log = z_data.max()
                    norm_param = LogNorm(vmin=vmin_log, vmax=vmax_log)
                else:
                    print("Warning: Z-axis contains no positive values. Using linear scale.")

            # Create scatter plot with color coding
            self.make_scatterplot(x_data, y_data, z_data, colormap_name, symsize, norm_param, x_label, y_label, z_label)
            # self.artist = self.ax.scatter(x_data, y_data, c=z_data, cmap=cmap_name, s=symsize, picker=True, pickradius=3, norm=norm_param)
            self.figure.colorbar(self.artist, ax=self.ax, label=z_label)

            # Reapply the red selection overlay using the shared helper.
            # _apply_selection_overlay handles the remove-then-recreate cycle
            # so the colormap scalar array is never mutated.
            self._apply_selection_overlay(x_data, y_data, symsize)
        else:
            # No z-axis, regular scatter plot
            self.make_scatterplot(x_data, y_data, np.empty(0, dtype=float), colormap_name, symsize, norm_param,
                                  x_label, y_label, z_label)
            # Reapply selection colors
            if self.selected_indices.size > 0:
                n_points = len(x_data)
                colors = np.ones((n_points, 4))
                default_color = self.artist.get_facecolors()[0] if len(self.artist.get_facecolors()) > 0 else to_rgba('tab:blue')
                colors[:] = default_color
                colors[self.selected_indices] = to_rgba('tab:red')
                colors[np.setdiff1d(np.arange(n_points), self.selected_indices)] = to_rgba('tab:blue')
                self.artist.set_facecolors(colors)
                self.original_colors = colors.copy()

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(self.catalog_manager.catalog_name)
        if self.xlogCheckBox.isChecked():
            self.ax.set_xscale('log')
        if self.ylogCheckBox.isChecked():
            self.ax.set_yscale('log')
#        self.ax.autoscale_view()
        if self.gridCheckBox.isChecked():
            self.ax.grid(True, color='gray', linestyle='--')
        if self.equalAspectCheckBox.isChecked():
            self.ax.set_aspect('equal')

#        if self.mirror_x_axis:
#            self.ax.set_xlim(self.ax.get_xlim()[1], self.ax.get_xlim()[0])
            
        # Set font sizes
        self.canvas.draw()
        self.canvas.flush_events()
        try:
            label_fontsize = self.ax.xaxis.label.get_fontsize()
        except AttributeError:
            label_fontsize = plt.rcParams['font.size']
        self.ax.set_xlabel(x_label, fontsize=label_fontsize)
        self.ax.set_ylabel(y_label, fontsize=label_fontsize)
        self.ax.set_title(self.catalog_manager.catalog_name, fontsize=label_fontsize)
        self.figure.tight_layout()
        self.canvas.draw()

    def sort_combobox(self, combobox: QComboBox):
        """
        Sorts the items in a QComboBox alphabetically.
        Args:
        combobox (QComboBox): The QComboBox widget to sort.
        """
        items = []
        for i in range(combobox.count()):
            items.append(combobox.itemText(i))
            
        items.sort() # Alphabetic sorting
        combobox.clear()
        
        for item in items:
            combobox.addItem(item) # Add sorted items back


    def make_scatterplot(self, x_data, y_data, z_data, cmap_name, symsize, norm_param, x_label, y_label, z_label):
        # figure.clear() destroys all axes and every artist on them, including
        # _selection_overlay, without notifying Python.  Null it here so
        # _remove_selection_overlay() never calls .remove() on a dead artist.
        self._selection_overlay = None
        self.figure.clear()
        
        # Sky scatter plot:
        # first section never evaluated because of deliberatly chosing non-existing dec keyword
        # (lasso selection does not work in WCS mode)
        if x_label == "RIGHT_ASCENSION" and y_label == "DECLINATION2":
            # Convert RA, Dec to SkyCoord object
            coords = SkyCoord(ra=x_data*u.degree, dec=y_data*u.degree, frame='icrs')

            wcs = WCS(naxis=2)
            wcs.wcs.crval = [coords.ra.mean().value, coords.dec.mean().value]
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            
            # 2. Create WCS Axes for tangential (gnomonic) projection
            self.ax = self.figure.add_subplot(111, projection=wcs)

            # Prepare the sources
            ra_deg = coords.ra.deg
            dec_deg = coords.dec.deg
            x_data, y_data = wcs.wcs_world2pix(ra_deg, dec_deg, 0)
            # Plot the data. When projection is set, scatter expects World coordinates
            if z_label and z_label != "":
                self.artist = self.ax.scatter(coords.ra.deg, coords.dec.deg, c=z_data, cmap=cmap_name,
                                              transform=self.ax.get_transform('world'), s=symsize,
                                              picker=True, pickradius=3, norm=norm_param)
            else:
                self.artist = self.ax.scatter(coords.ra.deg, coords.dec.deg, 
                                              transform=self.ax.get_transform('world'), s=symsize,
                                              picker=True, pickradius=3)
        else:
            self.ax = self.figure.add_subplot(111)
            if x_label == "RIGHT_ASCENSION" and y_label == "DECLINATION2":
                self.mirror_x_axis = True
            else:
                self.mirror_x_axis = False
            if z_label and z_label != "":
                self.artist = self.ax.scatter(x_data, y_data, c=z_data, cmap=cmap_name, s=symsize, picker=True,
                                              pickradius=3, norm=norm_param)
            else:
                self.artist = self.ax.scatter(x_data, y_data, s=symsize, picker=True, pickradius=3)

        # Note: the active LassoSelector is managed separately by activate/deactivate_lasso_selector.
        # Do NOT create one here — every redraw would leak an additional connected selector.
        self.artist.set_alpha(0.3)
        self.canvas.draw()
        
    def highlight_selected_points(self, indices):
        # Reset previous selection alpha
        self.artist.set_alpha(0.3)
        
        # Here we create or update a 'selection' scatter overlay
        # or simply print the count for now
        self.image_viewer.update_status(f"Selected {len(indices)} points.", 6000)
        
        # Update the parent table if needed
        if self.image_viewer and self.image_viewer.main_window:
            # Example: Select the first index in the table
            obj_id = self.catalog['OBJECT_ID'][indices[0]]
            self.image_viewer.control_dock.select_table_row(obj_id)
            
        self.canvas.draw_idle()

    # overriding class definition
    def on_select(self, verts):
        # Prevent error if scatter hasn't been created yet
        if not hasattr(self, 'artist') or self.artist is None:
            return

        path = Path(verts)
        # Because we use the 'world' transform in make_scatterplot,
        # get_offsets() returns degrees, and verts are in degrees for RA/Dec plots
        self.indices = np.nonzero(path.contains_points(self.artist.get_offsets()))[0]
        
        if self.indices.size > 0:
            # Use the existing method in your PlotDialog class
            self.highlight_selected_points(self.indices)
        else:
            # Reset view if nothing selected
            self.artist.set_alpha(0.7)
            self.canvas.draw_idle()

    def on_pick(self, event):
        """
        Single-click on a data point: treat it exactly like a one-point lasso selection.

        event.ind contains the indices (into plotted_x/y_data) of every point whose
        pick radius overlaps the click.  We take the first one and feed it through
        the same _select_indices() path as the lasso so all highlighting, overlay,
        image-viewer centering, and signal emission happen identically.

        The pick_event connection is registered in activate_lasso_selector and
        removed in deactivate_lasso_selector, so this handler is only active
        while lasso mode is on.
        """
        if event.artist != self.artist:
            return
        if event.ind is None or len(event.ind) == 0:
            return
        # Take the closest point when multiple overlap
        data_index = int(event.ind[0])
        self._select_indices(np.array([data_index]))

    def activate_lasso_selector(self):
        """
        Enter lasso selection mode.

        Deactivates any currently-running lasso first (guard against double
        activation).  Then:
          - silently unchecks the pan / zoom toolbar buttons (blockSignals so
            their toggled chains don't fire)
          - creates a LassoSelector bound to the current axes
          - connects pick_event so single clicks work identically to a
            one-point lasso (see on_pick / _select_indices)
          - reapplies any pre-existing selection highlight
        """
        if self.lasso_selector is not None:
            self.deactivate_lasso_selector()

        if self.artist is None or self.plotted_x_data is None or self.plotted_y_data is None:
            # Can't activate with no data — silently back out
            self.lasso_tool.set_active(False)
            self.select_action.blockSignals(True)
            self.select_action.setChecked(False)
            self.select_action.blockSignals(False)
            if self.image_viewer:
                self.image_viewer.update_status("No valid plot data for lasso.", 2000)
            return

        # Silently uncheck pan / zoom — blockSignals prevents their toggled
        # chains from firing and potentially re-entering this method.
        for name in ('pan', 'zoom'):
            if name in self.toolbar._actions:
                action = self.toolbar._actions[name]
                if action.isChecked():
                    action.blockSignals(True)
                    action.setChecked(False)
                    action.blockSignals(False)
                    # Call the toolbar's own pan()/zoom() to update internal state
                    getattr(self.toolbar, name)()

        self.original_toolbar_active_mode = self.toolbar.mode
        self.toolbar.mode = ''

        # Snapshot the base colours so deactivate can restore them
        if self.zaxisComboBox.currentText():
            self.original_colors = self.artist.get_array().copy()
            self.artist._original_array = self.original_colors.copy()
        else:
            n_points = len(self.plotted_x_data)
            default_color = (self.artist.get_facecolors()[0]
                             if len(self.artist.get_facecolors()) > 0
                             else to_rgba('tab:blue'))
            self.original_colors = np.tile(default_color, (n_points, 1))

        # Reapply the existing selection highlight (if any) on the fresh axes
        self.selected_indices = np.where(
            np.isin(self.plotted_object_ids, self.selected_object_ids)
        )[0]
        if self.selected_indices.size > 0:
            symsize = self.sizeSpinBox.value()
            if self.zaxisComboBox.currentText():
                self._apply_selection_overlay(self.plotted_x_data, self.plotted_y_data, symsize)
            else:
                n_points = len(self.plotted_x_data)
                colors = np.ones((n_points, 4))
                colors[:] = default_color
                colors[self.selected_indices] = to_rgba('tab:red')
                colors[np.setdiff1d(np.arange(n_points), self.selected_indices)] = to_rgba('tab:blue')
                self.artist.set_facecolors(colors)
                self.original_colors = colors.copy()
            self.canvas.draw()

        # LassoSelector manages its own button_press_event connection internally.
        # We must NOT add a second one with mpl_connect (double-fire bug).
        self.lasso_selector = LassoSelector(
            self.ax,
            onselect=self.on_lasso_select,
            useblit=True,
            props=dict(color='black', linewidth=1),
            button=1,
        )

        # With useblit=True the LassoSelector needs a valid background snapshot
        # before the first press.  If background is still None when the first
        # motion event fires, update() calls canvas.draw() itself — a full
        # synchronous redraw of the scatter plot that causes the 2-second freeze
        # and the straight-line artefact.
        #
        # canvas.draw() alone is not enough: in Qt, draw() schedules a repaint
        # via the event loop, so copy_from_bbox (called synchronously inside
        # update_background) may read an empty framebuffer.
        # The correct sequence:
        #   1. canvas.draw()          — render into the off-screen buffer
        #   2. processEvents()        — let Qt flush the paint pipeline so the
        #                               framebuffer is valid before we copy it
        #   3. update_background(None)— capture the now-valid buffer cheaply
        #                               (visible=False → needs_redraw=False →
        #                               just copy_from_bbox, no second draw())
        from PyQt5.QtWidgets import QApplication
        self.canvas.draw()
        QApplication.processEvents()
        self.lasso_selector.update_background(None)

        # Connect pick_event for single-click selection.
        # Stored in _pick_cid so deactivate can remove it cleanly.
        self._pick_cid = self.canvas.mpl_connect('pick_event', self.on_pick)

        self.toolbar.set_message = lambda x: None
        self.setCursor(Qt.CrossCursor)
        self.canvas.setFocus()
        if self.image_viewer:
            self.image_viewer.update_status(
                "Lasso/click selection mode: draw a lasso or click a point.", 3000
            )


    def deactivate_lasso_selector(self):
        """
        Exit lasso selection mode and restore the toolbar to its default state.

        Cleans up in this order:
          1. Disconnect pick_event (registered in activate_lasso_selector)
          2. Deactivate and discard the LassoSelector widget
          3. Remove or restore the selection highlight
          4. Restore toolbar message handler and cursor
        """
        try:
            # 1. Disconnect pick_event — always do this first so no stale
            #    callbacks fire while we tear down the lasso below.
            if hasattr(self, '_pick_cid') and self._pick_cid is not None:
                self.canvas.mpl_disconnect(self._pick_cid)
                self._pick_cid = None

            # 2. Discard LassoSelector — set_active(False) disconnects its
            #    own internal button_press_event handler.
            if self.lasso_selector is not None:
                self.lasso_selector.set_active(False)
                self.lasso_selector = None

            # 3. Restore point colours.
            #    z-axis active: colormap scalar array was never modified;
            #      only the overlay scatter needs removing.
            #    No z-axis: restore the saved RGBA facecolour snapshot.
            if self.artist is not None:
                if self.zaxisComboBox.currentText():
                    self._remove_selection_overlay()
                elif self.original_colors is not None:
                    self.artist.set_facecolors(self.original_colors)
                if self.canvas:
                    self.canvas.draw_idle()

            self.selected_indices = np.array([], dtype=int)
            self.original_colors  = None

            # 4. Restore toolbar.
            self.toolbar.set_message = self.original_toolbar_set_message
            self.toolbar.mode = ''
            # Use blockSignals to uncheck pan/zoom without firing their toggled
            # chains (direct emit would re-enter signal handlers).
            for name in ('pan', 'zoom'):
                if name in self.toolbar._actions:
                    action = self.toolbar._actions[name]
                    if action.isChecked():
                        action.blockSignals(True)
                        action.setChecked(False)
                        action.blockSignals(False)

            self.setCursor(Qt.ArrowCursor)
            if self.canvas:
                self.canvas.setFocus()
                self.canvas.draw_idle()

            if self.image_viewer:
                self.image_viewer.update_status("Lasso selection disabled.", 2000)

        except Exception as e:
            print(f"Error during lasso deactivation: {e}")

    def on_lasso_select(self, verts):
        """Convert lasso vertices to point indices and delegate to _select_indices."""
        if self.plotted_x_data is None or self.plotted_y_data is None or self.artist is None:
            return
        path   = Path(verts)
        points = np.column_stack([self.plotted_x_data, self.plotted_y_data])
        ind    = np.nonzero(path.contains_points(points))[0]
        self._select_indices(ind)

    def _select_indices(self, ind: np.ndarray):
        """
        Core selection handler — shared by on_lasso_select and on_pick.

        Adds the newly selected point indices to the persistent selection,
        updates highlight colours / overlay, emits lasso_points_selected,
        and tells the image viewer to centre on the selected sources.

        Parameters
        ----------
        ind : np.ndarray of int
            Indices into self.plotted_x/y_data of the newly selected points.
            May be empty (clears the visual state but keeps existing selection).
        """
        try:
            # Accumulate selected OBJECT_IDs across multiple lasso strokes
            if self.plotted_object_ids is not None and ind.size > 0:
                new_ids = self.plotted_object_ids[ind]
                self.selected_object_ids = np.union1d(self.selected_object_ids, new_ids)
                self.selected_indices = np.where(
                    np.isin(self.plotted_object_ids, self.selected_object_ids)
                )[0].flatten()
                self.lasso_points_selected.emit(self.selected_object_ids.tolist())
            else:
                # No OBJECT_ID column — fall back to index-based accumulation
                self.selected_indices = np.union1d(self.selected_indices, ind).flatten()
                self.lasso_points_selected.emit([])

            # Update colours to reflect the new selection.
            # z-axis active: colormap scalar array is untouched; the red overlay
            #   scatter is rebuilt for the current selected_indices.
            # No z-axis: write RGBA colours directly into the artist.
            symsize = self.sizeSpinBox.value()
            if self.zaxisComboBox.currentText():
                self._apply_selection_overlay(self.plotted_x_data, self.plotted_y_data, symsize)
            else:
                n_points = len(self.plotted_x_data)
                default_color = (self.artist.get_facecolors()[0]
                                 if len(self.artist.get_facecolors()) > 0
                                 else to_rgba('tab:blue'))
                colors = np.tile(default_color, (n_points, 1))
                colors[self.selected_indices] = to_rgba('tab:red')
                mask = np.ones(n_points, dtype=bool)
                mask[self.selected_indices] = False
                colors[mask] = to_rgba('tab:blue')
                self.artist.set_facecolors(colors)
                self.original_colors = colors.copy()
            self.canvas.draw_idle()  # draw_idle is safe during rapid lasso dragging

            if self.image_viewer and self.selected_indices.size > 0 and self.plotted_object_ids is not None:
                selected_ids = self.plotted_object_ids[self.selected_indices]
                self.image_viewer.display_selected_MER(selected_ids.tolist())
                self.image_viewer.update_status(f"Selected {len(selected_ids)} objects.", 2000)

        except ValueError as ve:
            print(f"Selection error (shape mismatch): {ve}")
        except Exception as e:
            print(f"Error in _select_indices: {e}")


    def closeEvent(self, event):
        """Ensure the Matplotlib figure is closed so it doesn't leak into the global figure manager."""
        if self.lasso_selector is not None:
            self.deactivate_lasso_selector()
        plt.close(self.figure)
        super().closeEvent(event)

    def add_navigation_toolbar(self):
        # --- Add Matplotlib Navigation Toolbar ---
        self.toolbar = NavigationToolbar(self.canvas, self.plotWidget)
        self.plotWidget.layout().addWidget(self.toolbar)  # Add to layout
        self.original_toolbar_set_message = self.toolbar.set_message  # Store original

        # Create "Select" toggle button for lasso
        self.select_action = QAction("Select", self.toolbar)
        self.select_action.setCheckable(True)
        self.select_action.setToolTip("Toggle lasso selection mode")
        self.toolbar.addAction(self.select_action)
        self.select_action.toggled.connect(lambda checked: self.lasso_tool.set_active(checked))

        # Create "Deselect" button to clear selection
        self.deselect_action = QAction("Deselect", self.toolbar)
        self.deselect_action.setToolTip("Clear current lasso selection")
        self.toolbar.addAction(self.deselect_action)
        self.deselect_action.triggered.connect(self.clear_lasso_selection)

