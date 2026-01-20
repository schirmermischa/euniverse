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

        # List to keep track of Matplotlib event connections so they can be disconnected
        self.mpl_connections = []
        self.original_toolbar_set_message = None
        self.original_toolbar_active_mode = None # Store the original toolbar active mode

        # Store IDs for toolbar's original event connections
        # Initialize them to None, they will be captured after the toolbar is created
        self._toolbar_press_cid = None
        self._toolbar_release_cid = None
        self._toolbar_motion_cid = None

        # Lasso selector variables
        self.lasso_selector = None
        self.select_action = None
        self.selected_object_ids = np.array([], dtype=int) # Store OBJECT_IDs of selected points
        self.selected_indices = np.array([], dtype=int) # Temporary indices for current plot
        self.original_colors = None  # Store original colors for resetting
        self.lasso_tool = None  # Store the lasso tool instance
        self._lasso_connection_id = None # Connection ID for the lasso selector's event
        
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
        
    def clear_lasso_selection(self):
        """Clears the lasso selection and resets point colors."""
        self.selected_object_ids = np.array([], dtype=int)
        self.selected_indices = np.array([], dtype=int)

        # Reset point colors
        if self.artist is not None and self.plotted_x_data is not None:
            if self.zaxisComboBox.currentText():
                # Restore original colormap values
                if hasattr(self.artist, '_original_array'):
                    colors = self.artist._original_array.copy()
                else:
                    colors = self.artist.get_array().copy()
                    self.artist._original_array = colors  # Store for future resets
                self.artist.set_array(colors)
            else:
                # Restore default color for no colormap
                n_points = len(self.plotted_x_data)
                colors = np.ones((n_points, 4))
                default_color = self.artist.get_facecolors()[0] if len(self.artist.get_facecolors()) > 0 else to_rgba('tab:blue')
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
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)  # Add a new subplot

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

            # Reapply selection colors
            if self.selected_indices.size > 0:
                colors = self.artist.get_array().copy()
#                colors[self.selected_indices] = np.max(colors)
                colors[self.selected_indices] =  to_rgba('tab:red')
                self.artist.set_array(colors)
                self.original_colors = colors.copy()
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

        # Initialize Lasso with the axes' display transformation
        # Lasso normally selects in data space, but when we plot RA/DEC with WCS transformation, then this fails.
        # hence we must select in display space
        self.lasso = LassoSelector(self.ax, onselect=self.on_select,
                                   props={'color': 'black', 'linewidth': 1, 'alpha': 0.8})
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

    def on_select_old(self, verts):
        # 1. Path is created from lasso vertices (Display/Pixel space)
        path = Path(verts)
        
        # 2. Get the raw numeric offsets from the scatter plot
        xy = self.scatter.get_offsets()
        
        # 3. Transform the data points into Display space (pixels)
        # This handles both linear axes and WCS projections automatically
        trans = self.scatter.get_transform()
        pixel_coords = trans.transform(xy)
        
        # 4. Perform the selection check
        self.indices = np.nonzero(path.contains_points(pixel_coords))[0]
        
        if self.indices.size > 0:
            # Instead of a missing method, use the logic to highlight points
            self.highlight_selected_points(self.indices)

    def on_select_old(self, verts):
        # 1. Get the path from the lasso (these are in Data coordinates)
        path = Path(verts)
        
        # 2. Get the raw data points from the scatter plot
        # These will be RA/Dec if using WCS, or X/Y if linear
        xy = self.artist.get_offsets()
        
        # 3. TRANSFORM the data points into the same space as the Lasso path
        # If it's a WCSAxes, we use its specific transformation logic
        trans = self.artist.get_transform()
        # This converts the catalog coordinates into the visual coordinates 
        # that match the 'verts' from the mouse movement.
        pixel_coords = trans.transform(xy)
        
        # 4. Perform the selection check in that transformed space
        self.indices = np.nonzero(path.contains_points(pixel_coords))[0]
        
        if self.indices.size > 0:
            print(f"Selected {len(self.indices)} sources.")
            # Trigger your highlighting/table selection logic here
            self.update_plots_selection()

    def on_analyse_plot_toggled(self, checked):
        """
        Toggles interactive mode for selecting data points on the plot.
        Disables/enables the standard Matplotlib navigation toolbar by
        disconnecting/reconnecting its event handlers.
        """
        if checked:
            # Deactivate Lasso if it's active
            if self.lasso_tool.get_active():
                self.lasso_tool.set_active(False)

            # Disconnect toolbar's default event handlers if they exist
            if self._toolbar_press_cid is not None:
                self.canvas.mpl_disconnect(self._toolbar_press_cid)
                # We don't set to None here, as we might need to reconnect it later
                # and the _toolbar_press_cid will still hold the original ID.
            if self._toolbar_release_cid is not None:
                self.canvas.mpl_disconnect(self._toolbar_release_cid)
            if self._toolbar_motion_cid is not None:
                self.canvas.mpl_disconnect(self._toolbar_motion_cid)

            # Clear toolbar mode and message
            self.original_toolbar_active_mode = self.toolbar.mode
            self.toolbar.mode = '' # Deactivate any active mode (e.g., pan, zoom)
            self.toolbar.set_message = lambda x: None # Disable status messages
            
            # Disconnect any previously connected custom pick event handlers to avoid duplicates
            for cid in self.mpl_connections:
                self.canvas.mpl_disconnect(cid)
            self.mpl_connections = []

            # Connect ONLY the pick_event when in analysis mode
            self.mpl_connections.append(self.canvas.mpl_connect('pick_event', self.on_pick))
            
            self.setCursor(Qt.CrossCursor) # Change cursor to crosshair
            if self.image_viewer:
                self.image_viewer.update_status("Plot analysis mode: Click points to select.", 3000)

        else:
            # Disconnect our custom pick_event handler
            for cid in self.mpl_connections:
                self.canvas.mpl_disconnect(cid)
            self.mpl_connections = [] # Clear the list of connections

            # Restore default toolbar message function
            self.toolbar.set_message = self.original_toolbar_set_message
            # Restore the original active mode of the toolbar
            self.toolbar.mode = self.original_toolbar_active_mode if self.original_toolbar_active_mode is not None else ''
            
            # Reconnect toolbar's default event handlers using the stored CIDs
            if self._toolbar_press_cid is not None:
                # Ensure it's not already connected before trying to reconnect
                # This check is important because canvas.mpl_disconnect doesn't raise error
                # if the cid is not active, but we don't want to double connect.
                # The stored _toolbar_press_cid will be the *original* one, so we just reconnect it.
                self.canvas.mpl_connect('button_press_event', self.toolbar.press)
                self.canvas.mpl_connect('button_release_event', self.toolbar.release)
                self.canvas.mpl_connect('motion_notify_event', self.toolbar.drag)
            
            self.setCursor(Qt.ArrowCursor) # Restore default cursor
            if self.image_viewer:
                self.image_viewer.update_status("Plot navigation mode enabled.", 3000)

    def on_plot_click(self, event):
        """
        Handles mouse click events on the plot.
        This method is primarily used to provide immediate feedback like cursor changes.
        The actual data point picking is handled by on_pick.
        """
        if event.inaxes == self.ax:
            # You can add visual feedback for clicks here, but actual data picking is in on_pick
            # For example, if you wanted to draw a temporary marker on click
            pass

    def on_pick(self, event):
        """
        Handles pick events on the plot, triggered when a data point is clicked/selected.
        """
        # CHECK: do only if lasso button is toggled
        if event.artist == self.artist:
            # Check if the picked artist is our scatter plot
            if event.ind is not None and len(event.ind) > 0:
                # event.ind gives the indices of the data points that were picked
                # We take the first one if multiple are picked (e.g., closely spaced points)
                data_index = event.ind[0]
                
                if self.plotted_object_ids is not None and data_index < len(self.plotted_object_ids):
                    object_id = self.plotted_object_ids[data_index]
                    self.plot_point_clicked.emit(object_id)
                    if self.image_viewer:
                        self.image_viewer.update_status(f"Selected OBJECT_ID: {object_id}", 2000)
                else:
                    if self.image_viewer:
                        self.image_viewer.update_status("No OBJECT_ID found for selected point.", 2000)

    def activate_lasso_selector(self):
        """Activates the lasso selector tool."""
        if self.lasso_selector is not None:
            self.deactivate_lasso_selector()

        if self.artist is None or self.plotted_x_data is None or self.plotted_y_data is None:
            self.lasso_tool.set_active(False)
            self.select_action.setChecked(False)
            if self.image_viewer:
                self.image_viewer.update_status("No valid plot data for lasso.", 2000)
            return

        # Deactivate pan and zoom tools
        if 'pan' in self.toolbar._actions and self.toolbar._actions['pan'].isChecked():
            self.toolbar.pan()
        if 'zoom' in self.toolbar._actions and self.toolbar._actions['zoom'].isChecked():
            self.toolbar.zoom()

        self.original_toolbar_active_mode = self.toolbar.mode
        self.toolbar.mode = ''  # Default mode

        # Disconnect toolbar's default event handlers
        if self._toolbar_press_cid is not None:
            self.canvas.mpl_disconnect(self._toolbar_press_cid)
            self._toolbar_press_cid = None
        if self._toolbar_release_cid is not None:
            self.canvas.mpl_disconnect(self._toolbar_release_cid)
            self._toolbar_release_cid = None
        if self._toolbar_motion_cid is not None:
            self.canvas.mpl_disconnect(self._toolbar_motion_cid)
            self._toolbar_motion_cid = None
            
        # Store original colors
        if self.zaxisComboBox.currentText():
            self.original_colors = self.artist.get_array().copy()
            self.artist._original_array = self.original_colors.copy()
        else:
            n_points = len(self.plotted_x_data)
            default_color = self.artist.get_facecolors()[0] if len(self.artist.get_facecolors()) > 0 else to_rgba('tab:blue')
            self.original_colors = np.tile(default_color, (n_points, 1))

        # Reapply existing selections
        self.selected_indices = np.where(np.isin(self.plotted_object_ids, self.selected_object_ids))[0]
        if self.selected_indices.size > 0:
            if self.zaxisComboBox.currentText():
                colors = self.original_colors.copy()
                colors[self.selected_indices] = np.max(self.original_colors)
                self.artist.set_array(colors)
            else:
                n_points = len(self.plotted_x_data)
                colors = np.ones((n_points, 4))
                colors[:] = default_color
                colors[self.selected_indices] = to_rgba('tab:red')
                colors[np.setdiff1d(np.arange(n_points), self.selected_indices)] = to_rgba('tab:blue')
                self.artist.set_facecolors(colors)
            self.original_colors = colors.copy()
            self.canvas.draw()

        points = np.column_stack([self.plotted_x_data, self.plotted_y_data])
        self.lasso_selector = LassoSelector(
            self.ax,
            onselect=self.on_lasso_select,
            useblit=True,
            props=dict(color='black', linewidth=1),
            button=1
        )
        self._lasso_connection_id = self.canvas.mpl_connect('button_press_event', self.lasso_selector.press)

        self.toolbar.set_message = lambda x: None
        self.setCursor(Qt.CrossCursor)
        self.canvas.setFocus()
        if self.image_viewer:
            self.image_viewer.update_status("Lasso selection mode: Draw around points to select.", 3000)


    def deactivate_lasso_selector(self):
        """Deactivates the lasso selector tool and restores toolbar navigation."""
        try:
            if self.lasso_selector is not None:
                self.lasso_selector.set_active(False)
                if self._lasso_connection_id is not None:
                    self.canvas.mpl_disconnect(self._lasso_connection_id)
                    self._lasso_connection_id = None
                self.lasso_selector = None

            # Restore point colors with safety checks
            if self.artist is not None and self.original_colors is not None:
                # Ensure original_colors is a valid shape for the artist
                if self.zaxisComboBox.currentText():
                    self.artist.set_array(np.atleast_1d(self.original_colors).flatten())
                else:
                    self.artist.set_facecolors(self.original_colors)
                
                if self.canvas:
                    self.canvas.draw_idle()

            self.selected_indices = np.array([], dtype=int)
            self.original_colors = None

            # Restore toolbar navigation
            self.toolbar.set_message = self.original_toolbar_set_message
            self.toolbar.mode = '' 
            for action_name in ['pan', 'zoom']:
                if action_name in self.toolbar._actions:
                    action = self.toolbar._actions[action_name]
                    if action.isChecked():
                        action.setChecked(False)
                        action.toggled.emit(False)

            self.setCursor(Qt.ArrowCursor)
            if self.canvas:
                self.canvas.setFocus()
                self.canvas.draw_idle()
            
            if self.image_viewer:
                self.image_viewer.update_status("Lasso selection disabled.", 2000)
        except Exception as e:
            print(f"Error during lasso deactivation: {e}")

    def deactivate_lasso_selector_old(self):
        """Deactivates the lasso selector tool and restores toolbar navigation."""
        if self.lasso_selector is not None:
            self.lasso_selector.set_active(False)
            if self._lasso_connection_id is not None:
                self.canvas.mpl_disconnect(self._lasso_connection_id)
                self._lasso_connection_id = None
            self.lasso_selector = None

        # Restore point colors
        if self.artist is not None and self.original_colors is not None:
            if self.zaxisComboBox.currentText():
                self.artist.set_array(self.original_colors)
            else:
                self.artist.set_facecolors(self.original_colors)
            self.canvas.draw()

        self.selected_indices = np.array([], dtype=int)  # Keep self.selected_object_ids
        self.original_colors = None

        # Restore toolbar navigation
        self.toolbar.set_message = self.original_toolbar_set_message
        self.toolbar.mode = ''  # Reset to default mode
        # Ensure pan and zoom buttons are unchecked
        for action_name in ['pan', 'zoom']:
            if action_name in self.toolbar._actions:
                action = self.toolbar._actions[action_name]
                if action.isChecked():
                    action.setChecked(False)
                    action.toggled.emit(False)

        # Clear any custom event handlers
        if self._toolbar_press_cid is not None:
            self.canvas.mpl_disconnect(self._toolbar_press_cid)
            self._toolbar_press_cid = None
        if self._toolbar_release_cid is not None:
            self.canvas.mpl_disconnect(self._toolbar_release_cid)
            self._toolbar_release_cid = None
        if self._toolbar_motion_cid is not None:
            self.canvas.mpl_disconnect(self._toolbar_motion_cid)
            self._toolbar_motion_cid = None

        self.setCursor(Qt.ArrowCursor)
        self.canvas.setFocus()  # Ensure canvas has focus
        self.canvas.draw_idle()  # Ensure interactivity
        if self.image_viewer:
            self.image_viewer.update_status("Lasso selection disabled.", 2000)

    def on_lasso_select(self, verts):
        """Handles the lasso selection event with defensive array shaping."""
        if self.plotted_x_data is None or self.plotted_y_data is None or self.artist is None:
            return

        try:
            # Create path from lasso vertices
            path = Path(verts)
            points = np.column_stack([self.plotted_x_data, self.plotted_y_data])
            
            # Find indices of points inside the lasso path
            ind = np.nonzero(path.contains_points(points))[0]

            # Update selected OBJECT_IDs
            if self.plotted_object_ids is not None and ind.size > 0:
                new_selected_ids = self.plotted_object_ids[ind]
                self.selected_object_ids = np.union1d(self.selected_object_ids, new_selected_ids)
                # Ensure 1D and flattened
                self.selected_indices = np.where(np.isin(self.plotted_object_ids, self.selected_object_ids))[0]
                self.selected_indices = self.selected_indices.flatten()
                self.lasso_points_selected.emit(self.selected_object_ids.tolist())
            else:
                self.selected_indices = np.union1d(self.selected_indices, ind).flatten()
                self.lasso_points_selected.emit([])

            # Update colors
            if self.zaxisComboBox.currentText():
                colors = self.original_colors.copy()
                # Ensure the color mapping is rank-1
                colors = np.atleast_1d(colors).flatten()
                colors[self.selected_indices] = np.max(colors)
                self.artist.set_array(colors)
            else:
                n_points = len(self.plotted_x_data)
                default_color = self.artist.get_facecolors()[0] if len(self.artist.get_facecolors()) > 0 else to_rgba('tab:blue')
                colors = np.tile(default_color, (n_points, 1))
                colors[self.selected_indices] = to_rgba('tab:red')
                # Ensure non-selected points keep their original color
                mask = np.ones(n_points, dtype=bool)
                mask[self.selected_indices] = False
                colors[mask] = to_rgba('tab:blue')
                self.artist.set_facecolors(colors)

            self.original_colors = colors.copy()
            self.canvas.draw_idle() # Safer than draw() during rapid selection

            if self.image_viewer and self.selected_indices.size > 0 and self.plotted_object_ids is not None:
                selected_ids = self.plotted_object_ids[self.selected_indices]
                self.image_viewer.display_selected_MER(selected_ids.tolist())
                self.image_viewer.update_status(f"Selected {len(selected_ids)} objects.", 2000)
                
        except ValueError as ve:
            print(f"Caught Matplotlib rank error: {ve}")
        except Exception as e:
            print(f"Error in on_lasso_select: {e}")


    def on_lasso_select_old(self, verts):
        """Handles the lasso selection event."""
        # DOES NOT WORK WITH WCS PLOT, BECAUSE DISPLAY COORDINATES ARE NOT PROPERLY TRANSFORMED TO DATA COORDINATES. UNCLEAR WHY
        
        if self.plotted_x_data is None or self.plotted_y_data is None or self.artist is None:
            return

        # Create path from lasso vertices
        path = Path(verts)
        points = np.column_stack([self.plotted_x_data, self.plotted_y_data])
        # Find indices of points inside the lasso path
        ind = np.nonzero(path.contains_points(points))[0]

        # Update selected OBJECT_IDs
        if self.plotted_object_ids is not None and ind.size > 0:
            new_selected_ids = self.plotted_object_ids[ind]
            self.selected_object_ids = np.union1d(self.selected_object_ids, new_selected_ids)
            # Update selected indices for the current plot
            self.selected_indices = np.where(np.isin(self.plotted_object_ids, self.selected_object_ids))[0]
            # Emit signal with selected OBJECT_IDs
            self.lasso_points_selected.emit(self.selected_object_ids.tolist())
        else:
            self.selected_indices = np.union1d(self.selected_indices, ind)
            # Emit empty list if no OBJECT_IDs are available
            self.lasso_points_selected.emit([])

        # Update colors
        if self.zaxisComboBox.currentText():
            # If using a colormap, modify the color array
            colors = self.original_colors.copy()
            colors[self.selected_indices] = np.max(self.original_colors)  # Bright red equivalent
            self.artist.set_array(colors)
        else:
            # If no colormap, create a per-point color array
            n_points = len(self.plotted_x_data)
            colors = np.ones((n_points, 4))
            default_color = self.artist.get_facecolors()[0] if len(self.artist.get_facecolors()) > 0 else to_rgba('tab:blue')
            colors[:] = default_color
            colors[self.selected_indices] = to_rgba('tab:red')
            colors[np.setdiff1d(np.arange(n_points), self.selected_indices)] = to_rgba('tab:blue')
            self.artist.set_facecolors(colors)

        self.original_colors = colors.copy()
        self.canvas.draw()

        if self.image_viewer and self.selected_indices.size > 0 and self.plotted_object_ids is not None:
            selected_ids = self.plotted_object_ids[self.selected_indices]
            
            # Call the new display method in the viewer
            self.image_viewer.display_selected_MER(selected_ids.tolist())
            self.image_viewer.update_status(f"Selected {len(selected_ids)} objects.", 2000)



    def add_navigation_toolbar(self):
        # --- Add Matplotlib Navigation Toolbar ---
        self.toolbar = NavigationToolbar(self.canvas, self.plotWidget)
        self.plotWidget.layout().addWidget(self.toolbar)  # Add to layout
        self.original_toolbar_set_message = self.toolbar.set_message  # Store original

        # Initialize event handler IDs (for tracking, not reconnection)
        self._toolbar_press_cid = None
        self._toolbar_release_cid = None
        self._toolbar_motion_cid = None

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

