from PyQt5.QtWidgets import QDialog, QAbstractItemView, QTableView, QVBoxLayout, QHeaderView, QGraphicsEllipseItem, QGraphicsView 
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QPointF
from PyQt5.QtGui import QColor, QPen
from astropy.table import Table, MaskedColumn
import numpy as np

class CatalogTableModel(QAbstractTableModel):
    """
    A table model to display an astropy Table in a QTableView,
    with support for filtering columns and preserving original data types.
    """

    def __init__(self, catalog, required_columns=None, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.required_columns = required_columns
        self.columns = self._get_visible_columns()
        self.data_types = self._get_column_data_types()

    def _get_visible_columns(self):
        """
        Determines which columns to display from the catalog,
        respecting the required_columns if provided.
        Excludes columns with object types.
        """
        visible_columns = []
        all_columns = self.catalog.colnames
        if self.required_columns:
            for col in self.required_columns:
                if col in all_columns and self.catalog[col].dtype.kind not in ['O', 'U', 'S']:
                    visible_columns.append(col)
        else:
            for col in all_columns:
                if self.catalog[col].dtype.kind not in ['O', 'U', 'S']:
                    visible_columns.append(col)
        return visible_columns

    def _get_column_data_types(self):
        """
        Gets the data types of the visible columns.
        """
        data_types = {}
        for col in self.columns:
            data_types[col] = self.catalog[col].dtype
        return data_types

    def rowCount(self, parent=QModelIndex()):
        return len(self.catalog) if self.catalog is not None else 0

    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not self.catalog:
            return None

        row = index.row()
        col = index.column()
        col_name = self.columns[col]
        value = self.catalog[row][col_name]

        if role == Qt.DisplayRole:
            if isinstance(value, MaskedColumn):
                return str(value.data[0])  # Or handle masked values as needed
            else:
                # Preserve original formatting!
                return str(value)
        elif role == Qt.TextAlignmentRole:
            if np.issubdtype(self.data_types[col_name], np.number):
                return Qt.AlignRight
            return Qt.AlignLeft
        elif role == Qt.BackgroundRole:
            if row % 2 == 0:
                return QColor(240, 240, 240)  # Light gray background on even rows
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns[section]
        return None


class TableDialog(QDialog):
    def __init__(self, catalog, viewer, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.viewer = viewer
        self.setWindowTitle("FITS Table Viewer")
        self.required_columns = [
            'OBJECT_ID', 'RIGHT_ASCENSION', 'DECLINATION', 'MAG_VIS_PSF',
            'MAG_Y_TEMPLFIT', 'MAG_J_TEMPLFIT', 'MAG_H_TEMPLFIT', 'FWHM'
        ]  # Define required columns here
        self.init_ui()
        self.selected_row = None

    def init_ui(self):
        layout = QVBoxLayout()

        # Create table view
        self.table_view = QTableView()
        self.table_model = CatalogTableModel(self.catalog, self.required_columns)  # Pass required_columns
        self.table_view.setModel(self.table_model)

        # Configure table appearance
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Make it read-only
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        layout.addWidget(self.table_view)
        self.setLayout(layout)
        self.resize(800, 600)

        # Connect row selection
        self.table_view.clicked.connect(self.on_row_clicked)

    def on_row_clicked(self, index):
        """
        Handles row clicks in the table: highlights ellipse and centers the *view* on it.
        Does not change the drag mode.
        """
        if not self.viewer or not self.catalog or not self.viewer.wcs or self.viewer.original_image is None:
            print("Invalid state: viewer, catalog, wcs, or original_image missing")
            return

        self.selected_row = index.row()
        object_id = self.catalog['OBJECT_ID'][index.row()]

        # Highlight corresponding ellipse
        for item in self.viewer.scene.items():
            if isinstance(item, QGraphicsEllipseItem) and item.data(0) == object_id:
                pen = QPen(QColor(255, 255, 0), 1.5)  # Yellow
                item.setPen(pen)

                # Center the *view* on the ellipse's center
                rect = item.rect()
                center_scene = item.mapToScene(rect.center())
                self.viewer.centerOn(center_scene)
                self.viewer.setDragMode(QGraphicsView.NoDrag)
                self.viewer.viewport().setCursor(Qt.ArrowCursor)  # Change the cursor too
            elif isinstance(item, QGraphicsEllipseItem) and item.data(0) != object_id:
                pen = QPen(QColor(255, 0, 0), 1.0)  # Red
                item.setPen(pen)

        self.viewer.scene.update()

    def select_row_by_object_id(self, object_id):
        if not self.catalog:
            return

        # Find row with matching OBJECT_ID
        for row in range(len(self.catalog)):
            if self.catalog['OBJECT_ID'][row] == object_id:
                index = self.table_model.index(row, 0)  # Get index for the first column
                self.table_view.selectRow(row)
                self.table_view.setCurrentIndex(index)
                self.selected_row = row
                return
