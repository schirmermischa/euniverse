import os
import re
import warnings
import gc
from astropy.table import Table
from astropy.io import fits
from PyQt5.QtWidgets import QGraphicsEllipseItem
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtCore import QRectF
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


# Define the unit locally if it doesn't exist
try:
    # We define it but don't need register_to_subclass=True for simple reading
    NA_UNIT = u.def_unit('NA', u.dimensionless_unscaled)
except ValueError:
    # If already defined (e.g., during a re-import), grab the existing one
    NA_UNIT = u.Unit('NA')


class CatalogManager:
    def __init__(self, tileID, wcs, search_dir=".", image_viewer=None):
        """
        Initialize CatalogManager with a tile ID, WCS, and optionally the image_viewer
        
        Args:
            tileID (str): Tile identifier (e.g., 'TILE123456789').
            wcs (WCSConverter): WCS object for coordinate conversion.
            search_dir (str): Directory to search for FITS table.
            image_viewer (ImageViewer, optional): Reference to the ImageViewer instance.
       """
        self.tileID = tileID
        self.wcs = wcs
        self.search_dir = search_dir
        self.catalog = None
        self.MER_items = []
        self.selected_MER_items = []
        self.image_viewer = image_viewer
        self.catalog_name = None
        self.catalog_path = None
        self.has_magnitudes = False
        self.load_catalog()
        self.numsources = self.get_catalog_row_count()
        
    def load_catalog(self):
        """
        Search for and load the FITS table with the matching tile_id.
        """

        # Clear an old catalog immediately to free up memory 
        # before starting the new search and load process.
        self.catalog = None
        gc.collect()
        
        try:
            # Ensure the custom unit is defined
            try:
                na_unit = u.Unit('NA')
            except ValueError:
                na_unit = u.def_unit('NA', u.dimensionless_unscaled)

            if not os.path.isdir(self.search_dir): return
            
            for filename in os.listdir(self.search_dir):
                if (
                    self.tileID in filename and
                    'EUC_MER_FINAL-CAT' in filename and
                    filename.lower().endswith('.fits')
                ):
                    self.catalog_name = filename[:-5] + "\n"
                    filepath = os.path.join(self.search_dir, filename)

                    # Wrap the read in a warning filter and enabled units context
                    with u.add_enabled_units([na_unit]), warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=u.UnitsWarning)
                        warnings.simplefilter('ignore', category=fits.verify.VerifyWarning)
                        
                        self.catalog = Table.read(filepath, format='fits')

                        # Optional: Force columns to stay in their native format if they were read as objects
                        for col in self.catalog.colnames:
                            if self.catalog[col].dtype == np.float64:
                                # Only keep as float64 if the FITS file explicitly used Double Precision (D)
                                # Otherwise, ensure we aren't upcasting single-precision (E) columns.
                                pass 

                    self.catalog_path = self.search_dir
                    self.delete_empty_columns()
                    self.compute_magnitudes()
                    
                    if self.image_viewer:
                        self.image_viewer.update_status(f"Loaded MER catalog: {filename}", 3000)
                    return
                
            raise FileNotFoundError(f"MER catalog not present for {self.tileID}")

        except Exception as e:
            if self.image_viewer:
                self.image_viewer.update_status(f"Error loading catalog: {e}", 10000)
#            print(f"Error loading MER catalog: {e}")
            else:
                print(f"INFO: {e}")
            self.catalog = None

    def get_catalog_row_count(self):
        """
        Returns the number of rows in the loaded catalog.  Returns 0 if no catalog is loaded.
        """
        if self.catalog is not None:
            return len(self.catalog)
        else:
            return 0

    def get_non_empty_columns(self):
        """Return list of non-empty column names in the catalog."""
        if not self.catalog:
            return []
        return [col for col in self.catalog.colnames if not all(self.catalog[col].mask)]

    def get_all_column_names(self):
        """Return a list of all column names in the catalog."""
        if self.catalog is not None:
            return self.catalog.colnames
        else:
            return []

    def delete_empty_columns(self):
        columns_to_remove = []
        for col_name in self.catalog.colnames:
            column_data = self.catalog[col_name]
            # Check if the column is entirely masked (for Astropy Table MaskedColumn)
            if hasattr(column_data, 'mask') and np.all(column_data.mask):
                columns_to_remove.append(col_name)
            # Check if all values are NaN or None (for regular numpy arrays or lists)
            elif isinstance(column_data, np.ndarray) and np.all(np.isnan(column_data)):
                columns_to_remove.append(col_name)
            elif all(item is None for item in column_data):
                 columns_to_remove.append(col_name)
        if columns_to_remove:
            self.catalog.remove_columns(columns_to_remove)
            if self.image_viewer:
                self.image_viewer.update_status(f"Removed {len(columns_to_remove)} empty columns.", 2000)

        
    def compute_magnitudes(self):
        # leave if done already
        if self.has_magnitudes:
            return
        
        # Compute AB magnitudes for FLUX columns in microJansky
        for col in self.catalog.colnames:
            if col.startswith('FLUX'):
                new_col_name = col.replace('FLUX', 'MAG')
                log_input = 1e-6 * self.catalog[col].data
                # Calculate mag only for positive fluxes, assign 99 otherwise
                mask = log_input > 0
                # For non-positive values, this part will be ignored by np.where
                good_mag = -2.5 * np.log10(log_input[mask]) + 8.90
                # Initialize an array of the same shape as log_input, filled with 99
                mag = np.full_like(log_input, 99.0, dtype=np.float32)
                # Assign the calculated magnitudes only where mask is True
                mag[mask] = good_mag
                self.catalog[new_col_name] = mag

        self.has_magnitudes = True


    def make_ellipse(self, x, y, a, b, pa, color, width, ID):
        ellipse = QGraphicsEllipseItem(QRectF(x-a, y-b, 2*a, 2*b))
        ellipse.setPen(QPen(color, width))
        ellipse.setTransformOriginPoint(x,y)
        ellipse.setRotation(90-pa)  # Correct translation of position angle to image display
        ellipse.setData(0, ID)  # Store OBJECT_ID
        return ellipse
        
    def get_MER(self, image_height):
        """
        Extract source parameters from the catalog and create MER ellipse items.
        
        Args:
        image_height (int): Image height in pixels for TIFF y-coordinate flip.
        
        Returns:
        list: List of QGraphicsEllipseItem objects.
        """
        if not self.catalog:
            print("No MER catalog available for plotting")
            return []
        
        if not self.wcs:
            print("No WCS; cannot overlay MER catalog")
            return []

        self.MER_items = []
        try:
            # Extract required columns
            required_columns = [
                'OBJECT_ID', 'RIGHT_ASCENSION', 'DECLINATION',
                'SEMIMAJOR_AXIS', 'POSITION_ANGLE', 'ELLIPTICITY'
            ]
            for col in required_columns:
                if col not in self.catalog.colnames:
                    raise KeyError(f"Column {col} missing in FITS table")
                
            # Extract RA and Dec
            ra = self.catalog['RIGHT_ASCENSION'].data
            dec = self.catalog['DECLINATION'].data
            a = self.catalog['SEMIMAJOR_AXIS'].data
            e = self.catalog['ELLIPTICITY'].data
            pa = self.catalog['POSITION_ANGLE'].data
            object_ids = self.catalog['OBJECT_ID'].data
            
            # Ensure RA and Dec are numeric arrays
            ra = np.array(ra, dtype=float)
            dec = np.array(dec, dtype=float)
            
            # Create SkyCoord for all coordinates
            sky_coords = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')

            # Convert RA/Dec to pixel coordinates
            x, y = self.wcs.world_to_pixel(sky_coords)
            y = image_height - y  # Flip y-axis to match image coordinates

            # Compute semi-minor axis: B = A * (1 - E)
            a = 3 * a # Rescale to make SourceExtractor shape parameters approximate the true extent
            b = a * (1 - e)
            # Create MER ellipse items
            for i in range(len(x)):
                ellipse = self.make_ellipse(x[i], y[i], a[i], b[i], pa[i], QColor(255, 0, 0), 1, object_ids[i])
                self.MER_items.append(ellipse)
                
            if self.image_viewer:
                self.image_viewer.update_status(f"Retrieved {len(self.MER_items)} sources from MER catalog")

        except Exception as e:
            if self.image_viewer:
                self.image_viewer.update_status(f"Error processing MER cat: {e}",5000)
            print(f"Error processing MER cat: {e}")

        
    def handle_selected_objects(self, object_ids):
        """
        Bridge method called by the plotter. 
        It tells the viewer to display these specific IDs.
        """
        if self.image_viewer:
            self.image_viewer.display_selected_MER(object_ids)

            
    def get_selected_MER(self, selected_object_ids, image_height):
        """
        Extract parameters and create items for selected IDs.
        Does NOT add them to the scene; returns them for the viewer to handle.
        """
        if not self.catalog or not self.wcs or not selected_object_ids:
            return []

        # Clear existing selection references
        self.selected_MER_items = []
        
        try:
            # Filter catalog for selected IDs
            mask = np.isin(self.catalog['OBJECT_ID'].data, selected_object_ids)
            if not np.any(mask):
                return []
            
            selected_catalog = self.catalog[mask]
            ra = np.array(selected_catalog['RIGHT_ASCENSION'].data, dtype=float)
            dec = np.array(selected_catalog['DECLINATION'].data, dtype=float)
            a = selected_catalog['SEMIMAJOR_AXIS'].data
            e = selected_catalog['ELLIPTICITY'].data
            pa = selected_catalog['POSITION_ANGLE'].data
            object_ids = selected_catalog['OBJECT_ID'].data

            sky_coords = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
            x, y = self.wcs.world_to_pixel(sky_coords)
            y = image_height - y 

            a_scaled = 3 * a 
            b_scaled = a_scaled * (1 - e)

            for i in range(len(x)):
                ellipse = self.make_ellipse(x[i], y[i], a_scaled[i], b_scaled[i], pa[i], QColor(255, 255, 0), 1, object_ids[i])
                self.selected_MER_items.append(ellipse)

            # Center on first object
            if len(x) > 0:
                self.image_viewer.centerOn(x[0], y[0])
            
            return self.selected_MER_items

        except Exception as e:
            print(f"Error in get_selected_MER: {e}")
            return []


    # Currently unused
    def clear_MER(self):
        """
        Clear the list of MER items.
        """
        self.image_viewer.clear_MER()
        self.MER_items = []

    def clear_selected_MER(self):
        """Wipes the selection data and tells viewer to clean the scene."""
        if self.image_viewer:
            self.image_viewer.clear_selected_MER()
        else:
            self.selected_MER_items = []

