import sys
import logging
import argparse
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QWidget, QMessageBox
from PyQt5.QtCore import Qt

from .image_viewer import ImageViewer
from .control_dock import ControlDock

# Set up basic logging to terminal
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Euniverse Explorer")
        
        # Centralized path management
        self.base_path = Path(__file__).parent
        
        try:
            self.init_ui()
        except Exception as e:
            logger.error(f"Failed to initialize UI: {e}")
            self.critical_error(f"UI Initialization failed: {e}")
            
        self.resize(1280, 1080)

    def init_ui(self):
        """Builds the main layout and connects components."""
        self.setDockOptions(QMainWindow.AllowNestedDocks | QMainWindow.AnimatedDocks)
        
        # Set all corners to belong to left/right docks (better for widescreen)
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        
        # Central widget
        self.viewer = ImageViewer(main_window=self)
        self.setCentralWidget(self.viewer)
        self.setStatusBar(self.viewer.status_bar)
        
        # Setup Control Dock
        self._setup_control_panel()
        
    def _setup_control_panel(self):
        """Internal helper to isolate docking logic."""
        self.control_widget = ControlDock(self.viewer)
        self.control_dock = QDockWidget("Controls", self)
        self.control_dock.setWidget(self.control_widget)
        self.control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Optional: Keep title bar but make it small/clean
        # self.control_dock.setTitleBarWidget(QWidget()) 
        
        self.addDockWidget(Qt.LeftDockWidgetArea, self.control_dock)
        self.viewer.set_control_dock(self.control_widget)

    def critical_error(self, message):
        """Graceful crash reporting."""
        QMessageBox.critical(self, "Critical Error", message)
        sys.exit(1)

def main():
    # 1. Versioning
    try:
        current_version = version("euniverse")
    except PackageNotFoundError:
        current_version = "dev-local"

    # 2. CLI Arguments
    parser = argparse.ArgumentParser(description="euniverse: Euclid data analysis")
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {current_version}')
    parser.add_argument('--debug', action='store_true', help="Enable verbose logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # 3. Application Lifecycle
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Application crashed on startup: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
