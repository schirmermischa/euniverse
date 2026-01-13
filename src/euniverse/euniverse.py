#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# eummy.py - A program to create color images from Euclid MER stacks
# Copyright (C) 2025 Mischa Schirmer

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

import sys
import argparse
from importlib.metadata import version, PackageNotFoundError
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QWidget
from PyQt5.QtCore import Qt
from .image_viewer import ImageViewer
from .control_dock import ControlDock

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Euniverse Explorer")
        self.init_ui()
        self.resize(1280, 1080)

    def init_ui(self):
        # Docking settings
        self.setDockOptions(QMainWindow.AllowNestedDocks | QMainWindow.AnimatedDocks)
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        
        # Central widget: ImageViewer
        self.viewer = ImageViewer(main_window=self)
        self.setCentralWidget(self.viewer)
        self.setStatusBar(self.viewer.status_bar)
        
        # Control widget (your custom class)
        self.control_widget = ControlDock(self.viewer)  # instantiate ControlDock
        
        # Wrap it in a QDockWidget
        self.control_dock = QDockWidget("", self)
        self.control_dock.setWidget(self.control_widget)  # embed ControlDock inside
        self.control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.control_dock.setTitleBarWidget(QWidget())  # Removes title bar
        self.addDockWidget(Qt.LeftDockWidgetArea, self.control_dock)
        
        # Pass the ControlDock to the viewer (not the QDockWidget!)
        self.viewer.set_control_dock(self.control_widget)

def main():
    # 1. Handle Versioning
    try:
        current_version = version("euniverse")
    except PackageNotFoundError:
        current_version = "dev"

    # 2. Setup Parser
    parser = argparse.ArgumentParser(
        description="euniverse: Euclid data visualization and analysis"
    )
    parser.add_argument(
        '-v', '--version', 
        action='version', 
        version=f'%(prog)s {current_version}'
    )
    parser.parse_args()

    # 3. Launch GUI
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Consistent button rendering
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
