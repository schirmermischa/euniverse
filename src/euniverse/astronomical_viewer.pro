TEMPLATE = app
QT += widgets

SOURCES += main.py image_viewer.py control_dock.py extended_viewer.py wcs_utils.py
FORMS += control_dock.ui \
    catalog_plotter.ui

DISTFILES += \
    CatalogPlotter.py
