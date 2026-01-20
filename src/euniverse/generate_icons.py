from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QBrush, QColor, QPainterPath
from PyQt5.QtCore import Qt, QRectF

def create_sunglasses_icon():
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Scale symbols by 1.5x, adjust to center in 32x32
    scale = 1.5
    painter.translate(16, 16)  # Center of 32x32
    painter.scale(scale, scale)
    painter.translate(-15 / scale, -11 / scale)  # Adjust for scaled center
    
    # Lenses (original 10x8, now 15x12, scaled to fit)
    painter.setPen(QPen(Qt.black, 1.5 / scale))
    painter.setBrush(QBrush(Qt.black))
    painter.drawEllipse(4, 6, 5, 4)  # Left lens
    painter.drawEllipse(11, 6, 5, 4)  # Right lens
    
    # Bridge (curved line)
    painter.setPen(QPen(Qt.black, 1.5 / scale))
    painter.drawArc(7, 6, 6, 4, 30 * 16, 120 * 16)
    
    # Temples
    painter.drawLine(3, 7, 4, 7)  # Left temple
    painter.drawLine(16, 7, 17, 7)  # Right temple
    
    painter.end()
    return QIcon(pixmap)

def create_MER_icon():
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Scale symbols by 1.5x, adjust to center in 32x32
    scale = 1.5
    painter.translate(16, 16)
    painter.scale(scale, scale)
    painter.translate(-14 / scale, -14 / scale)
    
    # Larger ellipse (original 20x10, now 30x15, scaled to fit)
    painter.save()
    painter.translate(9, 10)
    painter.rotate(45)
    painter.setPen(QPen(Qt.blue, 1.5 / scale))
    painter.setBrush(QBrush(Qt.blue, Qt.SolidPattern))
    painter.drawEllipse(QRectF(-8, -4, 10, 5))
    painter.restore()
    
    # Smaller ellipse (original 10x5, now 15x7.5, scaled to fit)
    painter.save()
    painter.translate(15, 14)
    painter.rotate(-30)
    painter.setPen(QPen(Qt.red, 1.5 / scale))
    painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
    painter.drawEllipse(QRectF(-5, -2.25, 7, 4))
    painter.restore()
    
    painter.end()
    return QIcon(pixmap)

def create_table_icon():
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Scale symbols by 1.5x, adjust to center in 32x32
    scale = 1.8
    painter.translate(16, 16)
    painter.scale(scale, scale)
    painter.translate(-18 / scale, -18 / scale)
    
    # Grid (original 20x24, now 30x36, scaled to fit)
    painter.setPen(QPen(Qt.black, 1.5 / scale))
    # Horizontal lines
    painter.drawLine(5, 6, 15, 6)
    painter.drawLine(5, 10, 15, 10)
    painter.drawLine(5, 14, 15, 14)
    # Vertical lines
    painter.drawLine(8, 4, 8, 16)
    painter.drawLine(12, 4, 12, 16)
    # Border
    painter.drawRect(5, 4, 10, 12)
    
    painter.end()
    return QIcon(pixmap)

def create_camera_icon():
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Scale symbols by 1.8x, adjust to center in 32x32
    scale = 1.8
    painter.translate(16, 16)
    painter.scale(scale, scale)
    painter.translate(-18 / scale, -18 / scale)
    
    # Camera body (solid black rectangle)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(Qt.black))
    painter.drawRect(4, 8, 12, 9)
    
    # Camera lens (solid light blue circle)
    painter.setBrush(QBrush(QColor(135,206,235)))
    painter.drawEllipse(7, 10, 5, 5)
    
    # Inner lens circle (solid white, slightly off-center)
    painter.setBrush(QBrush(Qt.white))
    painter.drawEllipse(9, 11, 2, 2)
    
    # Viewfinder on top (centered)
    painter.setBrush(QBrush(Qt.black))
    painter.drawRect(8, 6, 4, 2)
    
    painter.end()
    return QIcon(pixmap)


def create_scatter_plot_icon():
    """
    Generates a 32x32 icon that shows an x-y coordinate system with 6 closer scatter points.
    """
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # Define drawing area for the plot within the icon
    plot_left = 4
    plot_top = 4
    plot_width = 24
    plot_height = 24

    # Draw axes
    axis_pen = QPen(Qt.black, 1)
    painter.setPen(axis_pen)

    # Y-axis
    painter.drawLine(plot_left, plot_top, plot_left, plot_top + plot_height)
    # X-axis
    painter.drawLine(plot_left, plot_top + plot_height, plot_left + plot_width, plot_top + plot_height)

    # Draw scatter points
    point_brush = QBrush(QColor(50, 50, 255))
    point_pen = QPen(QColor(0, 0, 150), 0.5)
    painter.setBrush(point_brush)
    painter.setPen(point_pen)

    point_radius = 2

    # Define 6 points with closer spacing
    normalized_points = [
        (0.2, 0.7),  # Slightly top-left
        (0.4, 0.5),  # Center-ish
        (0.6, 0.6),  # Slightly top-right
        (0.3, 0.3),  # Bottom-left
        (0.5, 0.2),  # Bottom-center
        (0.7, 0.4)   # Bottom-right
    ]

    for px_norm, py_norm in normalized_points:
        x_icon = plot_left + px_norm * plot_width
        y_icon = plot_top + plot_height - py_norm * plot_height
        painter.drawEllipse(QRectF(x_icon - point_radius, y_icon - point_radius,
                                   2 * point_radius, 2 * point_radius))

    painter.end()
    return QIcon(pixmap)

def create_crosshair_icon():
    """
    Generates a 32x32 icon that displays a circle with short vertical and horizontal lines
    extending outwards from its sides.
    """
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)  # Start with a transparent background
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing) # For smoother lines

    # Set the pen for drawing
    pen = QPen(Qt.black, 2)  # Black color, 2 pixels wide
    painter.setPen(pen)

    center_x, center_y = 16, 16
    circle_radius = 6

    # Draw the central circle
    # QRectF(x, y, width, height) where x,y is top-left of bounding rectangle
    painter.drawEllipse(QRectF(center_x - circle_radius, center_y - circle_radius,
                               2 * circle_radius, 2 * circle_radius))

    line_length = 4 # Length of the lines extending from the circle

    # Draw the top line
    painter.drawLine(center_x, center_y - circle_radius - line_length,
                     center_x, center_y - circle_radius)

    # Draw the bottom line
    painter.drawLine(center_x, center_y + circle_radius,
                     center_x, center_y + circle_radius + line_length)

    # Draw the left line
    painter.drawLine(center_x - circle_radius - line_length, center_y,
                     center_x - circle_radius, center_y)

    # Draw the right line
    painter.drawLine(center_x + circle_radius, center_y,
                     center_x + circle_radius + line_length, center_y)

    painter.end()
    return QIcon(pixmap)

def create_lasso_icon():
    """Creates a lasso selector icon for the toolbar."""
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(Qt.black, 2)
    painter.setPen(pen)
    # Draw a lasso-like loop shape
    path = QPainterPath()
    path.moveTo(6, 6)
    path.quadTo(12, 4, 18, 6)
    path.quadTo(20, 12, 18, 18)
    path.quadTo(12, 20, 6, 18)
    path.quadTo(4, 12, 6, 6)
    painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)
