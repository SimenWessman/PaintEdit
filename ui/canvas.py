import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QInputDialog
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPolygon, QPainterPath, QPolygonF, QBrush
from PyQt5.QtCore import Qt, QPoint, QRect, QEvent, pyqtSignal



class Canvas(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(800, 600)  # Canvas size
        self.setAttribute(Qt.WA_StaticContents)

        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.original_image = self.image.copy()

        self.current_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.tool = None
        self.selection_start = None
        self.selection_path = []
        self.drawing_selection = False
        self.selected_area = None
        self.is_moving_selection = False
        self.selection_offset = QPoint()

        self.last_click_position = None

        # Drawing variables
        self.drawing = False
        self.last_point = QPoint()
        self.brush_color = Qt.black
        self.brush_size = 3
        self.eraser_size = 10

        self.brush_texture = None
        self.textures = {
            "None": None,
            "Dots": Qt.Dense5Pattern,
            "Stripes": Qt.Dense3Pattern,
            "Checkerboard": Qt.CrossPattern,
            "Horizontal Lines": Qt.HorPattern,
            "Vertical Lines": Qt.VerPattern,
        }

        self.text_tool_active = False
        self.text_color = Qt.black
        self.text_font = QtGui.QFont("Arial", 24)

        self.filter_brush_size = 10

        self.current_shape = None
        self.outline_color = Qt.black
        self.fill_color = Qt.transparent

        self.grabGesture(Qt.PinchGesture)

    def reset_colors(self):
        """Reset fill and outline colors to default values."""
        self.outline_color = Qt.black
        self.fill_color = Qt.transparent

        self.update()

    def set_outline_color(self, color):
        """Set the outline color for shapes."""
        self.outline_color = QColor(color)

    def set_fill_color(self, color):
        """Set the fill color for shapes."""
        self.fill_color = QColor(color)


    def set_filter_brush_size(self, size):
        """Set the filter brush size."""
        self.filter_brush_size = size

        self.update()


    def apply_gaussian_filter(self):
        """Apply a Gaussian blur to the entire image."""
        self.status_message.emit("Applying Gaussian Filter...")
        image_array = self.qimage_to_numpy(self.original_image)
        blurred = cv2.GaussianBlur(image_array, (15, 15), 0)

        self.update_image_from_numpy(blurred)


    def apply_sobel_filter(self):
        """Apply a Sobel filter to the entire image."""
        self.status_message.emit("Applying Sobel Filter...")

        image_array = self.qimage_to_numpy(self.original_image)
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))  # Normalize to 8-bit
        sobel_colored = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)

        self.update_image_from_numpy(sobel_colored)


    def apply_binary_filter(self, threshold=None):
        """Apply binary thresholding to the image."""
        self.status_message.emit("Applying Binary Filter...")

        image_array = self.qimage_to_numpy(self.original_image)

        # Grayscale
        if image_array.ndim == 3 and image_array.shape[2] == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array

        # Normalize grayscale to the full 0–255 range
        normalized_gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        print(f"Normalized grayscale array: {normalized_gray.shape}, dtype={normalized_gray.dtype}")

        # Automatically compute threshold if not provided
        if threshold is None or isinstance(threshold, bool):  # Added explicit boolean check
            otsu_threshold, _ = cv2.threshold(normalized_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold = float(otsu_threshold)  # Ensure it is a float
            self.status_message.emit(f"Using Otsu's threshold: {threshold:.2f}")
            print(f"Otsu threshold computed: {threshold} (type: {type(threshold)})")
        else:
            print(f"Threshold provided: {threshold} (type: {type(threshold)})")

        # Validate the threshold
        if not isinstance(threshold, (float, int)):
            print(f"Invalid threshold value: {threshold} (type: {type(threshold)})")
            raise ValueError("Threshold must be a numeric value, not a boolean or invalid type.")

        print(f"Final threshold to be used: {threshold} (type: {type(threshold)})")

        _, binary = cv2.threshold(normalized_gray, threshold, 255, cv2.THRESH_BINARY)

        binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        self.update_image_from_numpy(binary_colored)
        self.status_message.emit("Binary Filter applied successfully.")


    def apply_histogram_thresholding(self):
        """Apply histogram-based thresholding."""
        self.status_message.emit("Applying Histogram Thresholding...")
        image_array = self.qimage_to_numpy(self.original_image)
        print(f"Image shape: {image_array.shape}")
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        print(f"Gray shape: {gray.shape}, dtype: {gray.dtype}")
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Otsu's threshold value: {_}")
        thresholded_colored = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
        self.update_image_from_numpy(thresholded_colored)
        self.update()


    def qimage_to_numpy(self, qimage):
        """Convert QImage to a NumPy array."""
        qimage = qimage.convertToFormat(QImage.Format_RGB32)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        array = np.array(ptr).reshape(height, width, 4)
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)


    def numpy_to_qimage(self, array):
        """Convert a NumPy array to QImage."""
        height, width, channel = array.shape
        bytes_per_line = 3 * width
        qimage = QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage


    def update_image_from_numpy(self, array):
        """Update the QImage from a NumPy array."""
        self.original_image = self.numpy_to_qimage(array)
        self.image = self.original_image.scaled(
            int(self.original_image.width() * self.current_scale),
            int(self.original_image.height() * self.current_scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.update()


    def set_brush_filter(self, filter_name):
        """Set the active brush filter."""
        self.brush_filter = filter_name
        self.tool = 'filter_brush'
        self.status_message.emit(f"Filter brush set to: {filter_name}")


    def wheelEvent(self, event):
        """Zoom in or out based on mouse wheel, using cursor position."""
        if event.angleDelta().y() > 0:
            self.zoom_in(event.pos())
        else:  # Zoom out
            self.zoom_out(event.pos())


    def zoom_in(self, cursor_pos=None):
        """Zoom in on the canvas."""
        self.current_scale *= 1.2
        self.update_canvas_scale(cursor_pos)
        self.status_message.emit("Zoomed In")


    def zoom_out(self, cursor_pos=None):
        """Zoom out on the canvas."""
        self.current_scale *= 0.8
        self.update_canvas_scale(cursor_pos)
        self.status_message.emit("Zoomed Out")


    def enable_text_tool(self):
        """Enable the text tool for adding text to the canvas."""
        self.text_tool_active = True
        self.status_message.emit("Text tool activated. Click to add text.")


    def add_text_at_position(self, position):
        """Prompt user for text and render it on the canvas."""
        text, ok = QInputDialog.getText(self, "Add Text", "Enter text:")
        if ok and text:
            painter = QPainter(self.original_image)
            painter.setFont(self.text_font)
            painter.setPen(QPen(self.text_color))
            painter.drawText(position, text)
            painter.end()

            self.image = self.original_image.scaled(
                int(self.original_image.width() * self.current_scale),
                int(self.original_image.height() * self.current_scale),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.update()
            self.text_tool_active = False
            self.status_message.emit("Text added.")


    def set_text_font(self, font: QtGui.QFont):
        """Set the font for the text tool."""
        self.text_font = font
        self.status_message.emit(f"Font set to: {font.family()}")


    def set_text_color(self, color: QColor):
        """Set the color for the text tool."""
        self.text_color = color
        self.status_message.emit(f"Text color set to: {color.name()}")


    def set_brush_texture(self, texture_name):
        """Set the texture for the brush."""
        self.brush_texture = self.textures.get(texture_name, None)
        self.status_message.emit(f"Brush texture set to: {texture_name}")

        self.update()


    def set_brush_color(self, color):
        """Set the brush color."""
        self.brush_color = QColor(color)

        self.update()


    def set_brush_size(self, size):
        """Set the brush size."""
        self.brush_size = size

        self.update()


    def update_canvas_scale(self, cursor_pos=None):
        """Update the canvas scale and redraw the image, zooming into cursor position if provided."""
        if cursor_pos:
            cursor_pos = self.map_to_scaled_image(cursor_pos)
            cursor_offset_x = (cursor_pos.x() * self.current_scale) - cursor_pos.x()
            cursor_offset_y = (cursor_pos.y() * self.current_scale) - cursor_pos.y()

        else:
            cursor_offset_x, cursor_offset_y = 0, 0

        self.image = self.original_image.scaled(
            int(self.original_image.width() * self.current_scale),
            int(self.original_image.height() * self.current_scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.offset_x -= cursor_offset_x
        self.offset_y -= cursor_offset_y
        self.offset_x = max(0, self.offset_x)
        self.offset_y = max(0, self.offset_y)

        self.update()


    def set_eraser_size(self, size):
        """Set the eraser size."""
        self.eraser_size = size

        self.update()


    def set_tool(self, tool):
        """Set the active tool (None=drawing, 'rect', 'lasso', 'polygon' for selection)."""
        self.tool = tool

        if tool in ['rectangle', 'circle', 'triangle', 'line']:
            self.current_shape = tool
            self.status_message.emit(f"Shape tool set to: {tool}")
        else:
            self.current_shape = None

        self.selection_start = None
        self.selection_path = []
        self.drawing_selection = False
        self.selected_area = None

        self.update()


    def copy_selection(self):
        """Copy the selected area to the clipboard."""
        if not self.selection_path:
            self.status_message.emit("No selection to copy.")
            return

        self.extract_selection()

        if self.selected_area:
            clipboard = QApplication.clipboard()
            clipboard.setImage(self.selected_area)
            self.status_message.emit("Selection copied to clipboard.")


    def paste_selection(self, position=None):
        """Paste clipboard image onto the canvas."""
        clipboard = QApplication.clipboard()
        clipboard_image = clipboard.image()

        if clipboard_image.isNull():
            self.status_message.emit("Clipboard is empty.")
            return

        self.set_tool('rect')

        paste_position = position or self.last_click_position

        if not paste_position:
            paste_position = QPoint(
                (self.original_image.width() - clipboard_image.width()) // 2,
                (self.original_image.height() - clipboard_image.height()) // 2,
            )

        paste_position.setX(max(0, min(self.original_image.width() - clipboard_image.width(), paste_position.x())))
        paste_position.setY(max(0, min(self.original_image.height() - clipboard_image.height(), paste_position.y())))

        # Paste the image onto the canvas
        painter = QPainter(self.original_image)
        painter.drawImage(paste_position, clipboard_image)
        painter.end()

        self.image = self.original_image.scaled(
            int(self.original_image.width() * self.current_scale),
            int(self.original_image.height() * self.current_scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.selection_start = paste_position
        self.selection_path = [
            paste_position,
            paste_position + QPoint(clipboard_image.width(), clipboard_image.height()),
        ]
        self.selected_area = clipboard_image
        self.last_click_position = None

        self.update()
        self.status_message.emit(f"Pasted content at {paste_position.x()}, {paste_position.y()}")


    def draw_rectangle(self, rect: QRect, color: str):
        painter = QPainter(self)
        pen = QPen(color)
        painter.setPen(pen)
        painter.drawRect(rect)

        self.update()


    def draw_triangle(self, points: QPolygon, color: str):
        painter = QPainter(self)
        pen = QPen(color)
        painter.setPen(pen)
        painter.drawPolygon(points)

        self.update()


    def draw_circle(self, center: QPoint, radius: int, color: str):
        painter = QPainter(self)
        pen = QPen(color)
        painter.setPen(pen)
        painter.drawEllipse(center, radius, radius)

        self.update()


    def draw_line(self, start: QPoint, end: QPoint, color: str):
        painter = QPainter(self)
        pen = QPen(color)
        painter.setPen(pen)
        painter.drawLine(start, end)

        self.update()


    def paintEvent(self, event):
        """Render the image and selection on the canvas."""
        canvas_painter = QPainter(self)

        self.offset_x = max((self.width() - self.image.width()) // 2, 0)
        self.offset_y = max((self.height() - self.image.height()) // 2, 0)

        canvas_painter.drawImage(self.offset_x, self.offset_y, self.image)

        if self.selection_path:
            pen = QPen(Qt.blue, 2, Qt.DashLine)
            canvas_painter.setPen(pen)
            scaled_path = [self.map_from_scaled_image(point) for point in self.selection_path]

            if self.tool == 'polygon':
                for i in range(len(scaled_path) - 1):
                    canvas_painter.drawLine(scaled_path[i], scaled_path[i + 1])

                if self.drawing_selection:
                    canvas_painter.drawLine(scaled_path[-1], scaled_path[0])

            elif self.tool == 'rect' and len(self.selection_path) == 2:
                rect = QRect(scaled_path[0], scaled_path[1])
                canvas_painter.drawRect(rect)

            elif self.tool == 'lasso':
                poly = QPolygon(scaled_path)
                canvas_painter.drawPolyline(poly)


    def clear_canvas(self):
        """Clear the canvas to a blank state."""

        self.image.fill(Qt.white)
        self.original_image = self.image.copy()
        self.selected_area = None
        self.selection_start = None
        self.selection_path = []
        self.drawing_selection = False
        self.is_moving_selection = False

        self.update()


    def finalize_selection(self):
        """Finalize the selection and apply any masking if necessary."""
        if self.tool == 'polygon' and len(self.selection_path) > 2:

            if self.selection_path[0] != self.selection_path[-1]:
                self.selection_path.append(self.selection_path[0])

            # Create a mask for the polygon selection
            mask = QImage(self.image.size(), QImage.Format_ARGB32)
            mask.fill(Qt.transparent)

            path = QPainterPath()
            polygon = QPolygonF(self.selection_path)
            path.addPolygon(polygon)

            # Draw the polygon onto the mask
            mask_painter = QPainter(mask)
            mask_painter.fillPath(path, QColor(255, 255, 255, 255))  # White inside the polygon
            mask_painter.end()

            # Apply the mask to the image
            self.selected_area = QImage(self.image.size(), QImage.Format_ARGB32)
            self.selected_area.fill(Qt.transparent)

            painter = QPainter(self.selected_area)
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.drawImage(0, 0, self.image)
            painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            painter.drawImage(0, 0, mask)
            painter.end()

            self.status_message.emit("Polygon selection finalized.")

        elif self.selection_path and self.tool in ['lasso', 'rect']:
            self.selection_path = [
                self.map_to_scaled_image(point) for point in self.selection_path
            ]
            self.extract_selection()

        # Reset path and stop drawing
        self.drawing_selection = False
        self.update()


    def mousePressEvent(self, event):
        """Handle mouse press for both drawing and selection tools."""
        pos = self.map_to_scaled_image(event.pos())

        if pos.x() != -1 and pos.y() != -1:
            self.last_click_position = pos

            if self.tool == 'polygon':
                # Add the clicked point to the selection path
                self.selection_path.append(pos)
                self.drawing_selection = True
                self.update()

            elif self.text_tool_active:
                self.add_text_at_position(pos)

            elif self.current_shape:
                self.selection_start = self.map_to_scaled_image(event.pos())

            elif self.tool == 'rect':
                self.selection_start = pos
                self.selection_path = [pos]

            elif self.tool == 'lasso':
                if not self.drawing_selection:
                    self.selection_path = [pos]

                self.drawing_selection = True

            elif self.tool == 'filter_brush' and event.button() == Qt.LeftButton:
                self.drawing = True
                self.last_point = pos

            elif self.tool in ['pencil', 'brush'] and event.button() == Qt.LeftButton:
                self.drawing = True
                self.last_point = pos

            elif self.tool == 'erase':
                self.drawing = True
                self.last_point = pos

            self.update()


    def mouseMoveEvent(self, event):
        """Handle mouse move for selection tools and drawing."""
        pos = self.map_to_scaled_image(event.pos())  # Use map_to_scaled_image instead

        if self.is_moving_selection and self.selected_area:
            # Move the selection
            self.selection_start = pos - self.selection_offset
            self.update()

        elif self.tool == 'rect' and self.selection_start:
            self.selection_path = [self.selection_start, pos]
            self.update()

        elif self.tool in ['lasso', 'polygon'] and self.drawing_selection:
            self.selection_path.append(pos)
            self.update()

        elif self.tool in ['pencil', 'brush'] and self.drawing:
            # Drawing
            painter = QPainter(self.original_image)

            if self.tool == 'brush':
                # Use QBrush for the texture
                brush = QBrush(self.brush_color, self.brush_texture) if self.brush_texture else QBrush(self.brush_color)
                painter.setBrush(brush)
                painter.setPen(Qt.NoPen)  # Remove border to focus on texture

                # Draw filled circles for a textured effect
                rect = QRect(self.last_point.x() - self.brush_size // 2,
                             self.last_point.y() - self.brush_size // 2,
                             self.brush_size, self.brush_size)
                painter.drawEllipse(rect)

            else:
                # Pencil uses a basic QPen
                pen = QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(self.last_point, pos)
            self.last_point = pos
            painter.end()

            self.image = self.original_image.scaled(
                int(self.original_image.width() * self.current_scale),
                int(self.original_image.height() * self.current_scale),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.update()

        elif self.tool == 'filter_brush' and self.drawing:
            x, y = pos.x(), pos.y()
            brush_radius = self.filter_brush_size // 2
            x_start, x_end = max(0, x - brush_radius), min(self.original_image.width(), x + brush_radius)
            y_start, y_end = max(0, y - brush_radius), min(self.original_image.height(), y + brush_radius)

            image_array = self.qimage_to_numpy(self.original_image)

            # Apply the selected filter to the region
            if self.brush_filter == "gaussian":
                region = image_array[y_start:y_end, x_start:x_end]
                image_array[y_start:y_end, x_start:x_end] = cv2.GaussianBlur(region, (15, 15), 0)

            elif self.brush_filter == "sobel":
                region = image_array[y_start:y_end, x_start:x_end]
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel_combined = cv2.magnitude(sobelx, sobely)
                max_val = np.max(sobel_combined)

                if max_val > 0:
                    sobel_combined = np.uint8(255 * sobel_combined / max_val)

                else:
                    sobel_combined = np.zeros_like(sobel_combined, dtype=np.uint8)

                image_array[y_start:y_end, x_start:x_end] = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)

            elif self.brush_filter == "binary":
                region = image_array[y_start:y_end, x_start:x_end]
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
                binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

                mask = np.any(region != [255, 255, 255], axis=-1)
                region[mask] = binary_colored[mask]
                image_array[y_start:y_end, x_start:x_end] = region

            self.update_image_from_numpy(image_array)
            self.last_point = pos

        elif self.tool == 'erase' and self.drawing:
            painter = QPainter(self.original_image)
            pen = QPen(Qt.white, self.eraser_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, pos)
            self.last_point = pos
            self.image = self.original_image.scaled(
                int(self.original_image.width() * self.current_scale),
                int(self.original_image.height() * self.current_scale),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.update()


    def mouseReleaseEvent(self, event):
        """Handle mouse release for selection tools."""
        if self.is_moving_selection:
            self.is_moving_selection = False
        elif self.tool in ['rect', 'lasso']:
            self.drawing_selection = False
            self.finalize_selection()

        if self.current_shape and self.selection_start:
            end_pos = self.map_to_scaled_image(event.pos())
            painter = QPainter(self.original_image)
            pen = QPen(self.outline_color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QBrush(self.fill_color))

            if self.current_shape == 'rectangle':
                painter.drawRect(QRect(self.selection_start, end_pos))

            elif self.current_shape == 'circle':
                center = (self.selection_start + end_pos) / 2
                radius = abs(self.selection_start.x() - end_pos.x()) // 2
                painter.drawEllipse(center, radius, radius)

            elif self.current_shape == 'line':
                painter.drawLine(self.selection_start, end_pos)

            elif self.current_shape == 'triangle':
                points = [
                    self.selection_start,
                    QPoint(self.selection_start.x(), end_pos.y()),
                    QPoint(end_pos.x(), end_pos.y())
                ]
                painter.drawPolygon(QPolygon(points))

            painter.end()

            self.image = self.original_image.scaled(
                int(self.original_image.width() * self.current_scale),
                int(self.original_image.height() * self.current_scale),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.update()
            self.selection_start = None

        elif self.tool == 'polygon':
            self.drawing_selection = True
            self.update()

        elif self.tool in ['pencil', 'brush', 'erase', 'filter_brush']:
            self.drawing = False

        self.update()


    def cut_selection(self):
        """Cut and clear the selected area, replacing it with white."""
        if not self.selection_path:
            self.status_message.emit("No selection to cut.")
            return

        self.copy_selection()  # Copy selected area

        painter = QPainter(self.original_image)

        if self.tool == 'rect' and len(self.selection_path) == 2:
            rect = self.get_selection_rect()
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.fillRect(rect, Qt.white)

        elif self.tool in ['lasso', 'polygon']:
            mask = QImage(self.original_image.size(), QImage.Format_ARGB32)
            mask.fill(Qt.transparent)
            mask_painter = QPainter(mask)
            path = QPainterPath()
            path.addPolygon(QPolygonF(self.selection_path))
            mask_painter.fillPath(path, Qt.white)
            mask_painter.end()
            painter.drawImage(0, 0, mask)

        painter.end()

        self.image = self.original_image.scaled(
            int(self.original_image.width() * self.current_scale),
            int(self.original_image.height() * self.current_scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.update()
        self.selection_path = []
        self.selected_area = None
        self.status_message.emit("Selection cut.")


    def extract_selection(self):
        """Extract the selected area as a QImage."""
        if not self.selection_path:
            self.status_message.emit("No selection to extract.")
            return

        # Handle lasso/polygon selection
        if self.tool in ['lasso', 'polygon']:
            # Calculate the bounding rectangle of the selection
            polygon = QPolygonF(self.selection_path)
            bounding_rect = polygon.boundingRect()

            self.selected_area = self.original_image.copy(
                int(bounding_rect.left()), int(bounding_rect.top()),
                int(bounding_rect.width()), int(bounding_rect.height())
            )
        elif self.tool == 'rect' and self.selection_start:
            rect = self.get_selection_rect()

            if rect.isValid():
                self.selected_area = self.original_image.copy(rect)

        self.status_message.emit("Selection extracted.")


    def get_selection_rect(self):
        """Calculate the rectangular selection area."""
        if not self.selection_start or not self.selection_path:
            return QRect()

        start = self.selection_start
        end = self.selection_path[-1]
        x1, y1 = min(start.x(), end.x()), min(start.y(), end.y())
        x2, y2 = max(start.x(), end.x()), max(start.y(), end.y())

        return QRect(x1, y1, x2 - x1, y2 - y1)


    def map_to_scaled_image(self, pos):
        """Map widget coordinates to the scaled image coordinates."""
        x = (pos.x() - self.offset_x) / self.current_scale
        y = (pos.y() - self.offset_y) / self.current_scale

        return QPoint(int(x), int(y))


    def map_from_scaled_image(self, pos):
        """Map scaled image coordinates back to widget coordinates."""
        x = int(pos.x() * self.current_scale + self.offset_x)
        y = int(pos.y() * self.current_scale + self.offset_y)

        return QPoint(x, y)


    def event(self, event):
        """Handle gestures."""
        if event.type() == QEvent.Gesture:
            return self.gestureEvent(event)

        return super().event(event)


    def gestureEvent(self, event):
        """Handle gestures like pinch zoom."""
        gesture = event.gesture(Qt.PinchGesture)
        if gesture:
            self.handle_pinch(gesture)
            return True

        return super().event(event)


    def handle_pinch(self, gesture):
        """Zoom in or out based on pinch gesture."""
        if self.original_image.isNull():
            return

        if gesture.state() in (Qt.GestureStarted, Qt.GestureUpdated):
            scale_factor = gesture.scaleFactor()
            new_scale = self.current_scale * scale_factor

            new_scale = max(0.1, min(5.0, new_scale))

            if new_scale != self.current_scale:
                self.current_scale = new_scale
                self.image = self.original_image.scaled(
                    int(self.original_image.width() * self.current_scale),
                    int(self.original_image.height() * self.current_scale),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.update()


    def keyPressEvent(self, event):
        """Handle key press for closing the polygon selection."""
        if self.tool == 'polygon' and len(self.selection_path) > 2:
            if event.key() == Qt.Key_Return:

                # Close the polygon by connecting the last point to the first
                self.selection_path.append(self.selection_path[0])
                self.finalize_selection()
                self.drawing_selection = False
                self.update()
