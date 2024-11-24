import os
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QInputDialog, QMessageBox, QApplication, QColorDialog, QScrollArea
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QMenuBar, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage, QTransform, QPainter, QKeySequence  # Corrected import
from PyQt5.QtWidgets import QFontDialog
from ui.canvas import Canvas


class PhotoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.canvas = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Photo Editor")
        self.setGeometry(100, 100, 800, 600)

        # Set up scrollable canvas
        self.scroll_area = QScrollArea(self)
        self.canvas = Canvas(self)
        self.canvas.status_message.connect(self.statusBar().showMessage)
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        # Set scroll area as the central widget
        self.setCentralWidget(self.scroll_area)

        # Create keyboard shortcut for paste
        paste_shortcut = QAction("Paste", self)
        paste_shortcut.setShortcut(QKeySequence.Paste)
        paste_shortcut.triggered.connect(lambda: self.canvas.paste_selection(self.canvas.last_click_position))
        self.addAction(paste_shortcut)

        cut_shortcut = QAction("Cut", self)
        cut_shortcut.setShortcut(QKeySequence.Cut)
        cut_shortcut.triggered.connect(self.canvas.cut_selection)
        self.addAction(cut_shortcut)

        copy_shortcut = QAction("Copy", self)
        copy_shortcut.setShortcut(QKeySequence.Copy)
        copy_shortcut.triggered.connect(self.canvas.copy_selection)
        self.addAction(copy_shortcut)

        self.create_menus()

        self.add_keybindings()

        self.show()

    def add_keybindings(self):
        """Set up keybindings for tool selection."""
        self.addAction(self.create_key_action("T", lambda: self.canvas.set_tool('text'), "Text Tool"))
        self.addAction(self.create_key_action("B", lambda: self.canvas.set_tool('brush'), "Brush Tool"))
        self.addAction(self.create_key_action("P", lambda: self.canvas.set_tool('pencil'), "Pencil Tool"))
        self.addAction(self.create_key_action("S", self.open_selection_menu, "Select Tool Menu"))
        self.addAction(self.create_key_action("E", lambda: self.canvas.set_tool('erase'), "Eraser Tool"))
        self.addAction(self.create_key_action("R", self.reset_zoom, "Reset Zoom"))

    def create_key_action(self, key, action_func, description):
        """Helper to create a QAction with a shortcut."""
        action = QAction(self)
        action.setShortcut(key)
        action.triggered.connect(action_func)
        action.setText(description)

        return action

    def open_selection_menu(self):
        """Open a selection dialog to choose the selection tool."""
        options = ["Rectangular Selection", "Free-Form Selection (Lasso)", "Polygon Selection"]
        tool, ok = QInputDialog.getItem(self, "Select Selection Tool", "Choose a selection tool:", options, 0, False)
        if ok and tool:

            if tool == "Rectangular Selection":
                self.canvas.set_tool('rect')

            elif tool == "Free-Form Selection (Lasso)":
                self.canvas.set_tool('lasso')

            elif tool == "Polygon Selection":
                self.canvas.set_tool('polygon')

    def crop_image(self):
        """Crop the center of the canvas image."""
        if self.canvas.image.isNull():
            QMessageBox.warning(self, "Error", "No image to crop.")

            return

        image = self.canvas.image
        width = image.width()
        height = image.height()

        crop_size = min(width, height)
        x = (width - crop_size) // 2
        y = (height - crop_size) // 2

        cropped = image.copy(x, y, crop_size, crop_size)
        self.canvas.image = cropped
        self.canvas.update()
        self.statusBar().showMessage("Image cropped to center.")

    def resize_image(self):
        """Resize the canvas without scaling the existing image."""
        if self.canvas.image.isNull():
            QMessageBox.warning(self, "Error", "No image to resize.")
            return

        new_width, ok1 = QInputDialog.getInt(self, "Resize", "Enter new width:", min=1, max=5000)
        if not ok1:
            return

        new_height, ok2 = QInputDialog.getInt(self, "Resize", "Enter new height:", min=1, max=5000)
        if not ok2:
            return

        new_image = QImage(new_width, new_height, QImage.Format_RGB32)
        new_image.fill(Qt.white)

        painter = QPainter(new_image)
        x_offset = (new_width - self.canvas.image.width()) // 2
        y_offset = (new_height - self.canvas.image.height()) // 2
        painter.drawImage(x_offset, y_offset, self.canvas.image)
        painter.end()

        self.canvas.setFixedSize(new_width, new_height)
        self.canvas.image = new_image
        self.canvas.update()
        self.statusBar().showMessage(f"Canvas resized to {new_width}x{new_height}.")

    def rotate_image_90(self):
        """Rotate the canvas image 90 degrees clockwise."""
        if self.canvas.image.isNull():
            QMessageBox.warning(self, "Error", "No image to rotate.")
            return

        transform = QTransform().rotate(90)
        self.canvas.image = self.canvas.image.transformed(transform)
        self.canvas.update()

    def rotate_image_90_cc(self):
        """Rotate the canvas image 90 degrees counterclockwise."""
        if self.canvas.image.isNull():
            QMessageBox.warning(self, "Error", "No image to rotate.")
            return

        transform = QTransform().rotate(-90)
        self.canvas.image = self.canvas.image.transformed(transform)
        self.canvas.update()

    def create_menus(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # File Menu
        file_menu = menu_bar.addMenu("File")

        # New
        new_action = QAction("New", self)
        new_action.triggered.connect(self.new_file)
        file_menu.addAction(new_action)

        # Open
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # Save
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        # Save As
        save_as_action = QAction("Save As", self)
        save_as_action.triggered.connect(self.save_file_as)
        file_menu.addAction(save_as_action)

        # Properties
        properties_action = QAction("Properties", self)
        properties_action.triggered.connect(self.show_properties)
        file_menu.addAction(properties_action)

        # Quit
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        quit_action.setMenuRole(QAction.QuitRole)  # Explicitly set menu role for macOS
        file_menu.addAction(quit_action)

        # Image Menu
        image_menu = menu_bar.addMenu("Image")
        crop_action = QAction("Crop", self)
        crop_action.triggered.connect(self.crop_image)
        image_menu.addAction(crop_action)

        resize_action = QAction("Resize", self)
        resize_action.triggered.connect(self.resize_image)
        image_menu.addAction(resize_action)

        rotate_action = QAction("Rotate 90° Clockwise", self)
        rotate_action.triggered.connect(self.rotate_image_90)
        image_menu.addAction(rotate_action)

        rotate_cc_action = QAction("Rotate 90° Counterclockwise", self)
        rotate_cc_action.triggered.connect(self.rotate_image_90_cc)
        image_menu.addAction(rotate_cc_action)

        # Clipboard Menu
        clipboard_menu = self.menuBar().addMenu("Clipboard")

        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.copy_image)
        clipboard_menu.addAction(copy_action)

        paste_action = QAction("Paste", self)
        paste_action.triggered.connect(self.paste_image)
        clipboard_menu.addAction(paste_action)

        cut_action = QAction("Cut", self)
        cut_action.triggered.connect(self.cut_image)
        clipboard_menu.addAction(cut_action)

        image_menu = self.menuBar().addMenu("Image")

        # Flip Vertical
        flip_vertical_action = QAction("Flip Vertical", self)
        flip_vertical_action.triggered.connect(self.flip_vertical)
        image_menu.addAction(flip_vertical_action)

        # Flip Horizontal
        flip_horizontal_action = QAction("Flip Horizontal", self)
        flip_horizontal_action.triggered.connect(self.flip_horizontal)
        image_menu.addAction(flip_horizontal_action)

        # Shapes Menu
        shapes_menu = menu_bar.addMenu("Shapes")

        # List of Shapes
        list_of_shapes_action = QAction("List of Shapes", self)
        list_of_shapes_action.triggered.connect(self.select_shape)
        shapes_menu.addAction(list_of_shapes_action)

        # Outline Color
        outline_color_action = QAction("Outline Color", self)
        outline_color_action.triggered.connect(self.set_outline_color)
        shapes_menu.addAction(outline_color_action)

        # Fill Color
        fill_color_action = QAction("Fill Color", self)
        fill_color_action.triggered.connect(self.set_fill_color)
        shapes_menu.addAction(fill_color_action)

        reset_colors_action = QAction("Reset Colors", self)
        reset_colors_action.triggered.connect(self.reset_colors)
        shapes_menu.addAction(reset_colors_action)

        # Tools Menu
        tools_menu = self.menuBar().addMenu("Tools")

        filter_brush_size_action = QAction("Filter Brush Size", self)
        filter_brush_size_action.triggered.connect(self.set_filter_brush_size)
        tools_menu.addAction(filter_brush_size_action)

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        tools_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        tools_menu.addAction(zoom_out_action)

        brush_action = QAction("Brush", self)
        brush_action.triggered.connect(lambda: self.canvas.set_tool('brush'))
        tools_menu.addAction(brush_action)

        brush_settings_menu = tools_menu.addMenu("Brush Settings")


        # Brush Texture
        brush_textures_menu = brush_settings_menu.addMenu("Brush Textures")
        for texture_name in ["None", "Dots", "Stripes", "Checkerboard", "Horizontal Lines", "Vertical Lines"]:
            texture_action = QAction(texture_name, self)
            texture_action.triggered.connect(lambda checked, t=texture_name: self.canvas.set_brush_texture(t))
            brush_textures_menu.addAction(texture_action)

        # Brush Color and Size
        brush_color_action = QAction("Brush Color", self)
        brush_color_action.triggered.connect(self.set_brush_color)
        tools_menu.addAction(brush_color_action)

        brush_size_action = QAction("Brush Size", self)
        brush_size_action.triggered.connect(self.set_brush_size)
        tools_menu.addAction(brush_size_action)

        draw_action = QAction("Draw", self)
        draw_action.triggered.connect(lambda: self.canvas.set_tool('pencil'))
        tools_menu.addAction(draw_action)

        erase_action = QAction("Erase", self)
        erase_action.triggered.connect(lambda: self.canvas.set_tool('erase'))
        tools_menu.addAction(erase_action)

        # Eraser Size
        eraser_size_action = QAction("Eraser Size", self)
        eraser_size_action.triggered.connect(self.set_eraser_size)
        tools_menu.addAction(eraser_size_action)

        text_tool_action = QAction("Text Tool", self)
        text_tool_action.triggered.connect(self.enable_text_tool)
        tools_menu.addAction(text_tool_action)

        # Font Picker
        font_action = QAction("Select Font", self)
        font_action.triggered.connect(self.select_text_font)
        tools_menu.addAction(font_action)

        # Text Color Picker
        text_color_action = QAction("Select Text Color", self)
        text_color_action.triggered.connect(self.select_text_color)
        tools_menu.addAction(text_color_action)

        # Other tools like selection
        select_menu = tools_menu.addMenu("Select")
        rect_action = QAction("Rectangular Selection", self)
        rect_action.triggered.connect(lambda: self.canvas.set_tool('rect'))
        select_menu.addAction(rect_action)

        lasso_action = QAction("Free-Form Selection (Lasso)", self)
        lasso_action.triggered.connect(lambda: self.canvas.set_tool('lasso'))
        select_menu.addAction(lasso_action)

        polygon_action = QAction("Polygon Selection", self)
        polygon_action.triggered.connect(lambda: self.canvas.set_tool('polygon'))
        select_menu.addAction(polygon_action)

        # Tools Menu: Selection Manipulation
        selection_tools_menu = tools_menu.addMenu("Selection Tools")

        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.canvas.copy_selection)
        selection_tools_menu.addAction(copy_action)

        paste_action = QAction("Paste", self)
        paste_action.triggered.connect(self.canvas.paste_selection)
        selection_tools_menu.addAction(paste_action)

        cut_action = QAction("Cut", self)
        cut_action.triggered.connect(self.canvas.cut_selection)
        selection_tools_menu.addAction(cut_action)

        # Reset Zoom
        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.triggered.connect(self.reset_zoom)
        tools_menu.addAction(reset_zoom_action)

        # Zoom In
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        tools_menu.addAction(zoom_in_action)

        # Zoom Out
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        tools_menu.addAction(zoom_out_action)

        # Colors Menu
        colors_menu = menu_bar.addMenu("Colors")

        brush_color_action = QAction("Brush Color", self)
        brush_color_action.triggered.connect(self.set_brush_color)
        colors_menu.addAction(brush_color_action)

        filter_menu = self.menuBar().addMenu("Filters")

        # Add global filters
        gaussian_action = QAction("Gaussian Filter", self)
        gaussian_action.triggered.connect(self.canvas.apply_gaussian_filter)
        filter_menu.addAction(gaussian_action)

        sobel_action = QAction("Sobel Filter", self)
        sobel_action.triggered.connect(self.canvas.apply_sobel_filter)
        filter_menu.addAction(sobel_action)

        binary_action = QAction("Binary Filter", self)
        binary_action.triggered.connect(self.canvas.apply_binary_filter)
        filter_menu.addAction(binary_action)

        histogram_action = QAction("Histogram Thresholding", self)
        histogram_action.triggered.connect(self.canvas.apply_histogram_thresholding)
        filter_menu.addAction(histogram_action)

        brush_filter_menu = filter_menu.addMenu("Brush Filters")

        filter_menu.addSeparator()  # Optional: Add a separator between global and brush filters

        filter_menu.addMenu(brush_filter_menu)  # Add the Brush Filters submenu under Filters

        brush_gaussian_action = QAction("Gaussian Filter Brush", self)
        brush_gaussian_action.triggered.connect(lambda: [self.canvas.set_tool('filter_brush'),
                                                         self.canvas.set_brush_filter("gaussian")])
        brush_filter_menu.addAction(brush_gaussian_action)

        brush_sobel_action = QAction("Sobel Filter Brush", self)
        brush_sobel_action.triggered.connect(lambda: [self.canvas.set_tool('filter_brush'),
                                                      self.canvas.set_brush_filter("sobel")])
        brush_filter_menu.addAction(brush_sobel_action)

        brush_binary_action = QAction("Binary Filter Brush", self)
        brush_binary_action.triggered.connect(lambda: [self.canvas.set_tool('filter_brush'),
                                                       self.canvas.set_brush_filter("binary")])
        brush_filter_menu.addAction(brush_binary_action)

    def reset_colors(self):
        """Reset the fill and outline colors to default."""
        self.canvas.reset_colors()
        self.statusBar().showMessage("Fill and outline colors reset to default.")

    def select_shape(self):
        """Open a dialog to select a shape."""
        shapes = ["Rectangle", "Circle", "Triangle", "Line"]
        shape, ok = QInputDialog.getItem(self, "Select Shape", "Choose a shape:", shapes, 0, False)
        if ok and shape:
            self.canvas.set_tool(shape.lower())
            self.statusBar().showMessage(f"Selected shape: {shape}")

    def set_outline_color(self):
        """Set the outline color for shapes."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.canvas.set_outline_color(color)
            self.statusBar().showMessage(f"Outline color set to: {color.name()}")

    def set_fill_color(self):
        """Set the fill color for shapes."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.canvas.set_fill_color(color)
            self.statusBar().showMessage(f"Fill color set to: {color.name()}")

    def set_filter_brush_size(self):
        """Set the filter brush size."""
        size, ok = QInputDialog.getInt(self, "Filter Brush Size", "Enter filter brush size:", min=1, max=50)

        if ok:
            self.canvas.set_filter_brush_size(size)

    def new_file(self):
        """Clear the canvas for a new file."""
        self.canvas.clear_canvas()
        self.statusBar().showMessage("New file created.")

    def enable_text_tool(self):
        """Enable the text tool in the canvas."""
        self.canvas.enable_text_tool()

    def select_text_font(self):
        """Open a font dialog to select the font for the text tool."""
        font, ok = QFontDialog.getFont(self.canvas.text_font, self)

        if ok:
            self.canvas.set_text_font(font)

    def select_text_color(self):
        """Open a color dialog to select the color for the text tool."""
        color = QColorDialog.getColor(self.canvas.text_color, self)

        if color.isValid():
            self.canvas.set_text_color(color)

    def set_brush_size(self):
        """Set the brush size."""
        size, ok = QInputDialog.getInt(self, "Brush Size", "Enter brush size:", min=1, max=50)

        if ok:
            self.canvas.set_brush_size(size)

    def set_eraser_size(self):
        """Set the eraser size."""
        size, ok = QInputDialog.getInt(self, "Eraser Size", "Enter eraser size:", min=1, max=50)

        if ok:
            self.canvas.set_eraser_size(size)

    def set_brush_color(self):
        """Set the brush color."""
        color = QColorDialog.getColor()

        if color.isValid():
            self.canvas.set_brush_color(color.name())

    def save_file(self):
        """Save the current image."""
        if self.current_file:
            self.save_image_to_file(self.current_file)
            self.statusBar().showMessage(f"File updated: {self.current_file}")

        else:
            self.save_file_as()

    def save_file_as(self):
        """Save the current image to a new file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File As", "", "Images (*.png *.jpg *.bmp)", options=options
        )

        if file_path:
            self.current_file = file_path
            self.save_image_to_file(file_path)

    def save_image_to_file(self, file_path):
        """Save the canvas image to a file."""
        if self.canvas.image.isNull():
            QMessageBox.warning(self, "Error", "No image to save.")
            return

        if self.canvas.image.save(file_path):
            self.statusBar().showMessage(f"File saved: {file_path}")

        else:
            QMessageBox.critical(self, "Error", "Failed to save the file.")

    def show_properties(self):
        """Display properties of the current canvas image."""
        if self.canvas.image.isNull():
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        width = self.canvas.image.width()
        height = self.canvas.image.height()
        color_depth = self.canvas.image.depth()

        QMessageBox.information(
            self,
            "Image Properties",
            f"Width: {width} px\nHeight: {height} px\nColor Depth: {color_depth} bits",
        )

    def open_file(self):
        """Open an image file and display it on the canvas."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options
        )

        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path):
        """Load the image from the file into the canvas."""
        image = QImage(file_path)
        if image.isNull():
            QMessageBox.warning(self, "Error", "Failed to load the image.")
            return

        self.canvas.original_image = image.copy()
        self.canvas.current_scale = 1.0

        self.canvas.image = self.canvas.original_image.scaled(
            self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.canvas.setFixedSize(image.width(), image.height())
        self.canvas.update()
        self.statusBar().showMessage(f"Loaded image: {file_path}")

    def copy_image(self):
        """Copy the current canvas image to the clipboard."""
        if self.canvas.image.isNull():
            QMessageBox.warning(self, "Error", "No image to copy.")
            return

        clipboard = QApplication.clipboard()
        clipboard.setImage(self.canvas.image)
        self.statusBar().showMessage("Image copied to clipboard.")

    def paste_image(self):
        """Paste an image from the clipboard onto the canvas."""
        clipboard = QApplication.clipboard()
        image = clipboard.image()

        if image.isNull():
            QMessageBox.warning(self, "Error", "Clipboard is empty or doesn't contain an image.")
            return

        self.canvas.image = image.scaled(self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.canvas.update()
        self.statusBar().showMessage("Image pasted from clipboard.")

    def cut_image(self):
        """Cut the current canvas image (copy and clear)."""
        if self.canvas.image.isNull():
            QMessageBox.warning(self, "Error", "No image to cut.")
            return

        self.copy_image()
        self.canvas.image.fill(Qt.white)
        self.canvas.update()
        self.statusBar().showMessage("Image cut and cleared.")

    def flip_vertical(self):
        """Flip the canvas image vertically."""
        if self.canvas.image.isNull():
            QMessageBox.warning(self, "Error", "No image to flip.")
            return

        self.canvas.image = self.canvas.image.mirrored(vertical=True, horizontal=False)
        self.canvas.update()

    def flip_horizontal(self):
        """Flip the canvas image horizontally."""
        if self.canvas.image.isNull():
            QMessageBox.warning(self, "Error", "No image to flip.")
            return

        self.canvas.image = self.canvas.image.mirrored(vertical=False, horizontal=True)
        self.canvas.update()

    def zoom_in(self):
        """Trigger zoom in on the canvas."""
        self.canvas.zoom_in()

    def zoom_out(self):
        """Trigger zoom out on the canvas."""
        self.canvas.zoom_out()

    def pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.statusBar().showMessage(f"Color selected: {color.name()}")

    def reset_zoom(self):
        """Reset the zoom level to default (1.0x) and fit the image to the canvas."""
        if self.canvas.image.isNull():
            return

        self.canvas.current_scale = 1.0

        self.canvas.image = self.canvas.image.scaled(
            self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.canvas.update()
        self.statusBar().showMessage("Zoom level reset.")