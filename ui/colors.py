from PyQt5.QtWidgets import QColorDialog

def choose_color():
    color = QColorDialog.getColor()
    if color.isValid():
        return color.name()
    return None
