from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import QRect

def draw_rectangle(canvas, rect: QRect, color: str):
    painter = QPainter(canvas)
    pen = QPen(color)
    painter.setPen(pen)
    painter.drawRect(rect)
