from PyQt5.QtWidgets import QApplication
from ui.menus import PhotoEditor

import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = PhotoEditor()
    sys.exit(app.exec_())
