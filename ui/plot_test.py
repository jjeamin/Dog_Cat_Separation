import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import random


class Form(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = uic.loadUi('./main.ui', self)
        self.initUI()

    def initUI(self):
        self.ui.setWindowTitle('Test')
        self.ui.show()

        self.start_btn.clicked.connect(self.PlotFunc)

    def PlotFunc(self):
        randomNumbers = random.sample(range(0, 10), 10)
        self.ui.widget.canvas.ax.clear()
        self.ui.widget.canvas.ax.plot(randomNumbers)
        self.ui.widget.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Form()
    app.exec_()

