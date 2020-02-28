import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from src.search import Search
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DogCatForm(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = uic.loadUi('./ui/main.ui', self)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.initUI()

    def initUI(self):
        self.ui.setWindowTitle('DogCat')
        self.ui.show()

        self.model_btn.clicked.connect(self.get_model)
        self.img_btn.clicked.connect(self.get_img)
        self.start_btn.clicked.connect(self.search_start)

    def get_model(self):
        self.model_path, i = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.pth)")
        self.model_label.setText(self.model_path.split("/")[-1])

    def get_img(self):
        self.img_path, i = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.gif *.png)")

        print(self.img_path)

        if not self.img_path:
            print("\nNot Selected Image")
            self.img_label.setPixmap(QPixmap(None))
        else:
            print("\nInput Image")
            self.img_label.setPixmap(QPixmap(self.img_path).scaledToWidth(self.img_label.width()))

    def search_start(self):
        # backprop & get gradient
        search = Search(self.model_path,
                        self.img_path)

        true_grad = search.backprop()
        false_grad = search.backprop(inverse=True)

        search.diff_show(true_grad, false_grad)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = DogCatForm()
    app.exec_()
