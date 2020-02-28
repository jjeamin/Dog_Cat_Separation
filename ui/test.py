from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()

        figure1 = Figure()
        figure2 = Figure()
        figure3 = Figure()
        figure4 = Figure()
        canvas1 = FigureCanvas(figure1)
        canvas2 = FigureCanvas(figure2)
        canvas3 = FigureCanvas(figure3)
        canvas4 = FigureCanvas(figure4)
        ax1 = figure1.add_subplot(111)
        ax2 = figure2.add_subplot(111)
        ax3 = figure3.add_subplot(111)
        ax4 = figure4.add_subplot(111)

        toolbar1 = NavigationToolbar(canvas1, self)
        toolbar2 = NavigationToolbar(canvas2, self)
        toolbar3 = NavigationToolbar(canvas3, self)
        toolbar4 = NavigationToolbar(canvas4, self)

        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(toolbar1,0,0)
        mainLayout.addWidget(toolbar2,0,1)
        mainLayout.addWidget(toolbar3,2,0)
        mainLayout.addWidget(toolbar4,2,1)
        mainLayout.addWidget(canvas1,1,0)
        mainLayout.addWidget(canvas2,1,1)
        mainLayout.addWidget(canvas3,3,0)
        mainLayout.addWidget(canvas4,3,1)

        self.setLayout(mainLayout)
        self.setWindowTitle("Flow Layout")

if __name__ == '__main__':

    import sys

    app = QtWidgets.QApplication(sys.argv)
    mainWin = Window()
    mainWin.show()
    sys.exit(app.exec_())
