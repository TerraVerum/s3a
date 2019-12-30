import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavToolbar
from PyQt5 import uic, QtWidgets
#from PySide2 import QtWidgets

from os import path
from PIL import Image

import numpy as np
import sys


class MyWindow(QtWidgets.QMainWindow):
  def __init__(self):
    super(MyWindow, self).__init__()
    uic.loadUi('testMpl.ui', self) 
    # test data
    data = np.array([0.7,0.7,0.7,0.8,0.9,0.9,1.5,1.5,1.5,1.5])        
    fig, ax1 = plt.subplots()
    bins = np.arange(0.6, 1.62, 0.02)
    n1, bins1, patches1 = ax1.hist(data, bins, alpha=0.6, density=False, cumulative=False)
    # plot
    self.plotWidget = FigureCanvas(fig)
    #lay = QtWidgets.QVBoxLayout(self.content_plot)  
    #lay.setContentsMargins(0, 0, 0, 0)      
    #lay.addWidget(self.plotWidget)

class ImageROISuite(QtWidgets.QWidget):
  def __init__(self, *args):
    super().__init__()
    lay = QtWidgets.QVBoxLayout(self)
    self.setContentsMargins(0,0,0,0)
    fig, ax = plt.subplots()
    ax.imshow(np.array(Image.open('../../fast.tif')))
    canvas = FigureCanvas(fig)
    lay.addWidget(canvas)

if __name__ == '__main__':
  app = QtWidgets.QApplication([])
  window = MyWindow()
  window.show()
  sys.exit(app.exec_())  