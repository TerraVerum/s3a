import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import sys
# Makes sure that when the folder is run as a module, the app exists in the outermost
# scope of the application
appInst = QtWidgets.QApplication.instance()
if appInst is None:
  appInst = QtWidgets.QApplication(sys.argv)
# Now that the app was created with sys args, populate pg instance
pg.mkQApp()
# Change names so they can be labeled on export
from .Annotator import Annotator
from .tablemodel import makeCompDf