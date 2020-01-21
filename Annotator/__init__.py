import pyqtgraph as pg
# Makes sure that when the folder is run as a module, the app exists in the outermost
# scope of the application
appInst = pg.mkQApp()

# Change names so they can be labeled on export
from .Annotator import Annotator
from .tablemodel import makeCompDf