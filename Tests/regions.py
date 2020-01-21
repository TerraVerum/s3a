import pyqtgraph as pg
import numpy as np

from .ABGraphics.regions import VertexRegion

if __name__ == '__main__':
  app = pg.mkQApp()
  vr = VertexRegion()
  vr.updateRegion(np.array([[2,1], [5,5], [5,1]]))
  mw = pg.PlotWindow()
  mw.addItem(vr)
  mw.show()
  app.exec()