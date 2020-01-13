import pyqtgraph as pg

from sys import path
path.append('..')
from ABGraphics.regions import VertexRegion
from ABGraphics.table import CompTableView, CompTableModel
from component import *

import numpy as np

if __name__ == '__main__':
  app = pg.mkQApp()
  comps = []
  mgr = ComponentMgr()
  for ii in range(5):
    newcomp = Component()
    newcomp.vertices = np.random.randint(10,size=(5,2))
    comps.append(newcomp)
  mgr.addComps(comps)
  model = CompTableModel(mgr)
  t = CompTableView(model.colTitles)
  t.setModel(model)
  t.show()
  app.exec()