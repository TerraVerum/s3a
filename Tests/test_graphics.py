from sys import path

import pyqtgraph as pg

path.append('..')
from .ABGraphics.tableview import CompTableView
from .tablemodel import ComponentMgr, makeCompDf
from .constants import TEMPLATE_COMP as TC

import numpy as np

if __name__ == '__main__':
  app = pg.mkQApp()
  comps = makeCompDf(5)
  comps = comps.set_index(np.arange(5, dtype=int))
  mgr = ComponentMgr()
  for ii in range(len(comps)):
    comps.loc[ii, TC.VERTICES.name] = [np.random.randint(10,size=(5,2))]
  mgr.addComps(comps, addtype='new')
  t = CompTableView()
  t.setModel(mgr)
  t.show()
  app.exec()