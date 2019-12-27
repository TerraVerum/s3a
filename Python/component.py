from constants import ComponentTypes as ct
import numpy as np
from typing import Dict, List

import pyqtgraph as pg

# Ensure an application instance is running
pg.mkQApp()

class Component:
  _reqdUpdates: Dict[str, list] = {}

  '''
  IMPORTANT!! Update this list as more properties / plots are added.
  '''
  # Handles update behavior for traits that alter plot information

  def __init__(self):
    self.vertices = np.array([np.NaN, np.NaN])
    self.uid = -1
    self.devType = ct.N_A
    self.boardText = ''
    self.devText = ''
    self.logo = ''
    self.notes = ''
    self.valid = False

    self._boundPlt = pg.PlotDataItem([np.NaN], [np.NaN])
    self._txtPlt = pg.LabelItem('N/A')

    self._reqdUpdates = {
      'vertices': [self.updateBoundPlt, self.updateTxtPlt],
      'uid'     : [self.updateTxtPlt]
      }

  def __setattr__(self, prop, val):
    super().__setattr__(prop, val)
    # Apply plot updates depending on which variable was changed
    pltUpdateFns = self._reqdUpdates.get(prop, [])
    for fn in pltUpdateFns:
      fn()

  def updateBoundPlt(self):
    self._boundPlt.setData(self.vertices)

  def updateTxtPlt(self):
    self._txtPlt.setText(str(self.uid))
    newPos = np.mean(self.vertices, axis=0)
    # Use points in reverse to convert row-col -> x-y
    self._txtPlt.setPos(newPos[0], newPos[1])


class ComponentMgr():
  _compList: List[Component] = []
  _nextCompId = 0

  _mainImgView: pg.ViewBox
  _compImgView: pg.ViewBox

  def __init__(self, mainImgView: pg.ViewBox):
    self._mainImgView = mainImgView

  def addComps(self, comps: List[Component]):
    for comp in comps:
      comp.uid = self._nextCompId
      self._mainImgView.addItem(comp._boundPlt)
      self._mainImgView.addItem(comp._txtPlt)
      self._nextCompId += 1

if __name__== '__main__':
  c = Component()
  c.vertices = np.random.randint(0,100,size=(10,2))
  c.uid = 5
  c.boardText = 'test'
