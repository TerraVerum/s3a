from constants import ComponentTypes as ct
import numpy as np
from typing import Dict, List

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
Signal = QtCore.pyqtSignal

# Ensure an application instance is running
app = pg.mkQApp()

class Component(QtCore.QObject):
  _reqdUpdates: Dict[str, list] = {}

  sigCompClicked = Signal(object)

  def __init__(self):
    super().__init__()
    self.vertices = np.array([np.NaN, np.NaN])
    self.uid = -1
    self.devType = ct.N_A
    self.boardText = ''
    self.devText = ''
    self.logo = ''
    self.notes = ''
    self.valid = False

    self._boundPlt = pg.PlotDataItem([np.NaN, np.NaN], pen=pg.mkPen('b', width=5))
    self._boundPlt.curve.setClickable(True)


    self._txtPlt = pg.LabelItem('N/A')

    '''
    IMPORTANT!! Update this list as more properties / plots are added.
    '''
    # Handles update behavior for traits that alter plot information
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
    newSz = self._txtPlt.width(), self._txtPlt.height()
    newPos = np.mean(self.vertices, axis=0)
    self._txtPlt.setPos(newPos[0] - 0.25*newSz[0]/2, newPos[1] - newSz[1]/2)


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
      comp._boundPlt.sigClicked.connect(lambda a: print('caught'))

      # Listen for component signals
      comp.sigCompClicked.connect(self.updateCompPlot)

      self._compList.append(comp)

      self._nextCompId += 1

  def updateCompPlot(self, newComp: Component):
    pass

if __name__== '__main__':
  from PIL import Image
  mw = pg.GraphicsView()
  mainView = pg.ViewBox(lockAspect=True)
  item = pg.ImageItem(np.array(Image.open('../fast.tif')))
  mainView.addItem(item)
  mw.setCentralItem(mainView)
  mw.show()

  c = Component()
  c.vertices = np.random.randint(0,100,size=(10,2))
  c.uid = 5
  c.boardText = 'test'
  mgr = ComponentMgr(mainView)
  mgr.addComps([c])

  app.exec()