from constants import ComponentTypes
import numpy as np
from typing import Dict, List, Union

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot

import graphicshelpers
from SchemeEditor import SchemeEditor
from constants import SchemeValues as SV

# Ensure an application instance is running
app = pg.mkQApp()

class Component(QtCore.QObject):
  _reqdUpdates: Dict[str, list] = {}

  sigCompClicked = Signal(object)

  scheme = SchemeEditor()

  def __init__(self):
    super().__init__()
    self.instanceId = -1
    self.vertices = np.array([np.NaN, np.NaN])
    self.deviceType = ComponentTypes.N_A
    self.boardText = ''
    self.deviceText = ''
    self.logo = ''
    self.notes = ''
    self.validated = False

    # Shorthand for convenience
    penClr, penWidth, txtSize = Component.scheme.getCompProps(
      (SV.boundaryColor, SV.boundaryWidth, SV.idFontSize))

    self._boundPlt = pg.PlotDataItem([np.NaN, np.NaN], pen=pg.mkPen(color=penClr, width=penWidth))
    self._txtPlt = graphicshelpers.ClickableTextItem('N/A')
    curFont = self._txtPlt.textItem.font()
    curFont.setPointSize(txtSize)
    self._txtPlt.setFont(curFont)
    self._txtPlt.sigClicked.connect(self._rethrowItemClicked)

    '''
    IMPORTANT!! Update this list as more properties / plots are added.
    '''
    # Handles update behavior for traits that alter plot information
    self._reqdUpdates = {
      'vertices'   : [self._updateBoundPlt, self._updateTxtPlt],
      'instanceId' : [self._updateTxtPlt],
      'validated'  : [self._updateTxtPlt]
      }

  def __setattr__(self, prop, val):
    super().__setattr__(prop, val)
    # Apply plot updates depending on which variable was changed
    pltUpdateFns = self._reqdUpdates.get(prop, [])
    for fn in pltUpdateFns:
      fn()

  @staticmethod
  def setScheme(scheme: SchemeEditor):
    '''
    Responsible for customizing display aspects of each component. Only one scheme
    should exist per application, so this is a static method
    '''
    Component.scheme = scheme

  @Slot()
  def _rethrowItemClicked(self):
    self.sigCompClicked.emit(self)

  def _updateBoundPlt(self):
    self._boundPlt.setData(self.vertices)

  def _updateTxtPlt(self):
    schemeClrProp = SV.nonValidIdColor
    if self.validated:
      schemeClrProp = SV.validIdColor
    txtClr = Component.scheme.getCompProps(schemeClrProp)
    self._txtPlt.setText(str(self.instanceId))
    self._txtPlt.setColor(txtClr)
    #newSz = self._txtPlt.width(), self._txtPlt.height()
    newPos = np.mean(self.vertices, axis=0)
    self._txtPlt.setPos(newPos[0], newPos[1])

class ComponentMgr(QtCore.QObject):
  sigCompClicked = Signal(object)

  _compList: List[Component] = []
  _nextCompId = 0

  _mainImgArea: pg.ViewBox
  _compImgView: pg.ViewBox

  def __init__(self, mainImgArea: pg.GraphicsWidget):
    super().__init__()
    self._mainImgArea = mainImgArea

  def addComps(self, comps: List[Component]):
    for comp in comps:
      comp.instanceId = self._nextCompId
      self._mainImgArea.addItem(comp._boundPlt)
      self._mainImgArea.addItem(comp._txtPlt)

      # Listen for component signals and rethrow them
      comp.sigCompClicked.connect(self._rethrowCompClick)

      self._compList.append(comp)

      self._nextCompId += 1

  def rmComps(self, idList: Union[List[int], str] = 'all'):
    newCompList = []
    # Next ID will change depending on which components are deleted
    nextId = 0
    if idList == 'all':
      idList = [obj.instanceId for obj in self._compList]
    # Take each requested component off the main image and remove from list
    for ii, comp in enumerate(self._compList):
      if comp.instanceId in idList:
        [self._mainImgArea.removeItem(plt) for plt in (comp._boundPlt, comp._txtPlt)]
      else:
        newCompList.append(comp)
        if comp.instanceId >= nextId:
          nextId = comp.instanceId + 1
    self._compList = newCompList


  @Slot(object)
  def _rethrowCompClick(self, comp:Component):
    self.sigCompClicked.emit(comp)

if __name__== '__main__':
  from PIL import Image
  mw = pg.PlotWindow()
  item = pg.ImageItem(np.array(Image.open('../fast.tif')))
  mw.addItem(item)
  mw.setAspectLocked(True)
  mw.show()

  c = Component()
  c.vertices = np.random.randint(0,100,size=(10,2))
  c.instanceId = 5
  c.boardText = 'test'
  mgr = ComponentMgr(mw)
  mgr.addComps([c])

  app.exec()