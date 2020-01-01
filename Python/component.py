from constants import ComponentTypes
import numpy as np
from typing import Dict, List, Union

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot

from SchemeEditor import SchemeEditor
from constants import SchemeValues as SV
from ABGraphics.clickables import ClickableTextItem
from ABGraphics.regions import VertexRegion

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

    # Using VertexRegion is significantly faster than PlotDataItem
    #self._boundPlt = pg.PlotDataItem([np.NaN, np.NaN], pen=pg.mkPen(color=penClr, width=penWidth))
    self._boundPlt = VertexRegion()
    self._txtPlt = ClickableTextItem('N/A')
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
    #self._boundPlt.setData(self.vertices)
    self._boundPlt.updateVertices(self.vertices.copy())

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

  def __init__(self, mainImgArea: pg.GraphicsWidget, mainImgItem: pg.ImageItem):
    super().__init__()
    self._mainImgArea = mainImgArea
    self._mainImgItem = mainImgItem

  def addComps(self, comps: List[Component]):
    for comp in comps:
      comp.instanceId = self._nextCompId
      self._mainImgArea.addItem(comp._boundPlt)
      self._mainImgArea.addItem(comp._txtPlt)

      # Listen for component signals and rethrow them
      comp.sigCompClicked.connect(self._rethrowCompClick)

      self._compList.append(comp)

      self._nextCompId += 1

  def rmComps(self, idList: Union[np.array, str] = 'all'):
    # Use numpy array so size is preallocated
    newCompList = np.empty(len(self._compList), dtype=object)
    validIdxs = np.zeros(newCompList.size, dtype=bool)
    # Next ID will change depending on which components are deleted

    '''
    Removing items one by one is significantly more intensive than adding them.
    So, it is more effective to clear the whole list and re-add components to keep
    '''
    if idList == 'all':
      idList = [obj.instanceId for obj in self._compList]
    idList = np.array(idList)
    for ii, comp in enumerate(self._compList):
      if not np.any(comp.instanceId == idList):
        validIdxs[ii] = True
    newCompList = newCompList[validIdxs]
    self._mainImgArea.clear()
    self._mainImgArea.addItem(self._mainImgItem)
    newIdList = np.array([comp.instanceId for comp in newCompList])
    if newIdList.size > 0:
      self._nextCompId = np.max(newIdList) + 1
    else:
      self._nextCompId = 0
    self.addComps(list(newCompList))


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
  mgr = ComponentMgr(mw, item)
  mgr.addComps([c])

  app.exec()