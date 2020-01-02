import numpy as np
from typing import Dict, List, Union
from functools import partial

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot

from constants import ComponentTypes
from SchemeEditor import SchemeEditor
from constants import SchemeValues as SV
from ABGraphics.clickables import ClickableTextItem
from ABGraphics.regions import VertexRegion, MultiRegionPlot

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
    newPos = np.mean(self.vertices, axis=0)
    self._txtPlt.setPos(newPos[0], newPos[1])

class ComponentMgr(QtCore.QObject):
  sigCompClicked = Signal(object)

  _compList: List[Component] = []
  _nextCompId = 0

  _compImgView: pg.ViewBox

  def __init__(self, mainImgArea: pg.GraphicsWidget, mainImgItem: pg.ImageItem):
    super().__init__()
    self._mainImgArea = mainImgArea
    self._mainImgItem = mainImgItem
    self._compBounds = MultiRegionPlot()
    # Update comp image on main image change
    mainImgItem.sigImageChanged.connect(self._updateCompBoundsPlt)
    self._mainImgArea.addItem(self._compBounds)

  def addComps(self, comps: List[Component]):
    # Preallocate list size since we know its size in advance
    newVerts = [None]*len(comps)
    newIds = np.arange(self._nextCompId, self._nextCompId + len(comps), dtype=int)
    for ii, comp in enumerate(comps):
      newVerts[ii] = comp.vertices
      comp.instanceId = newIds[ii]
      self._mainImgArea.addItem(comp._txtPlt)

      # Listen for component signals and rethrow them
      comp.sigCompClicked.connect(self._rethrowCompClick)

    self._nextCompId += newIds[-1] + 1
    self._compList.extend(comps)
    self._compBounds.setRegions(newIds, newVerts)

  def rmComps(self, idsToRemove: Union[np.array, str] = 'all'):
    # Use numpy array so size is preallocated
    curComps = np.array(self._compList)
    # Generate ID list
    existingCompIds = [obj.instanceId for obj in self._compList]
    if idsToRemove == 'all':
      idsToRemove = existingCompIds
    idsToRemove = np.array(idsToRemove)
    existingCompIds = np.array(existingCompIds)

    tfRmIdx = np.isin(existingCompIds, idsToRemove)
    rmCompIdxs = np.nonzero(tfRmIdx)[0]
    keepCompIdxs = np.nonzero(np.invert(tfRmIdx))[0]
    for idx in rmCompIdxs:
      curComp: Component = self._compList[idx]
      self._mainImgArea.removeItem(curComp._txtPlt)

    # Reset manager's component list
    self._compList = list(curComps[keepCompIdxs])

    # Update bound plot
    self._compBounds.setRegions(idsToRemove, [[] for id in idsToRemove])

    # Determine next ID for new components
    self._nextCompId = 0
    if not np.all(tfRmIdx):
      self._nextCompId = np.max(existingCompIds[keepCompIdxs]) + 1

  @Slot(object)
  def _rethrowCompClick(self, comp:Component):
    self.sigCompClicked.emit(comp)

  @Slot()
  def _updateCompBoundsPlt(self):
    self._compBounds.resetRegionList()

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