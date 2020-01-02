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

  sigCompClicked = Signal()
  sigVertsChanged = Signal()

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

    self._txtPlt = ClickableTextItem('N/A')
    self._txtPlt.sigClicked.connect(self._rethrowItemClicked)

    '''
    IMPORTANT!! Update this list as more properties / plots are added.
    '''
    # Handles update behavior for traits that alter plot information
    self._reqdUpdates = {
      'vertices'   : [self.sigVertsChanged.emit, self._updateTxtPlt],
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
    # Pass scheme to ClickableTextItem
    ClickableTextItem.setScheme(scheme)

  @Slot()
  def _rethrowItemClicked(self):
    self.sigCompClicked.emit()

  def _updateTxtPlt(self):
    self._txtPlt.updateText(str(self.instanceId), self.validated)
    newPos = np.mean(self.vertices, axis=0)
    self._txtPlt.setPos(newPos[0], newPos[1])

class ComponentMgr(QtCore.QObject):
  sigCompClicked = Signal(object)

  _compList: List[Component] = []
  _nextCompId = 0

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
      comp.sigVertsChanged.connect(self._reflectVertsChanged)

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

  @Slot()
  def _rethrowCompClick(self):
    comp: Component = self.sender()
    self.sigCompClicked.emit(comp)

  @Slot()
  def _updateCompBoundsPlt(self):
    self._compBounds.resetRegionList()

  @Slot()
  def _reflectVertsChanged(self):
    comp: Component = self.sender()
    self._compBounds.setRegions(comp.instanceId, comp.vertices)

  @staticmethod
  def setScheme(scheme: SchemeEditor):
    # Pass this scheme to the MultiRegionPlot
    MultiRegionPlot.setScheme(scheme)

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