# Required to avoid cyclic dependency from CompTable annotation
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from typing import Dict, List, Union
from functools import partial
import warnings

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from constants import ComponentTypes
from ABGraphics.parameditors import SchemeEditor
from constants import SchemeValues as SV
from ABGraphics.clickables import ClickableTextItem
from ABGraphics.regions import MultiRegionPlot
from ABGraphics import CompTable

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot
# Ensure an application instance is running
app = pg.mkQApp()

class Component(QtCore.QObject):
  # TODO:
  # Since no fields will be added to this class, and potentially
  # thousands of components may be registered per image, utilize
  # 'slots' for memory efficiency
  #__slots__ = ['_reqdUpdates', 'sigCompClicked', 'sigVertsChanged',
               #'scheme', 'instanceId', 'vertices', 'deviceType',...]
  _reqdUpdates: Dict[str, list] = {}

  sigIdClicked = Signal()
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
    self._txtPlt.sigClicked.connect(self.sigIdClicked.emit)

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

  def _updateTxtPlt(self):
    # It is OK for NaN mean values, since this will hide the text
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      newPos = np.mean(self.vertices, axis=0)
      self._txtPlt.setPos(newPos[0], newPos[1])
    self._txtPlt.updateText(str(self.instanceId), self.validated)

class ComponentMgr(QtCore.QObject):
  # Emits 3-element dict: Deleted comp ids, changed comp ids, added comp ids
  _defaultEmitDict = {'deleted': np.array([]), 'changed': np.array([]), 'added': np.array([])}
  sigCompsChanged = Signal(dict)
  sigCompClicked = Signal(object)
  sigCompVertsChanged = Signal(object)

  _nextCompId = 0

  def __init__(self):
    super().__init__()
    colNames = [key for key in Component().__dict__.keys()]
    self._compList = df(columns=colNames)
    # Ensure ID is numeric
    self._compList['instanceId'] = self._compList['instanceId'].astype(int)

  def addComps(self, comps: List[Component], addtype='new'):
    toEmit = self._defaultEmitDict.copy()
    # Preallocate list size since we know its size in advance
    newVerts = [None]*len(comps)
    newIds = np.arange(self._nextCompId, self._nextCompId + len(comps), dtype=int)
    newDf_list = []
    for ii, comp in enumerate(comps):
      newVerts[ii] = comp.vertices
      comp.instanceId = newIds[ii]
      newDf_list.append([getattr(comp, field) for field in comp.__dict__])
      comp.sigIdClicked.connect(self._rethrowClick)
      comp.sigVertsChanged.connect(self._rethrowVerts)

    self._nextCompId += newIds[-1] + 1
    newDf = df(newDf_list, columns=self._compList.columns)
    self._compList = pd.concat((self._compList, newDf))
    toEmit['added'] = newIds
    self.sigCompsChanged.emit(toEmit)

  def rmComps(self, idsToRemove: Union[np.array, str] = 'all'):
    toEmit = self._defaultEmitDict.copy()
    # Generate ID list
    existingCompIds = self._compList.instanceId
    if idsToRemove == 'all':
      idsToRemove = existingCompIds
    elif not hasattr(idsToRemove, '__iter__'):
      # single number passed in
      idsToRemove = [idsToRemove]
      pass

    tfRmIdx = np.isin(existingCompIds, idsToRemove)

    # Reset manager's component list
    self._compList = self._compList.loc[np.invert(tfRmIdx),:]

    # Determine next ID for new components
    self._nextCompId = 0
    if not np.all(tfRmIdx):
      self._nextCompId = np.max(existingCompIds[np.invert(tfRmIdx)]) + 1

    # Reflect these changes to the component list
    toEmit['deleted'] = idsToRemove
    self.sigCompsChanged.emit(toEmit)

  # Allow direct indexing into comp list using comp IDs
  def __getitem__(self, keyWithIds):
    # If key is a slice, convert to array
    idList = keyWithIds[0]
    if isinstance(idList, slice):
      start, stop, step = idList.start, idList.stop, idList.step
      if start == None:
        start = 0
      if stop == None:
        stop = len(self._compList)
      idList = np.arange(start, stop, step)

    xpondingIdxs = self.idToRowIdx(idList)

    return self._compList.loc[xpondingIdxs, keyWithIds[1]]

  def idToRowIdx(self, idList: np.ndarray):
    """
    Returns indices into component list that correspond to the specified
    component ids
    """
    return self._compList.index.intersection(idList)

  def _rethrowClick(self):
    comp = self.sender()
    self.sigCompClicked.emit(comp)

  def _rethrowVerts(self):
    comp = self.sender()
    self.sigCompVertsChanged.emit(comp)

class CompDisplayFilter(QtCore.QObject):
  sigCompClicked = Signal(object)

  def __init__(self, compMgr: ComponentMgr, mainImg: pg.PlotWindow,
               compTbl: CompTable, filterWidget):
    super().__init__()
    self._mainImgArea = mainImg
    self._filter = filterWidget
    self._compTbl = compTbl
    self._compMgr = compMgr

    # Attach to manager signals
    self._compMgr.sigCompClicked.connect(self._rethrowCompClicked)
    self._compMgr.sigCompVertsChanged.connect(self._reflectVertsChanged)
    self._compMgr.sigCompsChanged.connect(self.redrawComps)

    self._compBounds = MultiRegionPlot()
    self._displayedIds = np.array([], dtype=int)
    # Keep copy of old id plots to delete when they are removed from compMgr
    # No need to keep vertices, since deleted vertices are handled by the
    # MultiRegionPlot and table rows will still have the associated ID
    self._oldPlotsDf = df(columns=['instanceId', '_txtPlt'])

    mainImg.addItem(self._compBounds)

  def redrawComps(self, idLists):
    # Following mix of cases are possible:
    # Components: DELETED, UNCHANGED, CHANGED, NEW
    # New is different from changed since id plot already exists (unhide vs. create)
    # Plots: DRAWN, UNDRAWN
    # Note that hiding the ID is chosen instead of deleting, since that is a costly graphics
    # operation

    # For new components: Add hidden id plot. This will be shown later if filter allows
    pltsToAdd = self._compMgr[idLists['added'],'_txtPlt']
    for plt in pltsToAdd: # Type: ClickableTextItem
      plt.hide()
      self._mainImgArea.addItem(plt)

    # Component deleted: Delete hidden id plot
    idsToRm = idLists['deleted']
    deletedIdxs = np.isin(self._oldPlotsDf.loc[:, 'instanceId'], idsToRm)
    pltsToRm = self._oldPlotsDf.loc[deletedIdxs,'_txtPlt']
    for plt in pltsToRm:
      self._mainImgArea.removeItem(plt)

    # Update filter list: hide/unhide ids and verts as needed. This should occur in other
    # functions that hook into signals sent from filter widget
    # TODO: Implement this logic. Will be placed in self._displayedIds
    # Only plot shown vertices
    self._displayedIds = self._compMgr[:,'instanceId']

    pltsToShow = self._compMgr[self._displayedIds, '_txtPlt']
    for plt in pltsToShow:
      plt.show()
    self._compBounds.resetRegionList(self._displayedIds, self._compMgr[self._displayedIds, 'vertices'])

    # Reset list of plot handles
    self._oldPlotsDf = self._compMgr[:,['instanceId', '_txtPlt']]


  @Slot(object)
  def _rethrowCompClicked(self, comp: Component):
    self.sigCompClicked.emit(comp)

  @Slot()
  def resetCompBounds(self):
    self._compBounds.resetRegionList()

  @Slot(object)
  def _reflectVertsChanged(self, comp: Component):
    self._compBounds.setRegions(comp.instanceId, comp.vertices)

  @staticmethod
  def setScheme(scheme: SchemeEditor):
    # Pass this scheme to the MultiRegionPlot
    MultiRegionPlot.setScheme(scheme)

if __name__ == '__main__':
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