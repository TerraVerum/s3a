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

from ABGraphics.parameditors import SchemeEditor, TableFilterEditor
from ABGraphics.regions import MultiRegionPlot
from ABGraphics.clickables import ClickableTextItem
from ABGraphics import table
from constants import (ComponentTypes, SchemeValues as SV,
                       ABParamGroup, ABParam, NewComponentTableFields as CTF)
from processing import sliceToArray

from dataclasses import dataclass, field

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot
# Ensure an application instance is running
app = pg.mkQApp()

Component = DataComponent

class DataComponent(QtCore.QObject, CTF):
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

  def __init__(self, **initValsForVars):
    super().__init__()
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

class DataComponentMgr(QtCore.QObject):
  # Emits 3-element dict: Deleted comp ids, changed comp ids, added comp ids
  defaultEmitDict = {'deleted': np.array([]), 'changed': np.array([]), 'added': np.array([])}
  sigCompsChanged = Signal(dict)
  # Used to alert models about data changes
  sigCompsAboutToChange = Signal()
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
    toEmit = self.defaultEmitDict.copy()
    # Preallocate list size since we know its size in advance
    newVerts = [None]*len(comps)
    newIds = np.arange(self._nextCompId, self._nextCompId + len(comps), dtype=int)
    newDf_list = []
    for ii, comp in enumerate(comps):
      newVerts[ii] = comp.vertices
      comp.instanceId = newIds[ii]
      newDf_list.append([getattr(comp, field) for field in comp.__dict__])
      comp.sigIdClicked.connect(self._rethrowClick)
      comp.sigVertsChanged.connect(self._handleVertsChanged)

    self._nextCompId = newIds[-1] + 1
    newDf = df(newDf_list, columns=self._compList.columns)

    self.sigCompsAboutToChange.emit()
    self._compList = pd.concat((self._compList, newDf))
    toEmit['added'] = newIds
    self.sigCompsChanged.emit(toEmit)

  def rmComps(self, idsToRemove: Union[np.array, str] = 'all'):
    toEmit = self.defaultEmitDict.copy()
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
    self.sigCompsAboutToChange.emit()
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
      idList = sliceToArray(idList, self._compList.loc[:,'instanceId'])
    elif not hasattr(idList, '__iter__'):
      # Single number passed
      idList = np.array([idList])
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

  def _handleVertsChanged(self):
    comp = self.sender()
    # Update component vertex entry
    compIdx = self.idToRowIdx([comp.instanceId])
    self._compList.at[compIdx,'vertices'] = [comp.vertices]
    self.sigCompVertsChanged.emit(comp)


class CompDisplayFilter(QtCore.QObject):
  sigCompClicked = Signal(object)

  def __init__(self, compMgr: ComponentMgr, mainImg: pg.PlotWindow,
               compTbl: table.CompTableView, filterEditor: TableFilterEditor):
    super().__init__()
    self._mainImgArea = mainImg
    self._filter = filterEditor.params.getValues()
    self._compTbl = compTbl
    self._compMgr = compMgr

    # Attach to manager signals
    self._compMgr.sigCompClicked.connect(self._rethrowCompClicked)
    self._compMgr.sigCompVertsChanged.connect(self._reflectVertsChanged)
    self._compMgr.sigCompsChanged.connect(self.redrawComps)

    # Retrieve filter changes
    filterEditor.sigParamStateUpdated.connect(self._updateFilter)

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
    addedIds = idLists['added']
    pltsToAdd = self._compMgr[addedIds,'_txtPlt']
    for plt in pltsToAdd: # Type: ClickableTextItem
      self._mainImgArea.addItem(plt)

    # Hide all other ids and table rows, since they will be reshown as needed after display filtering
    for rowIdx, plt in enumerate(self._compMgr[:, '_txtPlt']):
      plt.hide()
      self._compTbl.hideRow(rowIdx)

    # Component deleted: Delete hidden id plot
    idsToRm = np.array(idLists['deleted'])
    deletedIdxs = np.isin(self._oldPlotsDf.loc[:, 'instanceId'], idsToRm)
    pltsToRm = self._oldPlotsDf.loc[deletedIdxs,'_txtPlt']
    for plt in pltsToRm:
      self._mainImgArea.removeItem(plt)

    # Component changed: Nothing required here

    # Update filter list: hide/unhide ids and verts as needed. This should occur in other
    # functions that hook into signals sent from filter widget
    # TODO: Implement this logic. Will be placed in self._displayedIds
    # Only plot shown vertices
    self._populateDisplayedIds()

    pltsToShow = self._compMgr[self._displayedIds, '_txtPlt']
    tblIdxsToShow = self._compMgr.idToRowIdx(self._displayedIds)
    for plt in pltsToShow:
      plt.show()
    for rowIdx in tblIdxsToShow:
      self._compTbl.showRow(rowIdx)

    self._compBounds.resetRegionList(self._displayedIds, self._compMgr[self._displayedIds, 'vertices'])

    # Reset list of plot handles
    self._oldPlotsDf = self._compMgr[:,['instanceId', '_txtPlt']]

  def _updateFilter(self, newFilterDict):
    self._filter = newFilterDict
    self.redrawComps(self._compMgr.defaultEmitDict)

  def _populateDisplayedIds(self):
    curComps = self._compMgr[:,:]

    # idx 0 = value, 1 = children
    # ------
    # ID FILTERING
    # ------
    curParam = self._filter[CTF.INST_ID.value][1]
    curmin, curmax = [curParam[name][0] for name in ['min', 'max']]

    idList = np.array(curComps.loc[:, 'instanceId'], dtype=int)
    curComps = curComps.loc[(idList >= curmin) & (idList <= curmax),:]

    # ------
    # VALIDATED FILTERING
    # ------
    curParam = self._filter[CTF.VALIDATED.value][1]
    allowValid, allowInvalid = [curParam[name][0] for name in ['Validated', 'Not Validated']]

    validList = np.array(curComps.loc[:, 'validated'], dtype=bool)
    if not allowValid:
      curComps = curComps.loc[~validList, :]
    if not allowInvalid:
      curComps = curComps.loc[validList, :]

    # ------
    # DEVICE TYPE FILTERING
    # ------
    compTypes = np.array(curComps.loc[:, 'deviceType'])
    curParam = self._filter[CTF.DEVICE_TYPE.value][1]
    allowedTypes = []
    for curType in ComponentTypes:
      isAllowed = curParam[curType.value][0]
      if isAllowed:
        allowedTypes.append(curType)
    curComps = curComps.loc[np.isin(compTypes, allowedTypes),:]

    # ------
    # LOGO, NOTES, BOARD, DEVICE TEXT FILTERING
    # ------
    nextParamNames = [param.value for param in [CTF.LOGO, CTF.NOTES, CTF.BOARD_TEXT, CTF.DEVICE_TEXT]]
    nextParamCompNames = ['logo', 'notes', 'boardText', 'deviceText']
    for param, compParamName in zip(nextParamNames, nextParamCompNames):
      compParamVals = curComps.loc[:, compParamName]
      allowedRegex = self._filter[param][0]
      isCompAllowed = compParamVals.str.contains(allowedRegex, regex=True, case=False)
      curComps = curComps.loc[isCompAllowed,:]

    # ------
    # VERTEX FILTERING
    # ------
    compVerts = curComps.loc[:, 'vertices']
    vertsAllowed = np.ones(len(compVerts), dtype=bool)

    vertParam = self._filter[CTF.VERTICES.value][1]
    xParam = vertParam['X Bounds'][1]
    yParam = vertParam['Y Bounds'][1]
    xmin, xmax, ymin, ymax = [param[val][0] for param in (xParam, yParam) for val in ['min', 'max']]

    for ii, verts in enumerate(compVerts):
      xVerts = verts[:,0]
      yVerts = verts[:,1]
      isAllowed = np.all((xVerts >= xmin) & (xVerts <= xmax)) & \
                  np.all((yVerts >= ymin) & (yVerts <= ymax))
      vertsAllowed[ii] = isAllowed
    curComps = curComps.loc[vertsAllowed,:]

    # Give self the id list of surviving comps
    self._displayedIds = np.array(curComps.loc[:, 'instanceId'])

  @Slot(object)
  def _rethrowCompClicked(self, comp: Component):
    self.sigCompClicked.emit(comp)

  @Slot()
  def resetCompBounds(self):
    self._compBounds.resetRegionList()

  @Slot(object)
  def _reflectVertsChanged(self, comp: Component):
    self._compBounds[comp.instanceId] = comp.vertices
    # Don't forget to update table entry
    self._compTbl.model().layoutChanged.emit()

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