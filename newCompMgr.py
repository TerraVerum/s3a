from __future__ import annotations
from component import Component

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
from constants import ComponentTypes, SchemeValues as SV, ComponentTableFields as CTF
from processing import sliceToArray

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot
# Ensure an application instance is running
app = pg.mkQApp()

class ComponentDf(QtCore.QObject, df):
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
    self.instanceId = -1
    self.vertices = np.array([np.NaN, np.NaN])
    self.deviceType = ComponentTypes.N_A
    self.boardText = ''
    self.deviceText = ''
    self.logo = ''
    self.notes = ''
    self.validated = False
    self.__dict__.update(initValsForVars)


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


class CompMgrModel(QtCore.QAbstractTableModel):
  # Emits 3-element dict: Deleted comp ids, changed comp ids, added comp ids
  defaultEmitDict = {'deleted': np.array([]), 'changed': np.array([]), 'added': np.array([])}
  sigCompsChanged = Signal(dict)
  # Used to alert models about data changes
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


