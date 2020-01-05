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
    # It is OK for NaN mean values, since this will hide the text
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      newPos = np.mean(self.vertices, axis=0)
      self._txtPlt.setPos(newPos[0], newPos[1])
    self._txtPlt.updateText(str(self.instanceId), self.validated)

class ComponentMgr(QtCore.QObject):
  sigCompClicked = Signal(object)

  _nextCompId = 0

  def __init__(self, mainImgArea: pg.GraphicsWidget, compTbl: CompTable.CompTable):
    super().__init__()
    self._mainImgArea = mainImgArea
    self._compBounds = MultiRegionPlot()
    self._compTbl = compTbl
    # Update comp image on main image change
    self._mainImgArea.addItem(self._compBounds)
    name_typeTups = [(key, type(val)) for key, val in Component().__dict__.items()]
    self._compList = df(columns=[tup[0] for tup in name_typeTups])
    # Ensure ID is numeric
    self._compList['instanceId'] = self._compList['instanceId'].astype(int)

  def addComps(self, comps: List[Component], addtype='new'):
    # Preallocate list size since we know its size in advance
    newVerts = [None]*len(comps)
    newIds = np.arange(self._nextCompId, self._nextCompId + len(comps), dtype=int)
    newDf_list = []
    for ii, comp in enumerate(comps):
      newVerts[ii] = comp.vertices
      comp.instanceId = newIds[ii]
      newDf_list.append([getattr(comp, field) for field in comp.__dict__])
      self._mainImgArea.addItem(comp._txtPlt)
      # Listen for component signals and rethrow them
      comp.sigCompClicked.connect(self._rethrowCompClick)
      comp.sigVertsChanged.connect(self._reflectVertsChanged)

    self._nextCompId += newIds[-1] + 1
    newDf = df(newDf_list, columns=self._compList.columns)
    self._compList = pd.concat((self._compList, newDf))
    self._compBounds.setRegions(newIds, newVerts)
    self._compTbl.resetComps(self._compList)

  def rmComps(self, idsToRemove: Union[np.array, str] = 'all'):
    # Generate ID list
    existingCompIds = self._compList.instanceId
    if idsToRemove == 'all':
      idsToRemove = existingCompIds
    elif not hasattr(idsToRemove, '__iter__'):
      # single number passed in
      idsToRemove = [idsToRemove]
      pass

    tfRmIdx = np.isin(existingCompIds, idsToRemove)

    plotsToRemove = self._compList.loc[tfRmIdx, '_txtPlt']
    for plt in plotsToRemove:
      self._mainImgArea.removeItem(plt)

    # Reset manager's component list
    self._compList = self._compList.loc[np.invert(tfRmIdx),:]

    # Update bound plot
    self._compBounds.setRegions(idsToRemove, [[] for id in idsToRemove])
    self._compTbl.resetComps(self._compList)

    # Determine next ID for new components
    self._nextCompId = 0
    if not np.all(tfRmIdx):
      self._nextCompId = np.max(existingCompIds[np.invert(tfRmIdx)]) + 1

  @Slot()
  def _rethrowCompClick(self):
    comp: Component = self.sender()
    self.sigCompClicked.emit(comp)

  @Slot()
  def _reflectVertsChanged(self):
    comp: Component = self.sender()
    self._compBounds.setRegions(comp.instanceId, comp.vertices)

  @Slot()
  def resetCompBounds(self):
    self._compBounds.resetRegionList()

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