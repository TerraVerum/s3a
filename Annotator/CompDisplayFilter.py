# Required to avoid cyclic dependency from CompTable annotation
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore

from .ABGraphics import tableview
from .ABGraphics.clickables import ClickableTextItem
from .ABGraphics.parameditors import TableFilterEditor
from .ABGraphics.regions import MultiRegionPlot, makeMultiRegionDf
from .constants import TEMPLATE_COMP as TC
from .tablemodel import ComponentMgr, ComponentTypes

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot

class CompSortFilter(QtCore.QSortFilterProxyModel):
  colTitles = TC.paramNames()
  def __init__(self, compMgr: ComponentMgr, parent=None):
    super().__init__(parent)
    self.setSourceModel(compMgr)
    # TODO: Move code for filtering into the proxy too. It will be more efficient and
    #  easier to generalize than the current solution in CompDisplayFilter.


  def sort(self, column: int, order: QtCore.Qt.SortOrder=...) -> None:
    # Do nothing if the user is trying to sort by vertices, since the intention of
    # sorting numpy arrays is somewhat ambiguous
    if column == self.colTitles.index(TC.VERTICES.name):
      return
    else:
      super().sort(column, order)

  def lessThan(self, left: QtCore.QModelIndex, right: QtCore.QModelIndex) -> bool:
    # First, attempt to compare the object data
    leftObj = left.data(QtCore.Qt.EditRole)
    rightObj = right.data(QtCore.Qt.EditRole)
    try:
      return np.all(leftObj < rightObj)
    except ValueError:
      # If that doesn't work, default to stringified comparison
      return str(leftObj) < str(rightObj)


class CompDisplayFilter(QtCore.QObject):
  sigCompClicked = Signal(object)

  def __init__(self, compMgr: ComponentMgr, mainImg: pg.PlotWindow,
               compTbl: tableview.CompTableView, filterEditor: TableFilterEditor):
    super().__init__()
    self._mainImgArea = mainImg
    self._filter = filterEditor.params.getValues()
    self._compTbl = compTbl
    self._compMgr = compMgr

    # Attach to main image area signals
    mainImg.sigSelectionCreated.connect(self._compPointsSelected)

    # Attach to manager signals
    compMgr.sigCompsChanged.connect(self.redrawComps)

    # Retrieve filter changes
    filterEditor.sigParamStateUpdated.connect(self._updateFilter)

    # Update based on table selection
    compTbl.sigSelectionChanged.connect(self._reflectTableSelectionChange)
    #self.sigProxy = SignalProxy(self._compTbl.sigSelectionChanged, delay=0.25, slot=self._reflectTableSelectionChange)

    self._regionPlots = MultiRegionPlot()
    self.displayedIds = np.array([], dtype=int)

    for plt in self._regionPlots.boundPlt, self._regionPlots.idPlts:
      mainImg.addItem(plt)
    self._regionPlots.sigIdClicked.connect(self.handleCompClick)

  def redrawComps(self, idLists):
    # Following mix of cases are possible:
    # Components: DELETED, UNCHANGED, CHANGED, NEW
    # New is different from changed since id plot already exists (unhide vs. create)
    # Plots: DRAWN, UNDRAWN
    # Note that hiding the ID is chosen instead of deleting, since that is a costly graphics
    # operation
    compDf = self._compMgr.compDf

    # Update and add changed/new components
    # TODO: Find out why this isn't working. For now, just reset the whole comp list
    #  each time components are changed, since the overhead isn't too terrible.
    regCols = (TC.VERTICES, TC.VALIDATED)
    # changedIds = np.concatenate((idLists['added'], idLists['changed']))
    # self._regionPlots[changedIds, regCols] = compDf.loc[changedIds, compCols]

    # Hide all ids and table rows, since they will be reshown as needed after display filtering
    for rowIdx in range(len(compDf)):
      self._compTbl.hideRow(rowIdx)

    # Component deleted: Nothing to do, since only displayed IDs will remain in the
    # region manager anyway
    #idsToRm = idLists['deleted']

    # Update filter list: hide/unhide ids and verts as needed.
    self._populateDisplayedIds()
    # Remove all IDs that aren't displayed
    # FIXME: This isn't working correctly at the moment
    # self._regionPlots.drop(np.setdiff1d(self._regionPlots.data.index, self._displayedIds))
    self._regionPlots.resetRegionList(self.displayedIds, compDf.loc[self.displayedIds, regCols])

    tblIdxsToShow = np.nonzero(np.in1d(compDf.index, self.displayedIds))[0]
    for rowIdx in tblIdxsToShow:
      self._compTbl.showRow(rowIdx)

  def _updateFilter(self, newFilterDict):
    self._filter = newFilterDict
    self.redrawComps(self._compMgr.defaultEmitDict)

  @Slot(object)
  def _reflectTableSelectionChange(self, selectedIds: np.ndarray):
    self._regionPlots.selectById(selectedIds)

  @Slot(object)
  def _compPointsSelected(self, selectionBox: tuple):
    """
    :param selectionBox: bounding box of user selection: [xmin ymin xmax ymax]
    """
    selectedIds = self._regionPlots.idPlts.idsWithin(selectionBox)
    self.updateCompSelection(selectedIds, scrollTo=len(selectedIds) > 0)

  def updateCompSelection(self, selectedIds, scrollTo=True):
    self._compTbl.clearSelection()
    mode = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
    selectionModel = self._compTbl.selectionModel()
    sortModel = self._compTbl.model()
    # Select in reverse order so the first element can be accessed by the end of the operation
    for curId in selectedIds[::-1]:
      idRow = np.nonzero(self._compMgr.compDf.index == curId)[0][0]
      # Map this ID to its sorted position in the list
      idxForId = sortModel.mapFromSource(self._compMgr.index(idRow, 0))
      selectionModel.select(idxForId, mode)
    if scrollTo:
      # When the ID is selected, scroll to that row and highlight the ID
      self._compTbl.scrollTo(idxForId, self._compTbl.PositionAtCenter)
    self._compTbl.setFocus()


  def _populateDisplayedIds(self):
    curComps = self._compMgr.compDf

    # idx 0 = value, 1 = children
    # ------
    # ID FILTERING
    # ------
    curParam = self._filter[TC.INST_ID.name][1]
    curmin, curmax = [curParam[name][0] for name in ['min', 'max']]

    idList = curComps.index
    curComps = curComps.loc[(idList >= curmin) & (idList <= curmax),:]

    # ------
    # VALIDATED FILTERING
    # ------
    curParam = self._filter[TC.VALIDATED.name][1]
    allowValid, allowInvalid = [curParam[name][0] for name in ['Validated', 'Not Validated']]

    validList = np.array(curComps.loc[:, TC.VALIDATED], dtype=bool)
    if not allowValid:
      curComps = curComps.loc[~validList, :]
    if not allowInvalid:
      curComps = curComps.loc[validList, :]

    # ------
    # DEVICE TYPE FILTERING
    # ------
    compTypes = np.array(curComps.loc[:, TC.DEV_TYPE])
    curParam = self._filter[TC.DEV_TYPE.name][1]
    allowedTypes = []
    for curType in ComponentTypes:
      isAllowed = curParam[curType.value][0]
      if isAllowed:
        allowedTypes.append(curType)
    curComps = curComps.loc[np.isin(compTypes, allowedTypes),:]

    # ------
    # LOGO, NOTES, BOARD, DEVICE TEXT FILTERING
    # ------
    nextParamNames = [TC.LOGO, TC.NOTES, TC.BOARD_TEXT, TC.DEV_TEXT]
    for param in nextParamNames:
      compParamVals = curComps.loc[:, param]
      allowedRegex = self._filter[param.name][0]
      isCompAllowed = compParamVals.str.contains(allowedRegex, regex=True, case=False)
      curComps = curComps.loc[isCompAllowed,:]

    # ------
    # VERTEX FILTERING
    # ------
    compVerts = curComps.loc[:, TC.VERTICES]
    vertsAllowed = np.ones(len(compVerts), dtype=bool)

    vertParam = self._filter[TC.VERTICES.name][1]
    xParam = vertParam['X Bounds'][1]
    yParam = vertParam['Y Bounds'][1]
    xmin, xmax, ymin, ymax = [param[val][0] for param in (xParam, yParam) for val in ['min', 'max']]

    for vertIdx, verts in enumerate(compVerts):
      # Remove nan values for computation
      verts = verts[~np.isnan(verts[:,0]),:]
      xVerts = verts[:,0]
      yVerts = verts[:,1]
      isAllowed = np.all((xVerts >= xmin) & (xVerts <= xmax)) & \
                  np.all((yVerts >= ymin) & (yVerts <= ymax))
      vertsAllowed[vertIdx] = isAllowed
    curComps = curComps.loc[vertsAllowed,:]

    # Give self the id list of surviving comps
    self.displayedIds = curComps.index

  @Slot()
  def resetCompBounds(self):
    self._regionPlots.resetRegionList()

  @Slot(int)
  def handleCompClick(self, clickedId=None):
    self.updateCompSelection([clickedId], scrollTo=True)
    # noinspection PyTypeChecker
    self.sigCompClicked.emit(self._compMgr.compDf.loc[clickedId,:])
