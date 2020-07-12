from typing import Union
from warnings import warn

import numpy as np
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtGui

from s3a.views import tableview
from s3a.structures.typeoverloads import OneDArr
from s3a import FR_SINGLETON
from s3a.views.imageareas import FRMainImage
from s3a.views.regions import FRMultiRegionPlot
from s3a.projectvars import FR_CONSTS, REQD_TBL_FIELDS
from s3a.structures import FRVertices, FRParam, FRParamEditorError, FRS3AWarning
from s3a.models.tablemodel import FRComponentMgr

__all__ = ['FRCompSortFilter', 'FRCompDisplayFilter']

Signal = QtCore.Signal
TBL_FIELDS = FR_SINGLETON.tableData.allFields

class FRCompSortFilter(QtCore.QSortFilterProxyModel):
  colTitles = [f.name for f in TBL_FIELDS]
  def __init__(self, compMgr: FRComponentMgr, parent=None):
    super().__init__(parent)
    self.setSourceModel(compMgr)
    # TODO: Move code for filtering into the proxy too. It will be more efficient and
    #  easier to generalize than the current solution in FRCompDisplayFilter.


  def sort(self, column: int, order: QtCore.Qt.SortOrder=...) -> None:
    # Do nothing if the user is trying to sort by vertices, since the intention of
    # sorting numpy arrays is somewhat ambiguous
    if column == self.colTitles.index(REQD_TBL_FIELDS.VERTICES.name):
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

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_COMP_TBL)
class FRCompDisplayFilter(QtCore.QObject):
  sigCompsSelected = Signal(object)

  def __init__(self, compMgr: FRComponentMgr, mainImg: FRMainImage,
               compTbl: tableview.FRCompTableView, parent=None):
    super().__init__(parent)
    filterEditor = FR_SINGLETON.filter
    self._mainImgArea = mainImg
    self._filter = filterEditor.params.getValues()
    self._compTbl = compTbl
    self._compMgr = compMgr

    self.regionPlot = FRMultiRegionPlot()
    self.displayedIds = np.array([], dtype=int)
    self.selectedIds = np.array([], dtype=int)

    # Attach to UI signals
    mainImg.sigSelectionBoundsMade.connect(self._reflectSelectionBoundsMade)

    mainImg.mergeCompsAct.sigActivated.connect(lambda *args: self.mergeSelectedComps())
    mainImg.splitCompsAct.sigActivated.connect(lambda *args: self.splitSelectedComps())
    compMgr.sigCompsChanged.connect(self.redrawComps)
    filterEditor.sigParamStateUpdated.connect(self._updateFilter)
    FR_SINGLETON.scheme.sigParamStateUpdated.connect(lambda: self._updateFilter(self._filter))
    compTbl.sigSelectionChanged.connect(self._reflectTableSelectionChange)

    mainImg.addItem(self.regionPlot)

    self.filterableCols = self.findFilterableCols()

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
    regCols = (REQD_TBL_FIELDS.VERTICES,)
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
    self.regionPlot.resetRegionList(self.displayedIds, compDf.loc[self.displayedIds, regCols])
    # noinspection PyTypeChecker
    # self._reflectTableSelectionChange(np.intersect1d(self.displayedIds, self.selectedIds))

    tblIdsToShow = np.isin(compDf.index, self.displayedIds).nonzero()[0]
    model = self._compTbl.model()
    for rowId in tblIdsToShow:
      xpondingIdx = model.mapFromSource(self._compMgr.index(rowId,0)).row()
      self._compTbl.showRow(xpondingIdx)

  def _updateFilter(self, newFilterDict):
    self._filter = newFilterDict
    self.redrawComps(self._compMgr.defaultEmitDict)

  def splitSelectedComps(self):
    """See signature for :meth:`FRComponentMgr.splitCompsById`"""
    selection, _ = self._compTbl.getIds_colsFromSelection(ignoreNoEditCols=True)
    if selection is None:
      return
    self._compMgr.splitCompVertsById(selection)
    self.selectRowsById(selection, QtCore.QItemSelectionModel.ClearAndSelect)

  def mergeSelectedComps(self, keepId: int=None):
    """See signature for :meth:`FRComponentMgr.mergeCompsById`"""
    selection, _ = self._compTbl.getIds_colsFromSelection(ignoreNoEditCols=True)

    if selection is None or len(selection) < 2:
      # Nothing to do
      return
    if keepId is None:
      keepId = selection[0]
    try:
      self._compMgr.mergeCompVertsById(selection, keepId)
    except FRS3AWarning:
      # No merge was performed, don't alter the table selection
      raise
    else:
      self.selectRowsById(np.array([keepId]), QtCore.QItemSelectionModel.ClearAndSelect)

  def _reflectTableSelectionChange(self, selectedIds: OneDArr):
    self.selectedIds = selectedIds
    self.regionPlot.selectById(selectedIds)
    self.sigCompsSelected.emit(self._compMgr.compDf.loc[selectedIds, :])

  def selectRowsById(self, ids: OneDArr,
                     selectionMode=QtCore.QItemSelectionModel.Rows):
    selectionModel = self._compTbl.selectionModel()
    sortModel = self._compTbl.model()
    isFirst = True
    shouldScroll = len(ids) > 0
    selectionList = QtCore.QItemSelection()
    for curId in ids:
      idRow = np.nonzero(self._compMgr.compDf.index == curId)[0][0]
      # Map this ID to its sorted position in the list
      idxForId = sortModel.mapFromSource(self._compMgr.index(idRow, 0))
      selectionList.select(idxForId, idxForId)
      if isFirst and shouldScroll:
        self._compTbl.scrollTo(idxForId, self._compTbl.PositionAtCenter)
        isFirst = False
    # noinspection PyTypeChecker
    selectionModel.select(selectionList, selectionMode)
    self.selectedIds = ids
    self._compTbl.setFocus()

  def _reflectSelectionBoundsMade(self, selection: Union[OneDArr, FRVertices]):
    """
    :param selection: bounding box of user selection: [xmin ymin; xmax ymax]
    """
    # If min and max are the same, just check for points at mouse position
    if len(selection) == 1 or np.abs(selection[0] - selection[1]).sum() < 0.01:
      qtPoint = QtCore.QPointF(*selection[0])
      selectedSpots = self.regionPlot.pointsAt(qtPoint)
      selectedIds = [spot.data() for spot in selectedSpots]
    else:
      selectedIds = self.regionPlot.boundsWithin(selection)
      selectedIds = np.unique(selectedIds)

    # -----
    # Obtain table idxs corresponding to ids so rows can be highlighted
    # -----
    # Add to current selection depending on modifiers
    mode = QtCore.QItemSelectionModel.Rows
    if QtGui.QGuiApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
      mode |= QtCore.QItemSelectionModel.Select
    else:
      mode |= QtCore.QItemSelectionModel.ClearAndSelect
    self.selectRowsById(selectedIds, mode)
    # TODO: Better management of widget focus here

  def _populateDisplayedIds(self):
    curComps = self._compMgr.compDf.copy()
    for param in self.filterableCols:
      curComps = self.filterByParamType(curComps, param)

    # Give self the id list of surviving comps
    self.displayedIds = curComps[REQD_TBL_FIELDS.INST_ID]

  def findFilterableCols(self):
    curComps = self._compMgr.compDf.copy()
    filterableCols = []
    badCols = []
    for param in curComps.columns:
      try:
        curComps = self.filterByParamType(curComps, param)
        filterableCols.append(param)
      except FRParamEditorError:
        badCols.append(param)
    if len(badCols) > 0:
      badTypes = np.unique([f'"{col.valType}"' for col in badCols])
      badCols = map(lambda val: f'"{val}"', badCols)
      warn(f'The table filter does not know how to handle'
           f' columns {", ".join(badCols)} since no'
           f' filter exists for types {", ".join(badTypes)}',
           FRS3AWarning)
    return filterableCols

  def filterByParamType(self, compDf: df, param: FRParam):
    # TODO: Each type should probably know how to filter itself. That is,
    #  find some way of keeping this logic from just being an if/else tree...
    if param.name not in self._filter:
      return compDf
    valType = param.valType
    # idx 0 = value, 1 = children
    curFilterParam = self._filter[param.name][1]
    dfAtParam = compDf.loc[:, param]

    if valType in ['int', 'float']:
      curmin, curmax = [curFilterParam[name][0] for name in ['min', 'max']]

      compDf = compDf.loc[(dfAtParam >= curmin) & (dfAtParam <= curmax),:]
    elif valType == 'bool':
      allowTrue, allowFalse = [curFilterParam[name][0] for name in
                               [f'{param.name}', f'Not {param.name}']]

      validList = np.array(dfAtParam, dtype=bool)
      if not allowTrue:
        compDf = compDf.loc[~validList, :]
      if not allowFalse:
        compDf = compDf.loc[validList, :]
    elif valType == 'FRParam':
      existingParams = np.array(dfAtParam)
      allowedParams = []
      for groupSubParam in param.value.group:
        isAllowed = curFilterParam[groupSubParam.name][0]
        if isAllowed:
          allowedParams.append(groupSubParam)
      compDf = compDf.loc[np.isin(existingParams, allowedParams),:]
    elif valType in ['str', 'text']:
      allowedRegex = self._filter[param.name][0]
      isCompAllowed = dfAtParam.str.contains(allowedRegex, regex=True, case=False)
      compDf = compDf.loc[isCompAllowed,:]
    elif valType == 'FRComplexVertices':
      vertsAllowed = np.ones(len(dfAtParam), dtype=bool)

      xParam = curFilterParam['X Bounds'][1]
      yParam = curFilterParam['Y Bounds'][1]
      xmin, xmax, ymin, ymax = [param[val][0] for param in (xParam, yParam) for val in ['min', 'max']]

      for vertIdx, verts in enumerate(dfAtParam):
        stackedVerts: FRVertices = verts.stack()
        xVerts, yVerts = stackedVerts.x, stackedVerts.y
        isAllowed = np.all((xVerts >= xmin) & (xVerts <= xmax)) & \
                    np.all((yVerts >= ymin) & (yVerts <= ymax))
        vertsAllowed[vertIdx] = isAllowed
      compDf = compDf.loc[vertsAllowed,:]
    else:
      warn('No filter type exists for parameters of type ' f'{valType}.'
           f' Did not filter column {param.name}.',
           FRS3AWarning)
    return compDf


  def resetCompBounds(self):
    self.regionPlot.resetRegionList()