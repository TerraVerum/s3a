from typing import Union
from warnings import warn

import numpy as np
from pandas import DataFrame as df
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from s3a import FR_SINGLETON
from s3a.generalutils import cornersToFullBoundary
from s3a.models.tablemodel import FRComponentMgr
from s3a.constants import FR_CONSTS, REQD_TBL_FIELDS, FR_ENUMS
from s3a.structures import FRVertices, FRParam, FRParamEditorError, FRS3AWarning, \
  FRComplexVertices
from s3a.structures.typeoverloads import OneDArr
from s3a.views import tableview
from s3a.views.imageareas import FRMainImage
from s3a.views.regions import FRMultiRegionPlot

__all__ = ['FRCompSortFilter', 'FRCompDisplayFilter']

Signal = QtCore.Signal
TBL_FIELDS = FR_SINGLETON.tableData.allFields
QISM = QtCore.QItemSelectionModel

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

    self.regionCopier = mainImg.regionCopier

    # Attach to UI signals
    mainImg.sigSelectionBoundsMade.connect(self._reflectSelectionBoundsMade)
    self.regionCopier.sigCopyStarted.connect(lambda *args: self.activateRegionCopier())
    self.regionCopier.sigCopyStopped.connect(lambda *args: self.finishRegionCopier())

    mainImg.mergeCompsAct.sigActivated.connect(lambda *args: self.mergeSelectedComps())
    mainImg.splitCompsAct.sigActivated.connect(lambda *args: self.splitSelectedComps())
    compMgr.sigCompsChanged.connect(self.redrawComps)
    compMgr.sigFieldsChanged.connect(lambda: self._reflectFieldsChanged())
    filterEditor.sigParamStateUpdated.connect(self._updateFilter)
    FR_SINGLETON.colorScheme.sigParamStateUpdated.connect(lambda: self._updateFilter(self._filter))
    compTbl.sigSelectionChanged.connect(self._reflectTableSelectionChange)

    mainImg.addItem(self.regionPlot)
    mainImg.addItem(self.regionCopier)

    self.filterableCols = self.findFilterableCols()

  def redrawComps(self, idLists=None):
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
    selection = self._compTbl.ids_rows_colsFromSelection(excludeNoEditCols=False,
                                                            warnNoneSelection=False)
    if len(selection) == 0:
      return
    changes = self._compMgr.splitCompVertsById(np.unique(selection[:,0]))
    self.selectRowsById(changes['added'], QISM.ClearAndSelect)

  def mergeSelectedComps(self, keepId: int=None):
    """See signature for :meth:`FRComponentMgr.mergeCompsById`"""
    selection = self._compTbl.ids_rows_colsFromSelection(excludeNoEditCols=False,
                                                         warnNoneSelection=False)

    if len(selection) < 2:
      # Nothing to do
      return
    if keepId is None:
      keepId = selection[0,0]
    try:
      self._compMgr.mergeCompVertsById(np.unique(selection[:,0]), keepId)
    except FRS3AWarning:
      # No merge was performed, don't alter the table selection
      raise
    else:
      self.selectRowsById(np.array([keepId]), QISM.ClearAndSelect)

  def _reflectFieldsChanged(self):
    self.filterableCols = self.findFilterableCols()
    self.redrawComps()

  def _reflectTableSelectionChange(self, selectedIds: OneDArr):
    self.selectedIds = selectedIds
    self.regionPlot.selectById(selectedIds)
    selectedComps = self._compMgr.compDf.loc[selectedIds, :]
    self.sigCompsSelected.emit(selectedComps)
    self.scaleViewboxToSelectedIds()

  def scaleViewboxToSelectedIds(self, selectedIds: OneDArr=None, onlyGrow=None,
                                padding: int=None):
    """
    Rescales the main image viewbox to encompass the selection

    :param selectedIds: Ids to scale to. If *None*, this is the current selection
    :param onlyGrow: If *True*, the viewbox will never shrink to the selection.
      If *None*, the value is determined from the parameter editor.
    :param padding: Padding around the selection. If *None*, defaults to pad value
      in param editor.
    """
    if onlyGrow is None:
      onlyGrow = self._mainImgArea.onlyGrowViewbox
    if padding is None:
      if onlyGrow:
        padding = 0
      else:
        padding = self._mainImgArea.compCropMargin
        if self._mainImgArea.treatMarginAsPct:
          padding = int(max(self._mainImgArea.image.shape[:2])*padding/100)
    if selectedIds is None:
      selectedIds = self.selectedIds
    if len(selectedIds) == 0: return
    # Calculate how big the viewbox needs to be
    selectedVerts = self._compMgr.compDf.loc[selectedIds, REQD_TBL_FIELDS.VERTICES]
    allVerts = np.vstack([v.stack() for v in selectedVerts])
    mins = allVerts.min(0) - padding//2
    maxs = allVerts.max(0) + padding//2
    vb: pg.ViewBox = self._mainImgArea.getViewBox()
    curXRange = vb.state['viewRange'][0]
    curYRange = vb.state['viewRange'][1]
    if onlyGrow:
      mins[0] = np.min(curXRange + [mins[0]])
      maxs[0] = np.max(curXRange + [maxs[0]])
      mins[1] = np.min(curYRange + [mins[1]])
      maxs[1] = np.max(curYRange + [maxs[1]])
    viewRect = QtCore.QRectF(*mins, *(maxs - mins))
    vb.setRange(viewRect, padding=0)


  def selectRowsById(self, ids: OneDArr,
                     selectionMode=QISM.Rows|QISM.ClearAndSelect,
                     onlyEditableRetList=True):
    selectionModel = self._compTbl.selectionModel()
    sortModel = self._compTbl.model()
    isFirst = True
    shouldScroll = len(ids) > 0
    selectionList = QtCore.QItemSelection()
    retLists = [] # See tableview ids_rows_colsFromSelection
    if onlyEditableRetList:
      selectedCols = self._compMgr.editColIdxs
    else:
      selectedCols = np.arange(len(self._compMgr.colTitles))
    numCols = len(self._compMgr.colTitles)
    for curId in ids:
      idRow = np.nonzero(self._compMgr.compDf.index == curId)[0][0]
      # Map this ID to its sorted position in the list
      idxForId = sortModel.mapFromSource(self._compMgr.index(idRow, 0))
      selectionList.select(idxForId, idxForId)
      if isFirst and shouldScroll:
        self._compTbl.scrollTo(idxForId, self._compTbl.PositionAtCenter)
        isFirst = False
      tblRow = idxForId.row()
      retLists.extend([[curId, tblRow, col] for col in selectedCols])
    # noinspection PyTypeChecker
    selectionModel.select(selectionList, selectionMode)
    return np.array(retLists)
    # if int(selectionMode & QISM.ClearAndSelect) > 0:
    #   self.selectedIds = ids
    # else: # Add to selection without clearing old selection
    #   self.selectedIds = np.concatenate([self.selectedIds, ids])


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
    mode = QISM.Rows
    if QtGui.QGuiApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
      mode |= QISM.Select
    else:
      mode |= QISM.ClearAndSelect
    if not self.regionCopier.active:
      self.selectRowsById(selectedIds, mode)
    # TODO: Better management of widget focus here

  def _populateDisplayedIds(self):
    curComps = self._compMgr.compDf.copy()
    for param in self.filterableCols:
      curComps = self.filterByParamType(curComps, param)

    # Give self the id list of surviving comps
    self.displayedIds = curComps[REQD_TBL_FIELDS.INST_ID]

  def activateRegionCopier(self, selectedIds: OneDArr=None):
    if selectedIds is None:
      selectedIds = self.selectedIds
    if len(selectedIds) == 0: return
    vertsList = self._compMgr.compDf.loc[selectedIds, REQD_TBL_FIELDS.VERTICES]
    self.regionCopier.resetBaseData(vertsList, selectedIds)
    self.regionCopier.active = True

  def finishRegionCopier(self, keepResult=True):
    if not keepResult: return
    newComps = self._compMgr.compDf.loc[self.regionCopier.regionIds].copy()
    regionOffset = self.regionCopier.offset.astype(int)
    # TODO: Truncate vertices that lie outside image boundaries
    # Invalid if any verts are outside image bounds
    truncatedCompIds = []
    # imShape_xy = self._mainImgArea.image.shape[:2][::-1]
    for idx in newComps.index:
      newVerts = []
      for verts in newComps.at[idx, REQD_TBL_FIELDS.VERTICES]:
        verts = verts + regionOffset
        # goodVerts = np.all(verts >= imShape_xy, 1)
        # if not np.all(goodVerts):
        #   verts = verts[goodVerts,:]
        #   truncatedCompIds.append(idx)
        newVerts.append(verts)
      newComps.at[idx, REQD_TBL_FIELDS.VERTICES] = FRComplexVertices(newVerts)
    # truncatedCompIds = np.unique(truncatedCompIds)
    if self.regionCopier.inCopyMode:
      self._mainImgArea.sigCompsCreated.emit(newComps)
      self.activateRegionCopier(self.regionCopier.regionIds)
    else: # Move mode
      self.regionCopier.erase()
      self._compMgr.addComps(newComps, FR_ENUMS.COMP_ADD_AS_MERGE)
    # if len(truncatedCompIds) > 0:
    #   warn(f'Some regions extended beyond image dimensions. Boundaries for the following'
    #        f' components were altered: {truncatedCompIds}', FRS3AWarning)

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
      badTypes = np.unique([f'"{col.pType}"' for col in badCols])
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
    pType = param.pType
    # idx 0 = value, 1 = children
    curFilterParam = self._filter[param.name][1]
    dfAtParam = compDf.loc[:, param]

    if pType in ['int', 'float']:
      curmin, curmax = [curFilterParam[name][0] for name in ['min', 'max']]

      compDf = compDf.loc[(dfAtParam >= curmin) & (dfAtParam <= curmax),:]
    elif pType == 'bool':
      allowTrue, allowFalse = [curFilterParam[name][0] for name in
                               [f'{param.name}', f'Not {param.name}']]

      validList = np.array(dfAtParam, dtype=bool)
      if not allowTrue:
        compDf = compDf.loc[~validList, :]
      if not allowFalse:
        compDf = compDf.loc[validList, :]
    elif pType in ['FRParam', 'list', 'popuplineeditor']:
      existingParams = np.array(dfAtParam)
      allowedParams = []
      if pType == 'FRParam':
        groupSubParams = [p.name for p in param.value.group]
      else:
        groupSubParams = param.opts['limits']
      for groupSubParam in groupSubParams:
        isAllowed = curFilterParam[groupSubParam][0]
        if isAllowed:
          allowedParams.append(groupSubParam)
      compDf = compDf.loc[np.isin(existingParams, allowedParams),:]
    elif pType in ['str', 'text']:
      allowedRegex = self._filter[param.name][0]
      isCompAllowed = dfAtParam.str.contains(allowedRegex, regex=True, case=False)
      compDf = compDf.loc[isCompAllowed,:]
    elif pType == 'FRComplexVertices':
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
      warn('No filter type exists for parameters of type ' f'{pType}.'
           f' Did not filter column {param.name}.',
           FRS3AWarning)
    return compDf


  def resetCompBounds(self):
    self.regionPlot.resetRegionList()