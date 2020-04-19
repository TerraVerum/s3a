from typing import Union

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui

from cdef.structures.typeoverloads import OneDArr
from .frgraphics import tableview
from .frgraphics.imageareas import FRMainImage
from .frgraphics.parameditors import FR_SINGLETON
from .frgraphics.regions import MultiRegionPlot
from .projectvars import FR_CONSTS, TEMPLATE_COMP as TC, TEMPLATE_COMP_CLASSES as COMP_CLASSES
from .structures import FRVertices
from .tablemodel import FRComponentMgr

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot

class CompSortFilter(QtCore.QSortFilterProxyModel):
  colTitles = TC.paramNames()
  def __init__(self, compMgr: FRComponentMgr, parent=None):
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

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_COMP_TBL)
class CompDisplayFilter(QtCore.QObject):
  sigCompsSelected = Signal(object)

  def __init__(self, compMgr: FRComponentMgr, mainImg: FRMainImage,
               compTbl: tableview.CompTableView, parent=None):
    super().__init__(parent)
    filterEditor = FR_SINGLETON.filter
    self._mainImgArea = mainImg
    self._filter = filterEditor.params.getValues()
    self._compTbl = compTbl
    self._compMgr = compMgr

    self.regionPlots = MultiRegionPlot()
    self.displayedIds = np.array([], dtype=int)
    self.selectedIds = np.array([], dtype=int)

    # Attach to UI signals
    mainImg.sigSelectionBoundsMade.connect(self._reflectSelectionBoundsMade)
    compMgr.sigCompsChanged.connect(self.redrawComps)
    filterEditor.sigParamStateUpdated.connect(self._updateFilter)
    compTbl.sigSelectionChanged.connect(self._reflectTableSelectionChange)

    for plt in self.regionPlots.boundPlt, self.regionPlots.idPlts:
      mainImg.addItem(plt)

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
    self.regionPlots.resetRegionList(self.displayedIds, compDf.loc[self.displayedIds, regCols])
    # noinspection PyTypeChecker
    self._reflectTableSelectionChange(np.intersect1d(self.displayedIds, self.selectedIds))

    tblIdxsToShow = np.in1d(compDf.index, self.displayedIds).nonzero()[0]
    for rowIdx in tblIdxsToShow:
      self._compTbl.showRow(rowIdx)

  def _updateFilter(self, newFilterDict):
    self._filter = newFilterDict
    self.redrawComps(self._compMgr.defaultEmitDict)

  @Slot(object)
  def _reflectTableSelectionChange(self, selectedIds: OneDArr):
    self.selectedIds = selectedIds
    self.regionPlots.selectById(selectedIds)
    self.sigCompsSelected.emit(self._compMgr.compDf.loc[selectedIds, :])

  @Slot(object)
  def _reflectSelectionBoundsMade(self, selection: Union[OneDArr, FRVertices]):
    """
    :param selection: bounding box of user selection: [xmin ymin; xmax ymax]
    """
    # If min and max are the same, just check for points at mouse position
    if np.abs(selection[0] - selection[1]).sum() < 0.01:
      qtPoint = QtCore.QPointF(*selection[0])
      selectedSpots = self.regionPlots.idPlts.pointsAt(qtPoint)
      selectedIds = [spot.data() for spot in selectedSpots]
    else:
      selectedIds = self.regionPlots.idPlts.idsWithin(selection)

    # -----
    # Obtain table idxs corresponding to ids so rows can be highlighted
    # -----
    # Add to current selection depending on modifiers
    mode = QtCore.QItemSelectionModel.Rows
    if QtGui.QGuiApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
      mode |= QtCore.QItemSelectionModel.Select
    else:
      mode |= QtCore.QItemSelectionModel.ClearAndSelect
    selectionModel = self._compTbl.selectionModel()
    sortModel = self._compTbl.model()
    isFirst = True
    shouldScroll = len(selectedIds) > 0
    selectionList = QtCore.QItemSelection()
    for curId in selectedIds:
      idRow = np.nonzero(self._compMgr.compDf.index == curId)[0][0]
      # Map this ID to its sorted position in the list
      idxForId = sortModel.mapFromSource(self._compMgr.index(idRow, 0))
      selectionList.select(idxForId, idxForId)
      if isFirst and shouldScroll:
        self._compTbl.scrollTo(idxForId, self._compTbl.PositionAtCenter)
        isFirst = False
    # noinspection PyTypeChecker
    selectionModel.select(selectionList, mode)
    self.selectedIds = selectedIds
    self._compTbl.setFocus()
    # TODO: Better management of widget focus here

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
    compTypes = np.array(curComps.loc[:, TC.COMP_CLASS])
    curParam = self._filter[TC.COMP_CLASS.name][1]
    allowedTypes = []
    for curType in COMP_CLASSES:
      isAllowed = curParam[curType.name][0]
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
      xVerts = verts.x_flat
      yVerts = verts.y_flat
      isAllowed = np.all((xVerts >= xmin) & (xVerts <= xmax)) & \
                  np.all((yVerts >= ymin) & (yVerts <= ymax))
      vertsAllowed[vertIdx] = isAllowed
    curComps = curComps.loc[vertsAllowed,:]

    # Give self the id list of surviving comps
    self.displayedIds = curComps.index

  @Slot()
  def resetCompBounds(self):
    self.regionPlots.resetRegionList()