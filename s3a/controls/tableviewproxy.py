from typing import Union, Sequence

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from s3a.constants import PRJ_CONSTS, REQD_TBL_FIELDS, PRJ_ENUMS
from s3a.models.tablemodel import ComponentMgr
from s3a.shared import SharedAppSettings
from s3a.structures import OneDArr
from s3a.structures import XYVertices, ComplexXYVertices
from s3a.views import tableview
from s3a.views.imageareas import MainImage
from s3a.views.regions import MultiRegionPlot
from utilitys import DeferredActionStackMixin as DASM
from utilitys import EditorPropsMixin, RunOpts, ParamContainer

__all__ = ['CompSortFilter', 'CompDisplayFilter']

Signal = QtCore.Signal
QISM = QtCore.QItemSelectionModel

class CompSortFilter(EditorPropsMixin, QtCore.QSortFilterProxyModel):
  def __initEditorParams__(self, shared: SharedAppSettings):
    self.tableData = shared.tableData
    
  def __init__(self, compMgr: ComponentMgr, parent=None):
    super().__init__(parent)
    self.setSourceModel(compMgr)
    # TODO: Move code for filtering into the proxy too. It will be more efficient and
    #  easier to generalize than the current solution in CompDisplayFilter.


  def sort(self, column: int, order: QtCore.Qt.SortOrder=...) -> None:
    # Do nothing if the user is trying to sort by vertices, since the intention of
    # sorting numpy arrays is somewhat ambiguous
    noSortCols = []
    for ii, col in enumerate(self.tableData.allFields):
      if isinstance(col.value, (list, np.ndarray)):
        noSortCols.append(ii)
    if column in noSortCols:
      return
    else:
      super().sort(column, order)

  def lessThan(self, left: QtCore.QModelIndex, right: QtCore.QModelIndex) -> bool:
    # First, attempt to compare the object data
    leftObj = left.data(QtCore.Qt.EditRole)
    rightObj = right.data(QtCore.Qt.EditRole)
    try:
      return bool(np.all(leftObj < rightObj))
    except (ValueError, TypeError):
      # If that doesn't work, default to stringified comparison
      return str(leftObj) < str(rightObj)

class CompDisplayFilter(DASM, EditorPropsMixin, QtCore.QObject):
  sigCompsSelected = Signal(object)

  __groupingName__ = 'Main Image'

  def __initEditorParams__(self, shared: SharedAppSettings):
    self.props = ParamContainer()
    shared.generalProps.registerProp(PRJ_CONSTS.PROP_COMP_SEL_BHV,
                                            container=self.props)
    self.sharedAttrs = shared

  def __init__(self, compMgr: ComponentMgr, mainImg: MainImage,
               compTbl: tableview.CompTableView, parent=None):
    super().__init__(parent)
    filterEditor = self.sharedAttrs.filter
    self._mainImgArea = mainImg
    self._filter = filterEditor
    self._compTbl = compTbl
    self._compMgr = compMgr
    self.regionPlot = MultiRegionPlot()
    self.displayedIds = np.array([], dtype=int)
    self.selectedIds = np.array([], dtype=int)
    self.labelCol = REQD_TBL_FIELDS.INST_ID
    self.updateLabelCol()

    self.regionCopier = mainImg.regionCopier
    attrs = self.sharedAttrs

    with attrs.colorScheme.setBaseRegisterPath(self.regionPlot.__groupingName__):
      proc, argsParam = attrs.colorScheme.registerFunc(
        self.updateLabelCol, runOpts=RunOpts.ON_CHANGED, returnParam=True, nest=False)
    def updateLblList():
      fields = attrs.tableData.allFields
      # TODO: Filter out non-viable field types
      argsParam.child('labelCol').setLimits([f.name for f in fields])
    attrs.tableData.sigCfgUpdated.connect(updateLblList)

    # Attach to UI signals
    def _maybeRedraw():
      """
      Since an updated filter can also result from refreshed table fields, make sure not to update in that case
      (otherwise errors may occur from missing classes, etc.)
      """
      if np.array_equal(attrs.tableData.allFields, self._compMgr.compDf.columns):
        self.redrawComps()
    self._filter.sigChangesApplied.connect(_maybeRedraw)

    self.regionCopier.sigCopyStarted.connect(lambda *args: self.activateRegionCopier())
    self.regionCopier.sigCopyStopped.connect(lambda *args: self.finishRegionCopier())

    compMgr.sigCompsChanged.connect(self.redrawComps)
    compMgr.sigFieldsChanged.connect(lambda: self._reflectFieldsChanged())
    compTbl.sigSelectionChanged.connect(self._reflectTableSelectionChange)

    mainImg.addItem(self.regionPlot)
    mainImg.addItem(self.regionCopier)

    # self.filterableCols = self.findFilterableCols()

  def updateLabelCol(self, labelCol=REQD_TBL_FIELDS.INST_ID.name):
    """
    Changes the data column used to label (color) the region plot data
    :param labelCol:
      helpText: New column to use
      title: Labeling Column
      pType: list
    """
    self.labelCol = self.sharedAttrs.tableData.fieldFromName(labelCol)
    newLblData = self.labelCol.toNumeric(self._compMgr.compDf.loc[
                                           self.displayedIds, self.labelCol], rescale=True)

    self.regionPlot.regionData[PRJ_ENUMS.FIELD_LABEL] = newLblData
    self.regionPlot.updateColors()

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
    # changedIds = np.concatenate((idLists['added'], idLists['changed']))
    # self._regionPlots[changedIds, regCols] = compDf.loc[changedIds, compCols]

    # Hide all ids and table rows, since they will be reshown as needed after display filtering
    for rowIdx in range(len(compDf)):
      self._compTbl.hideRow(rowIdx)

    # Component deleted: Nothing to do, since only displayed IDs will remain in the
    # region manager anyway
    #idsToRm = idLists['deleted']

    # Update filter list: hide/unhide ids and verts as needed.
    self._updateDisplayedIds()
    # Remove all IDs that aren't displayed
    # FIXME: This isn't working correctly at the moment
    # self._regionPlots.drop(np.setdiff1d(self._regionPlots.data.index, self._displayedIds))
    self.regionPlot.resetRegionList(compDf.loc[self.displayedIds], lblField=self.labelCol)
    # noinspection PyTypeChecker
    # self._reflectTableSelectionChange(np.intersect1d(self.displayedIds, self.selectedIds))

    tblIdsToShow = np.isin(compDf.index, self.displayedIds).nonzero()[0]
    model = self._compTbl.model()
    for rowId in tblIdsToShow:
      xpondingIdx = model.mapFromSource(self._compMgr.index(rowId,0)).row()
      self._compTbl.showRow(xpondingIdx)

  @DASM.undoable('Split Components', asGroup=True)
  def splitSelectedComps(self):
    """Makes a separate component for each distinct boundary of all selected components"""
    selection = self.selectedIds

    if len(selection) == 0:
      return
    changes = self._compMgr.splitCompVertsById(selection)
    self.selectRowsById(changes['added'], QISM.ClearAndSelect)

  @DASM.undoable('Merge Components', asGroup=True)
  def mergeSelectedComps(self, keepId=-1):
    """
    Merges the selected components into one, keeping all properties of the first in the selection
    :param keepId: If specified and >0, this is the ID whose peripheral data will be retained
      during the merge. Otherwise, the first selected component is used as the keep ID.
    """
    selection = self.selectedIds

    if len(selection) < 2:
      # Nothing to do
      return
    if keepId < 0:
      keepId = selection[0]

    self._compMgr.mergeCompVertsById(selection, keepId)
    self.selectRowsById(np.array([keepId]), QISM.ClearAndSelect)

  def removeSelectedCompOverlap(self):
    """
    Makes sure all specified components have no overlap. Preference is given
    in order of the selection, i.e. the last selected component in the list
    is guaranteed to keep its full shape.
    """
    if self.selectedIds.size == 0:
      return
    self._compMgr.removeOverlapById(self.selectedIds)

  def _reflectFieldsChanged(self):
    self.redrawComps()

  def _reflectTableSelectionChange(self, selectedIds: OneDArr):
    self.selectedIds = selectedIds
    self.regionPlot.selectById(selectedIds)
    selectedComps = self._compMgr.compDf.loc[selectedIds]
    self.sigCompsSelected.emit(selectedComps)

  def scaleViewboxToSelectedIds(self, selectedIds: OneDArr=None, padding: int=None):
    """
    Rescales the main image viewbox to encompass the selection

    :param selectedIds: Ids to scale to. If *None*, this is the current selection
    :param padding: Padding around the selection. If *None*, defaults to
      pyqtgraph padding behavior
    """
    if selectedIds is None:
      selectedIds = self.selectedIds
    if len(selectedIds) == 0: return
    # Calculate how big the viewbox needs to be
    selectedVerts = self._compMgr.compDf.loc[selectedIds, REQD_TBL_FIELDS.VERTICES]
    allVerts = np.vstack([v.stack() for v in selectedVerts])
    mins = allVerts.min(0)
    maxs = allVerts.max(0)
    if padding is not None:
      mins -= padding//2
      maxs += padding//2
    vb: pg.ViewBox = self._mainImgArea.getViewBox()
    viewRect = QtCore.QRectF(*mins, *(maxs - mins))
    vb.setRange(viewRect, padding=padding)

  def selectRowsById(self, ids: Sequence[int],
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


  def reflectSelectionBoundsMade(self, selection: Union[OneDArr, XYVertices]):
    """
    :param selection: bounding box of user selection: [xmin ymin; xmax ymax]
    """
    # If min and max are the same, just check for points at mouse position
    if selection.size == 0: return
    if len(selection) == 1 or np.abs(selection[0] - selection[1]).sum() < 0.01:
      qtPoint = QtCore.QPointF(*selection[0])
      selectedSpots = self.regionPlot.pointsAt(qtPoint, self.props[PRJ_CONSTS.PROP_COMP_SEL_BHV]=='Boundary Only')
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
      # Toggle select on already active ids
      toDeselect = np.intersect1d(self.selectedIds, selectedIds)
      self.selectRowsById(toDeselect, mode|QISM.Deselect)
      selectedIds = np.setdiff1d(selectedIds, toDeselect)
      mode |= QISM.Select

    else:
      mode |= QISM.ClearAndSelect
    if not self.regionCopier.active:
      self.selectRowsById(selectedIds, mode)
      self.selectRowsById(selectedIds, mode)
    # TODO: Better management of widget focus here

  def _updateDisplayedIds(self):
    curComps = self._filter.filterCompDf(self._compMgr.compDf.copy())
    # Give self the id list of surviving comps
    self.displayedIds = curComps[REQD_TBL_FIELDS.INST_ID]
    return self.displayedIds

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
      newComps.at[idx, REQD_TBL_FIELDS.VERTICES] = ComplexXYVertices(newVerts)
    # truncatedCompIds = np.unique(truncatedCompIds)
    if self.regionCopier.inCopyMode:
      self._compMgr.addComps(newComps)
      self.activateRegionCopier(self.regionCopier.regionIds)
    else: # Move mode
      self.regionCopier.erase()
      self._compMgr.addComps(newComps, PRJ_ENUMS.COMP_ADD_AS_MERGE)
    # if len(truncatedCompIds) > 0:
    #   warn(f'Some regions extended beyond image dimensions. Boundaries for the following'
    #        f' components were altered: {truncatedCompIds}', UserWarning)

  def exportCompOverlay(self, outFile='', toClipboard=False):
    """
    :param outFile:
      pType: filepicker
      existing: False
    """
    pm = self._mainImgArea.imgItem.getPixmap()
    painter = QtGui.QPainter(pm)
    self.regionPlot.paint(painter)
    if outFile:
      # if outFile.endswith('svg'):
      #   svgr = QtSvg.QSvgRenderer(outFile)
      #   svgr.render(painter)
      #   painter.end()
      # else:
      painter.end()
      pm.save(outFile)
    if toClipboard:
      QtWidgets.QApplication.clipboard().setImage(pm.toImage())
    return pm


  def resetCompBounds(self):
    self.regionPlot.resetRegionList()