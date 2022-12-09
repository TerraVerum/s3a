import sys
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.parametertree import InteractiveFunction
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from qtextras import (
    DeferredActionStackMixin as DASM,
    ParameterContainer,
    ParameterEditor,
    bindInteractorOptions as bind,
    FROM_PREV_IO,
)

from ..constants import PRJ_CONSTS, PRJ_ENUMS, REQD_TBL_FIELDS
from ..models.tablemodel import ComponentManager
from ..structures import ComplexXYVertices, OneDArr, XYVertices
from ..views.fielddelegates import FieldDisplay
from ..views.imageareas import MainImage
from ..views.regions import MultiRegionPlot
from ..views.tableview import ComponentTableView

__all__ = ["ComponentSorterFilter", "ComponentController"]

Signal = QtCore.Signal
QISM = QtCore.QItemSelectionModel


class ComponentSorterFilter(QtCore.QSortFilterProxyModel):
    __groupingName__ = "Component Table"

    def __init__(self, componentManager: ComponentManager, parent=None):
        super().__init__(parent)
        self.props = ParameterContainer()
        self.props[PRJ_CONSTS.PROP_VERT_SORT_BHV] = PRJ_CONSTS.PROP_VERT_SORT_BHV.value
        self.setSourceModel(componentManager)
        # TODO: Move code for filtering into the proxy too. It will be more efficient and
        #  easier to generalize than the current solution in ComponentController.

    @property
    def verticesSortAxis(self):
        """
        Returns the column index to sort by based on whether the user wants x first or y
        """
        if self.props[PRJ_CONSTS.PROP_VERT_SORT_BHV] == "X First":
            return 0
        return 1

    def sort(self, column: int, order: QtCore.Qt.SortOrder = ...) -> None:
        # Do nothing if the user is trying to sort by vertices, since the intention of
        # sorting numpy arrays is somewhat ambiguous

        noSortCols = []
        for ii, col in enumerate(self.sourceModel().tableData.allFields):
            if isinstance(col.value, (list, np.ndarray)) and not isinstance(
                col.value, (XYVertices, ComplexXYVertices)
            ):
                noSortCols.append(ii)
        if column in noSortCols:
            return
        else:
            super().sort(column, order)

    def lessThan(self, left: QtCore.QModelIndex, right: QtCore.QModelIndex) -> bool:
        # First, attempt to compare the object data
        # For some reason, data doesn't preserve the true type so get from the source
        # model
        model = self.sourceModel()
        leftObj = model.data(left, QtCore.Qt.ItemDataRole.EditRole)
        rightObj = model.data(right, QtCore.Qt.ItemDataRole.EditRole)

        # Special case: Handle vertices
        if isinstance(leftObj, (ComplexXYVertices, XYVertices)):
            return self.lessThanVertices(leftObj, rightObj)

        # General case
        try:
            return bool(np.all(leftObj < rightObj))
        except (ValueError, TypeError):
            # If that doesn't work, default to stringified comparison
            return str(leftObj) < str(rightObj)

    def lessThanVertices(self, leftObj, rightObj):
        """Sort implementation for vertices objects"""
        if isinstance(leftObj, ComplexXYVertices):
            leftObj = leftObj.stack()
            rightObj = rightObj.stack()
        leftObj = np.min(leftObj, axis=0, initial=sys.maxsize)
        rightObj = np.min(rightObj, axis=0, initial=sys.maxsize)
        sortCol = self.verticesSortAxis
        otherCol = 1 - sortCol
        return leftObj[sortCol] < rightObj[sortCol] or (
            leftObj[sortCol] == rightObj[sortCol]
            and leftObj[otherCol] < rightObj[otherCol]
        )


class ComponentController(DASM, QtCore.QObject):
    sigComponentsSelected = Signal(object)

    __groupingName__ = "Main Image"

    def __init__(
        self,
        componentManager: ComponentManager,
        mainImage: MainImage,
        componentTable: ComponentTableView,
        parent=None,
    ):
        super().__init__(parent)
        self.props = ParameterContainer()
        for prop in PRJ_CONSTS.PROP_SCALE_PEN_WIDTH, PRJ_CONSTS.PROP_FIELD_INFO_ON_SEL:
            self.props[prop] = prop.value

        self.tableData = componentManager.tableData
        self._mainImageArea = mainImage
        self._filter = self.tableData.filter
        self._componentTable = componentTable
        self._componentManager = componentManager
        self.regionPlot = MultiRegionPlot(disableMouseClick=True)
        self.displayedIds = np.array([], dtype=int)
        self.selectedIds = np.array([], dtype=int)
        self.labelColumn = REQD_TBL_FIELDS.ID
        self.updateLabelColumn()

        self._regionIntersectionCache: Tuple[
            Optional[np.ndarray], Optional[np.ndarray]
        ] = (None, None)
        """
        Checking whether a region intersction occurred is expensive when several thousand
        regions exist. Results are cached until the region plot changes. "lru_cache"
        could be used, except "selection" is not a hashable argument. The primitive
        solution is to simply preserve the cache across at most one "selection" value
        """

        componentManager.sigUpdatedFocusedComponent.connect(
            self._onFocusedComponentChange
        )

        # Attach to UI signals
        def _maybeRedraw():
            """
            Since an updated filter can also result from refreshed table fields,
            make sure not to update in that case (otherwise errors may occur from
            missing classes, etc.)
            """
            if np.array_equal(
                self.tableData.allFields, self._componentManager.compDf.columns
            ):
                self.redrawComponents()

        self._filter.applyButton.clicked.connect(_maybeRedraw)

        self.regionMover.sigMoveStarted.connect(
            lambda *args: self.activateRegionCopier()
        )
        self.regionMover.sigMoveStopped.connect(lambda *args: self.finishRegionCopier())

        componentManager.sigComponentsChanged.connect(self.redrawComponents)
        componentManager.sigFieldsChanged.connect(self._reflectFieldsChanged)
        componentTable.sigSelectionChanged.connect(self._reflectTableSelectionChange)

        mainImage.addItem(self.regionPlot)
        mainImage.addItem(self.regionMover.manipRoi)
        self.vb = mainImage.getViewBox()
        self.vb.sigRangeChanged.connect(self.recomputePenWidth)

        self.fieldDisplay = FieldDisplay(mainImage)
        self.fieldsShowing = False
        self.fieldInfoProc = self._createFieldDisplayProcess()
        self.fieldDisplay.callDelegateFunction("hide")
        # Populate initial field options
        self._reflectFieldsChanged()

    def _onFocusedComponentChange(self, newComp: pd.Series):
        self.regionPlot.focusById(np.array([newComp[REQD_TBL_FIELDS.ID]]))

    def _createFieldDisplayProcess(self):
        io = {}
        interactor = ParameterEditor.defaultInteractor
        for deleg in self.fieldDisplay.availableDelegates.values():
            delegIo = interactor.functionToParameterDict(deleg.setData)
            useIo = {
                ch["name"]: ch
                for ch in delegIo["children"]
                if ch["value"] is not FROM_PREV_IO
            }
            io.update(useIo)
        toReturn = InteractiveFunction(self.showFieldInfoById)
        interactor(toReturn, **io)
        return toReturn

    def recomputePenWidth(self):
        if not self.props[PRJ_CONSTS.PROP_SCALE_PEN_WIDTH]:
            return
        newWidth = np.ceil(max(1 / min(self.vb.viewPixelSize()), 1))
        if newWidth == 1:
            # Performance gains
            newWidth = 0
        self.regionPlot.props["penWidth"] = newWidth

    def updateLabelColumn(self, labelColumn=REQD_TBL_FIELDS.ID.name):
        """
        Changes the data column used to label (color) the region plot data
        """
        self.labelColumn = self.tableData.fieldFromName(labelColumn)
        newLblData = self.labelColumn.toNumeric(
            self._componentManager.compDf.loc[self.displayedIds, self.labelColumn],
            rescale=True,
        )

        self.regionPlot.regionData[PRJ_ENUMS.FIELD_LABEL] = newLblData
        self.regionPlot.updateColors()

    def redrawComponents(self, idLists=None):
        # Following mix of cases are possible:
        # Components: DELETED, UNCHANGED, CHANGED, NEW
        # New is different from changed since id plot already exists (unhide vs. create)
        # Plots: DRAWN, UNDRAWN
        # Note that hiding the ID is chosen instead of deleting, since that is a costly
        # graphics operation
        compDf = self._componentManager.compDf

        # Invalidate selection cache
        self._regionIntersectionCache = (None, None)

        # Update and add changed/new components
        # TODO: Find out why this isn't working. For now, just reset the whole comp list
        #  each time components are changed, since the overhead isn't too terrible.

        # Note that components that were visible but then deleted shouldn't trigger
        # false positives
        previouslyVisible = np.intersect1d(self.displayedIds, compDf.index)

        # Update filter list: hide/unhide ids and vertices as needed.
        self._updateDisplayedIds()
        self.regionPlot.resetRegionList(
            compDf.loc[self.displayedIds], labelField=self.labelColumn
        )

        tblIdsToShow = np.isin(compDf.index, self.displayedIds).nonzero()[0]
        # Don't go through the effort of showing an already visible row
        tblIdsToShow = np.setdiff1d(tblIdsToShow, previouslyVisible)
        model = self._componentTable.model()
        for rowId in tblIdsToShow:
            xpondingIdx = model.mapFromSource(
                self._componentManager.index(rowId, 0)
            ).row()
            self._componentTable.showRow(xpondingIdx)

        # Hide no longer visible components
        for rowId in np.setdiff1d(previouslyVisible, self.displayedIds):
            xpondingIdx = model.mapFromSource(
                self._componentManager.index(rowId, 0)
            ).row()
            self._componentTable.hideRow(xpondingIdx)

    @DASM.undoable("Split Components", asGroup=True)
    def splitSelectedComponents(self):
        """
        Makes a separate component for each distinct boundary of all selected
        components
        """
        selection = self.selectedIds

        if len(selection) == 0:
            return
        changes = self._componentManager.splitById(selection)
        self.selectRowsById(changes["added"], QISM.ClearAndSelect)

    @DASM.undoable("Merge Components", asGroup=True)
    def mergeSelectedComponents(self, keepId=-1):
        """
        Merges the selected components into one, keeping all properties of the first in
        the selection

        Parameters
        ----------
        keepId
            If specified and >0, this is the ID whose peripheral data will be retained
            during the merge. Otherwise, the first selected component is used as the
            keep ID.
        """
        selection = self.selectedIds

        if len(selection) < 2:
            # Nothing to do
            return
        if keepId < 0:
            keepId = selection[0]

        self._componentManager.mergeById(selection, keepId)
        self.selectRowsById(np.array([keepId]), QISM.ClearAndSelect)

    def removeSelectedComponentOverlap(self):
        """
        Makes sure all specified components have no overlap. Preference is given
        in order of the selection, i.e. the last selected component in the list
        is guaranteed to keep its full shape.
        """
        if self.selectedIds.size == 0:
            return
        self._componentManager.removeOverlapById(self.selectedIds)

    def _reflectFieldsChanged(self):
        fields = self.tableData.allFields
        # TODO: Filter out non-viable field types
        if "labelColumn" in self.props.parameters:
            self.props.parameters["labelColumn"].setLimits([f.name for f in fields])

        self.redrawComponents()

    def _reflectTableSelectionChange(self, selectedIds: OneDArr):
        self.selectedIds = selectedIds
        # Silently update selected ids since focusById will force another graphic update
        self.regionPlot.updateSelectedAndFocused(
            selectedIds=selectedIds, updatePlot=True
        )
        selectedComps = self._componentManager.compDf.loc[selectedIds]
        self.sigComponentsSelected.emit(selectedComps)
        if self.props[PRJ_CONSTS.PROP_FIELD_INFO_ON_SEL]:
            self.fieldInfoProc(ids=selectedIds, force=True)

    def scaleViewboxToSelectedIds(
        self, selectedIds: OneDArr = None, paddingPct: float = 0.1
    ):
        """
        Rescales the main image viewbox to encompass the selection

        Parameters
        ----------
        selectedIds
            Ids to scale to. If *None*, this is the current selection
        paddingPct
            Padding around the selection. If *None*, defaults to pyqtgraph padding
            behavior
        """
        if selectedIds is None:
            selectedIds = self.selectedIds
        if len(selectedIds) == 0:
            return
        # Calculate how big the viewbox needs to be
        selectedVerts = self._componentManager.compDf.loc[
            selectedIds, REQD_TBL_FIELDS.VERTICES
        ]
        allVerts = np.vstack([v.stack() for v in selectedVerts])
        mins = allVerts.min(0)
        maxs = allVerts.max(0)
        vb: pg.ViewBox = self._mainImageArea.getViewBox()
        viewRect = QtCore.QRectF(*mins, *(maxs - mins))
        vb.setRange(viewRect, padding=paddingPct)

    def selectRowsById(
        self,
        ids: Sequence[int],
        selectionMode=QISM.Rows | QISM.ClearAndSelect,
        onlyEditableRetList=True,
    ):
        selectionModel = self._componentTable.selectionModel()
        sortModel = self._componentTable.model()
        isFirst = True
        shouldScroll = len(ids) > 0
        selectionList = QtCore.QItemSelection()
        retLists = []  # See tableview ids_rows_colsFromSelection
        if onlyEditableRetList:
            selectedCols = self._componentManager.editColIdxs
        else:
            selectedCols = np.arange(len(self._componentManager.columnTitles))
        ids = np.intersect1d(ids, self._componentManager.compDf.index)
        for curId in ids:
            idRow = np.nonzero(self._componentManager.compDf.index == curId)[0][0]
            # Map this ID to its sorted position in the list
            idxForId = sortModel.mapFromSource(self._componentManager.index(idRow, 0))
            selectionList.select(idxForId, idxForId)
            if isFirst and shouldScroll:
                self._componentTable.scrollTo(
                    idxForId, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter
                )
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

    @bind(
        ids=dict(ignore=True),
        fields=dict(type="checklist", limits=[], expanded=False),
        force=dict(ignore=True),
    )
    def showFieldInfoById(self, ids=None, fields=None, force=False, **kwargs):
        if not self.fieldsShowing and not force:
            return
        if not fields:
            self.fieldDisplay.callDelegateFunction("clear")
            # Sometimes artifacts are left on the scene at this point
            self._mainImageArea.scene().update()
            return

        if ids is None:
            ids = self.selectedIds
        comps = self._componentManager.compDf.loc[ids]
        self.fieldDisplay.showFieldData(comps, fields, **kwargs)
        self.fieldsShowing = True

    def toggleFieldInfoDisplay(self):
        func = "hide" if self.fieldsShowing else "show"
        self.fieldDisplay.callDelegateFunction(func)
        self.fieldsShowing = not self.fieldsShowing
        # May need to refresh data
        if func == "show" and not self.fieldDisplay.inUseDelegates:
            # Using the proc maintains user settings
            self.fieldInfoProc()

    def reflectSelectionBoundsMade(
        self,
        selection: Union[OneDArr, XYVertices],
        clearExisting=True,
    ):
        """
        Called when user makes a selection in the main image area

        Parameters
        ----------
        selection
            bounding box of user selection: [xmin ymin; xmax ymax]
        clearExisting
            If ``True``, already selected points are cleared before this selection is
            incorporated
        """
        # If min and max are the same, just check for points at mouse position
        if selection.size == 0:
            return
        selectedIds = self.regionPlot.boundsWithin(selection)

        # -----
        # Obtain table idxs corresponding to ids so rows can be highlighted
        # ---`--
        # Add to current selection depending on modifiers
        mode = QISM.Rows
        if (
            not clearExisting
            or QtGui.QGuiApplication.keyboardModifiers()
            == QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            # Toggle select on already active ids
            toDeselect = np.intersect1d(self.selectedIds, selectedIds)
            self.selectRowsById(toDeselect, mode | QISM.Deselect)
            selectedIds = np.setdiff1d(selectedIds, toDeselect)
            mode |= QISM.Select

        else:
            mode |= QISM.ClearAndSelect
        if not self.regionMover.active:
            self.selectRowsById(selectedIds, mode)
        # TODO: Better management of widget focus here

    def selectionIntersectsRegion(self, selection):
        cache = self._regionIntersectionCache
        result = False
        if cache[0] is not None and np.array_equal(cache[0], selection):
            result = cache[-1]
        elif len(self.regionPlot.boundsWithin(selection)):
            result = True
        else:
            for pt in selection:
                if len(self.regionPlot.pointsAt(pt)):
                    result = True
                    break
        self._regionIntersectionCache = (selection, result)
        return result

    def _updateDisplayedIds(self):
        curComps = self._filter.filterComponentDf(self._componentManager.compDf.copy())
        # Give self the id list of surviving components
        self.displayedIds = curComps[REQD_TBL_FIELDS.ID]
        return self.displayedIds

    def activateRegionCopier(self, selectedIds: OneDArr = None):
        if selectedIds is None:
            selectedIds = self.selectedIds
        if len(selectedIds) == 0:
            return
        comps = self._componentManager.compDf.loc[selectedIds].copy()
        self.regionMover.resetBaseData(comps)
        self.regionMover.active = True

    def finishRegionCopier(self, keepResult=True):
        if not keepResult:
            return
        newComps = self.regionMover.baseData
        # TODO: Truncate vertices that lie outside image boundaries
        # Invalid if any vertices are outside image bounds
        # truncatedCompIds = []
        # imShape_xy = self._mainImageArea.image.shape[:2][::-1]
        for idx in newComps.index:
            verts = newComps.at[idx, REQD_TBL_FIELDS.VERTICES].removeOffset(
                self.regionMover.dataMin
            )
            newComps.at[
                idx, REQD_TBL_FIELDS.VERTICES
            ] = self.regionMover.transformedData(verts)
        # truncatedCompIds = np.unique(truncatedCompIds)
        if self.regionMover.inCopyMode:
            change = self._componentManager.addComponents(newComps)
            self.activateRegionCopier(change["added"])
        else:  # Move mode
            self.regionMover.erase()
            self._componentManager.addComponents(
                newComps, PRJ_ENUMS.COMPONENT_ADD_AS_MERGE
            )

    @bind(file=dict(type="file", fileMode="AnyFile"))
    def exportComponentOverlay(self, file="", toClipboard=False):
        """
        Exports the current component overlay to a file or clipboard

        Parameters
        ----------
        file : str | Path
            File to save to. If empty, no file is saved
        toClipboard : bool
            If ``True``, the image is copied to the clipboard
        """
        oldShowFocused = self.regionPlot.showFocused
        oldShowSelected = self.regionPlot.showSelected

        pm = self._mainImageArea.imageItem.getPixmap()
        painter = QtGui.QPainter(pm)
        try:
            self.regionPlot.showFocused = True
            self.regionPlot.showSelected = True
            self.regionPlot.updatePlot()
            self.regionPlot.paint(painter)
        finally:
            self.regionPlot.showFocused = oldShowFocused
            self.regionPlot.showSelected = oldShowSelected
            self.regionPlot.updatePlot()
        if file:
            # if file.endswith('svg'):
            #   svgr = QtSvg.QSvgRenderer(file)
            #   svgr.render(painter)
            #   painter.end()
            # else:
            painter.end()
            pm.save(file)
        if toClipboard:
            QtWidgets.QApplication.clipboard().setImage(pm.toImage())
        return pm

    @property
    def regionMover(self):
        return self._mainImageArea.regionMover
