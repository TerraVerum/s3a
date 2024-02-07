from __future__ import annotations

from typing import Any, Sequence
from warnings import warn

import numpy as np
import pandas as pd

import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.Parameter import PARAM_TYPES
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..compio.helpers import deserialize, serialize
from ..constants import PRJ_CONSTS, PRJ_ENUMS, REQD_TBL_FIELDS
from ..generalutils import enumConverter
from ..models.tablemodel import ComponentManager
from ..structures import TwoDArr
from ..tabledata import TableData

__all__ = ["ComponentTableView", "PopupTableDialog"]

from qtextras import (
    DeferredActionStackMixin as DASM,
    OptionsDict,
    ParameterContainer,
    ParameterEditor,
    PgParameterDelegate,
    PgPopupDelegate,
    bindInteractorOptions as bind,
)

Signal = QtCore.Signal


class SerDesDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, param: OptionsDict, asLineEdit=False, parent=None):
        """
        Creates a string-editable item delegate which uses ComponentIO's serialize and
        deserialize methods to convert to actual values.

        Parameters
        ----------
        param
            OptionsDict which dictates the serialization/deserialization of this value
        asLineEdit
            Whether to give this delegate a line editor text edit to set the serialized
            form of value
        parent
            Parent widget
        """
        super().__init__(parent)
        self.asLineEdit = asLineEdit
        self.param = param

    def createEditor(self, parent, option, index: QtCore.QModelIndex):
        editorType = "str" if self.asLineEdit else "text"
        pgParam = Parameter.create(
            name="dummy",
            type=editorType,
            value=index.data(QtCore.Qt.ItemDataRole.DisplayRole),
        )
        editor = pgParam.itemClass(pgParam, 0).makeWidget()
        editor.setParent(parent)
        editor.setMaximumSize(option.rect.width(), option.rect.height())
        return editor

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractTableModel,
        index: QtCore.QModelIndex,
    ):
        values, errs = deserialize(self.param, [editor.value()])
        if not len(values):
            raise ValueError(f"Error during deserialize:\n{errs[0]}")
        value = values[0]
        model.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)

    def setEditorData(self, editor: QtWidgets.QWidget, index):
        value = index.data(QtCore.Qt.ItemDataRole.EditRole)
        strVals, strErrs = serialize(self.param, [value])
        if len(strErrs):
            raise ValueError(f"Error during serialize:\n{strErrs[0]}")
        strVal = strVals[0]
        editor.setValue(strVal)

    def updateEditorGeometry(
        self,
        editor: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ):
        editor.setGeometry(option.rect)


class MinimalTableModel(ComponentManager):
    """
    The CheckState flag is only possible on a minimal popup table, but Qt makes it
    challenging to override just this portion of the ComponentManager class. So,
    this simple subclass maintains a checkstate and checkable columns on top of the
    traditional component manager.
    """

    def __init__(self):
        super().__init__()
        # Map true/false to check state
        self.csMap = {
            True: enumConverter(QtCore.Qt.CheckState.Checked),
            False: enumConverter(QtCore.Qt.CheckState.Unchecked),
        }

        # Since the minimal model only contains one row, everything checkable can be on a
        # per-column basis
        self.checkedColIdxs = set()

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        return super().flags(index) | QtCore.Qt.ItemFlag.ItemIsUserCheckable

    def data(self, index: QtCore.QModelIndex, role) -> Any:
        if role == QtCore.Qt.ItemDataRole.CheckStateRole:
            return self.csMap[index.column() in self.checkedColIdxs]
        return super().data(index, role)

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole) -> bool:
        if role == QtCore.Qt.ItemDataRole.CheckStateRole:
            if value == enumConverter(QtCore.Qt.CheckState.Checked):
                self.checkedColIdxs.add(index.column())
            else:
                self.checkedColIdxs.remove(index.column())
            return True
        yield from super().setData(index, value, role)


class PopupTableDialog(QtWidgets.QDialog):
    def __init__(self, *args):
        super().__init__(*args)
        self.setModal(True)
        # -----------
        # Table View
        # -----------
        self.model = MinimalTableModel()
        self.titles = []
        self.tableView = ComponentTableView(manager=self.model, minimal=True)
        self.reflectDelegateChange()

        # -----------
        # Warning Message
        # -----------
        self.warnLbl = QtWidgets.QLabel(
            "Check the boxes for data you wish to propagate to all selected cells.\n"
            "Selections where every row has the same data are checkd by default.",
            self,
        )
        self.warnLbl.setStyleSheet("font-weight: bold; color:red; font-size:14")

        # -----------
        # Widget buttons
        # -----------
        self.applyButton = QtWidgets.QPushButton("Apply")
        self.closeButton = QtWidgets.QPushButton("Close")

        # -----------
        # Widget layout
        # -----------
        btnLayout = QtWidgets.QHBoxLayout()
        btnLayout.addWidget(self.applyButton)
        btnLayout.addWidget(self.closeButton)

        centralLayout = QtWidgets.QVBoxLayout()
        centralLayout.addWidget(self.warnLbl)
        centralLayout.addWidget(self.tableView)
        centralLayout.addLayout(btnLayout)
        self.setLayout(centralLayout)
        self.setMinimumWidth(self.tableView.width())

        # -----------
        # UI Element Signals
        # -----------
        self.closeButton.clicked.connect(self.close)
        self.applyButton.clicked.connect(self.accept)

    def reflectDelegateChange(self):
        # TODO: Find if there's a better way to see if changes happen in a table
        self.titles = np.array(list([f.name for f in self.model.tableData.allFields]))
        self.tableView.setColDelegates()

        # Avoid loop scoping
        def wrapper(col):
            def onUpdate():
                self.model.checkedColIdxs.add(col)

            return onUpdate

        for colIdx in range(len(self.titles)):
            deleg = self.tableView.itemDelegateForColumn(colIdx)
            deleg.commitData.connect(wrapper(colIdx))

    @property
    def data(self):
        return self.tableView.manager.compDf.iloc[[0], :]

    def setData(
        self,
        componentDf: pd.DataFrame,
        colIdxs: Sequence[int],
        dirtyColIdxs: Sequence[int] = None,
    ):
        # New selection, so reset dirty columns
        if dirtyColIdxs is None:
            dirtyColIdxs = []
        # Hide columns that weren't selected by the user since these changes
        # Won't propagate
        self.model.checkedColIdxs.clear()
        self.model.checkedColIdxs.update(dirtyColIdxs)

        for ii in range(len(self.titles)):
            if ii not in colIdxs:
                self.tableView.hideColumn(ii)
            else:
                self.tableView.showColumn(ii)
        self.tableView.manager.removeComponents()
        self.tableView.manager.addComponents(
            componentDf, addType=PRJ_ENUMS.COMPONENT_ADD_AS_MERGE
        )
        return True

    def reject(self):
        # On dialog close be sure to unhide all columns / reset dirty cols
        for ii in range(len(self.titles)):
            self.tableView.showColumn(ii)
        super().reject()


class ComponentTableView(DASM, QtWidgets.QTableView):
    __groupingName__ = "Component Table"
    """
    Table for displaying :class:`ComponentManager` data.
    """
    sigSelectionChanged = Signal(object)

    def __init__(self, *args, minimal=False, manager: ComponentManager | Any = None):
        """
        Creates the table.

        Parameters
        ----------
        minimal
            Whether to make a table with minimal features. If false, creates extensible
            table with context menu options. Otherwise, only contains minimal features.
        """
        super().__init__(*args)

        self.props = ParameterContainer()
        prop = PRJ_CONSTS.PROP_SHOW_TBL_ON_COMP_CREATE
        self.props[prop] = prop.value

        self.toolsEditor = ParameterEditor(name="Component Table Tools")

        self.setStyleSheet(
            "QTableView { selection-color: white; selection-background-color: #0078d7; }"
        )
        self._previouslySelectedRows = np.array([])
        self.setSortingEnabled(True)

        self.manager = None
        self.instanceIdIndex = 0
        self.minimal = minimal
        # self.setColDelegates()

        if not minimal:
            self.popup = PopupTableDialog(*args)
            # Create context menu for changing table rows
            self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            cursor = QtGui.QCursor()
            self.customContextMenuRequested.connect(
                lambda: self.menu.exec_(cursor.pos())
            )
        self.setModel(manager or ComponentManager())
        self._onTableChange()

    def _onTableChange(self, *_args):
        self.setColDelegates()
        if not self.minimal:
            self.popup.reflectDelegateChange()
        if not self.tableData or "visibleColumns" not in self.props.parameters:
            # Possible for this view to be used without registering visible columns
            # as a parameter, or without setting table information
            return
        lims = []
        val = []
        for f in self.tableData.allFields:
            name = f.name
            if f.opts.get("visible", True):
                val.append(name)
            lims.append(name)
        lims = [f.name for f in self.tableData.allFields]
        self.props.parameters["visibleColumns"].setOpts(limits=lims, value=val)

    @bind(visibleColumns=dict(type="checklist", expanded=False))
    def setVisibleColumns(self, visibleColumns: Sequence[str]):
        """
        Determines which columns to show. All unspecified columns will be hidden.
        """
        for ii, col in enumerate(self.manager.columnTitles):
            self.setColumnHidden(ii, col not in visibleColumns)

    def setColDelegates(self):
        if not self.tableData:
            return
        for ii, field in enumerate(self.tableData.allFields):
            curType = field.type.lower()
            curval = field.value
            paramDict = dict(type=curType, default=curval, **field.opts)
            if curType == "enum":
                paramDict["type"] = "list"
                paramDict.update(limits=list(type(curval)))
            elif curType == "OptionsDict":
                paramDict["type"] = "list"
                paramDict.update(limits=list(curval.group))
            elif curType == "bool":
                # TODO: Get checkbox to stay in table after editing for a smoother
                #  appearance. For now, the easiest solution is to just use dropdown
                paramDict["type"] = "list"
                paramDict.update(limits={"True": True, "False": False})
            try:
                self.setItemDelegateForColumn(ii, PgParameterDelegate(paramDict, self))
            except TypeError:
                if paramDict["type"] in PARAM_TYPES:
                    paramDict["name"] = field.name
                    self.setItemDelegateForColumn(ii, PgPopupDelegate(paramDict, self))
                else:
                    # Parameter doesn't have a registered pyqtgraph editor, so default to
                    # generic serdes editor
                    self.setItemDelegateForColumn(
                        ii, SerDesDelegate(field, parent=self)
                    )

        self.horizontalHeader().setSectionsMovable(True)

    # When the model is changed, get a reference to the ComponentManager
    def setModel(self, modelOrProxy: QtCore.QAbstractTableModel):
        super().setModel(modelOrProxy)
        if self.tableData:
            pg.disconnect(self.tableData.sigConfigUpdated, self._onTableChange)
        try:
            # If successful we were given a proxy model
            self.manager = modelOrProxy.sourceModel()
        except AttributeError:
            self.manager = modelOrProxy
        self.manager: ComponentManager
        if not self.minimal:
            self.popup.model.tableData = self.manager.tableData
            self.popup.reflectDelegateChange()
        self.tableData.sigConfigUpdated.connect(self._onTableChange)
        self.instanceIdIndex = self.tableData.allFields.index(REQD_TBL_FIELDS.ID)

    @property
    def tableData(self) -> TableData | None:
        return self.manager.tableData if self.manager else None

    def selectionChanged(
        self, curSel: QtCore.QItemSelection, prevSel: QtCore.QItemSelection
    ):
        """
        When the selected rows in the table change, this retrieves the corresponding
        previously and newly selected IDs. They are then emitted in sigSelectionChanged.
        """
        super().selectionChanged(curSel, prevSel)
        selectedIds = []
        selection = self.selectionModel().selectedIndexes()
        for item in selection:
            selectedIds.append(
                item.sibling(item.row(), self.instanceIdIndex).data(
                    QtCore.Qt.ItemDataRole.EditRole
                )
            )
        newRows = pd.unique(np.array(selectedIds))
        if np.array_equal(newRows, self._previouslySelectedRows):
            return
        self._previouslySelectedRows = newRows
        self.sigSelectionChanged.emit(newRows)

    def removeSelectedRowsGui(self):
        if self.minimal:
            return

        idList = [
            idx.siblingAtColumn(self.instanceIdIndex).data(
                QtCore.Qt.ItemDataRole.EditRole
            )
            for idx in self.selectedIndexes()
        ]
        if len(idList) == 0:
            return
        # Make sure the user actually wants this
        idList = pd.unique(np.array(idList))
        dlg = QtWidgets.QMessageBox()
        btnType = dlg.StandardButton
        confirm = dlg.question(
            self,
            "Remove Rows",
            f"Are you sure you want to remove {len(idList)} selected row(s)?",
            btnType.Yes | btnType.Cancel,
            btnType.Yes,
        )
        if confirm == btnType.Yes:
            # Proceed with operation
            # Since each selection represents a row, remove duplicate row indices
            self.manager.removeComponents(idList)
            self.clearSelection()

    def idsRowsColumnsFromSelection(
        self,
        excludeNoEditColumns=True,
        warnNoneSelection=True,
        selectedIdndexes: Sequence[QtCore.QModelIndex] = None,
    ):
        """
        Returns Nx3 np array of (ids, rows, cols) from current table selection. Ids
        refer to the 'Instance ID' column, while rows refer to the placement in the
        visible table. Thus, sorting the table will not affect `id` values, but will
        affect `row` values.

        Parameters
        ----------
        excludeNoEditColumns
            Whether to consider columns that do not allow editing, like Instance ID and
            Vertices
        warnNoneSelection
            Whether to raise a warning if no values are selected
        selectedIdndexes
            Selection to get ids, rows, and cols from. If *None*, defaults to the
            current table selection (self.selectedIndexes())
        """
        if selectedIdndexes is None:
            selectedIdndexes = self.selectedIndexes()
        retLists = []  # (Ids, rows, cols)
        for idx in selectedIdndexes:
            row = idx.row()
            # 0th row contains instance ID
            # TODO: If the user is allowed to reorder columns this needs to be revisited
            idAtIdx = idx.siblingAtColumn(self.instanceIdIndex).data(
                QtCore.Qt.ItemDataRole.EditRole
            )
            retLists.append([idAtIdx, row, idx.column()])
        retLists = np.array(retLists, dtype=int)
        if excludeNoEditColumns and len(retLists) > 0:
            # Set diff will eliminate any repeats, so use a slower op that at least
            # preserves duplicates
            retLists = retLists[~np.isin(retLists[:, 2], self.manager.noEditColIdxs)]
        if len(retLists) == 0:
            retLists.shape = (-1, 3)
        if warnNoneSelection and len(retLists) == 0:
            warn("No editable columns selected.", UserWarning, stacklevel=2)
        return retLists

    def setSelectedCellsAsFirst(self):
        """
        Sets all cells in the selection to be the same as the first row in the selection.
        See the project wiki for a detailed description
        """
        selection = self.idsRowsColumnsFromSelection()
        if len(selection):
            overwriteData = self.manager.compDf.loc[selection[0, 0]]
            self.setSelectedCellsAs(selection, overwriteData)

    def setSelectedCellsAsGui(self, selectionIndexes: TwoDArr = None):
        """
        Sets all cells in the selection to the values specified in the popup table. See
        the project wiki for a detailed description
        """
        if selectionIndexes is None:
            selectionIndexes = self.idsRowsColumnsFromSelection()
        if len(selectionIndexes) == 0:
            return
        overwriteData = self.manager.compDf.loc[[selectionIndexes[0, 0]]].copy()
        with self.actionStack.ignoreActions():
            self.popup.setData(
                overwriteData,
                pd.unique(selectionIndexes[:, 2]),
                self._getDuplicateDataColumns(selectionIndexes),
            )
            wasAccepted = self.popup.exec_()
        # Convert to list or isin() check below fails
        dirtyCols = list(self.popup.model.checkedColIdxs)
        if not wasAccepted or not dirtyCols:
            return

        selectionIndexes = selectionIndexes[np.isin(selectionIndexes[:, 2], dirtyCols)]
        self.setSelectedCellsAs(selectionIndexes, self.popup.data)

    def _getDuplicateDataColumns(self, selectionIndexes: np.ndarray) -> list[int]:
        """
        From a selection of components, returns the column names which have the same
        data in every row. This is useful for determining which columns should be
        propagated by default when performing a "set cells" operation

        Parameters
        ----------
        selectionIndexes
            Selection list of dataframe cells to consider. See
            ``ids_rows_colsFormSelection`` for a description of this array

        Returns
        -------
        list[int]
            List of column idxs which have only duplicate data selected, i.e. all
            entries in every row of that column are the same
        """
        dupCols = []
        # First, find out all rows for each column to select, then test data at those rows
        for col in np.unique(selectionIndexes[:, 2]):
            name = self.manager.compDf.columns[col]
            ids = selectionIndexes[selectionIndexes[:, 2] == col, 0]
            data = self.manager.compDf.loc[ids, name]
            try:
                numDups = len(pd.unique(data))
            except TypeError:
                # Occurs for unhashable types
                numDups = 2
            if numDups <= 1:
                dupCols.append(col)
        return dupCols

    def setSelectedCellsAs(
        self, selectionIndexes: TwoDArr, overwriteData: pd.DataFrame
    ):
        """
        Overwrites the data from rows and cols with the information in ``overwriteData``.
        Each (id, row, col) index is treated as a single index

        Parameters
        ----------
        selectionIndexes
            Selection idxs to overwrite. If *None*, defaults to current selection.
        overwriteData
            What to fill in the overwrite locations. If *None*, a popup table is
            displayed and its data is used.
        """
        if self.minimal:
            return

        if len(selectionIndexes) == 0:
            return
        overwriteData = overwriteData.squeeze()
        uniqueIds = pd.unique(selectionIndexes[:, 0])
        newDataDf = self.manager.compDf.loc[uniqueIds].copy()
        # New data ilocs will no longer match, fix this using loc + indexed columns
        colsForLoc = self.manager.compDf.columns[selectionIndexes[:, 2]]
        for idxTriplet, colForLoc in zip(selectionIndexes, colsForLoc):
            newDataDf.at[idxTriplet[0], colForLoc] = overwriteData.iat[idxTriplet[2]]
        self.manager.addComponents(newDataDf, addType=PRJ_ENUMS.COMPONENT_ADD_AS_MERGE)
