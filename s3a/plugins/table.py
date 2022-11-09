from __future__ import annotations

from typing import TYPE_CHECKING

from pyqtgraph.parametertree import RunOptions
from qtextras import ParameterEditor

from .base import ParameterEditorPlugin
from ..constants import PRJ_CONSTS as CNST
from ..shared import SharedAppSettings

if TYPE_CHECKING:
    from ..tabledata import TableData


class ComponentTablePlugin(ParameterEditorPlugin):
    name = "Component Table"
    tableData: TableData

    def __initSharedSettings__(self, shared: SharedAppSettings = None, **kwargs):
        shared.generalProperties.registerParameter(
            CNST.PROP_VERT_SORT_BHV, container=self.window.sortFilterProxy.props
        )
        shared.generalProperties.registerParameterList(
            [CNST.PROP_SCALE_PEN_WIDTH, CNST.PROP_FIELD_INFO_ON_SEL],
            container=self.window.componentController.props,
        )
        shared.colorScheme.registerFunction(
            self.window.componentController.updateLabelColumn,
            labelColumn=dict(
                type="list", limits=[f.name for f in self.tableData.allFields]
            ),
            runOptions=RunOptions.ON_CHANGED,
            nest=False,
            container=self.window.componentController.props,
        )
        shared.generalProperties.registerParameter(
            CNST.PROP_SHOW_TBL_ON_COMP_CREATE, container=self.window.tableView.props
        )
        shared.generalProperties.registerFunction(
            self.window.tableView.setVisibleColumns,
            runOptions=RunOptions.ON_CHANGED,
            nest=False,
            visibleColumns=[],
        )
        super().__initSharedSettings__(shared, **kwargs)

    def attachToWindow(self, window):
        tbl = window.tableView
        self.tableData = window.tableData
        for func, param in zip(
            [
                lambda: tbl.setSelectedCellsAsGui(),
                tbl.removeSelectedRowsGui,
                tbl.setSelectedCellsAsFirst,
                lambda: window.componentController.scaleViewboxToSelectedIds(),
            ],
            [
                CNST.TOOL_TBL_SET_AS,
                CNST.TOOL_TBL_DEL_ROWS,
                CNST.TOOL_TBL_SET_SAME_AS_FIRST,
                CNST.TOOL_TBL_ZOOM_TO_COMPS,
            ],
        ):
            # Optionally scope shortcuts to only work in the table widget
            # param.opts["ownerWidget"] = tbl
            self.registerFunction(func, name=param.name, runActionTemplate=param)
        tbl.menu = self.createActionsFromFunctions(stealShortcuts=False)
        filter_: ParameterEditor = self.tableData.filter
        self.tableData = window.tableData
        super().attachToWindow(window)
        self.createDockWithoutFunctionMenu(filter_)
        self.registeredEditors.append(filter_)
