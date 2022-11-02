from __future__ import annotations

from typing import TYPE_CHECKING

from utilitys import ParamEditorPlugin, RunOpts

from ..constants import PRJ_CONSTS as CNST
from ..shared import SharedAppSettings

if TYPE_CHECKING:
    from ..models.s3abase import S3ABase
    from ..tabledata import TableData


class CompTablePlugin(ParamEditorPlugin):
    name = "Component Table"
    win: S3ABase
    tableData: TableData

    def __initEditorParams__(self, shared: SharedAppSettings, **kwargs):
        shared.generalProperties.registerProp(
            CNST.PROP_VERT_SORT_BHV, container=self.win.sortFilterProxy.props
        )
        shared.generalProperties.registerProps(
            [CNST.PROP_SCALE_PEN_WIDTH, CNST.PROP_FIELD_INFO_ON_SEL],
            container=self.win.componentController.props,
        )
        shared.colorScheme.registerFunc(
            self.win.componentController.updateLabelColumn,
            labelColumn=dict(
                type="list", limits=[f.name for f in self.win.tableData.allFields]
            ),
            runOpts=RunOpts.ON_CHANGED,
            nest=False,
            container=self.win.componentController.props,
        )
        shared.generalProperties.registerProp(
            CNST.PROP_SHOW_TBL_ON_COMP_CREATE, container=self.win.tableView.props
        )
        shared.generalProperties.registerFunc(
            self.win.tableView.setVisibleColumns,
            runOpts=RunOpts.ON_CHANGED,
            nest=False,
            returnParam=True,
            visibleColumns=[],
        )

    def attachWinRef(self, win: S3ABase):
        tbl = win.tableView
        for func, param in zip(
            [
                lambda: tbl.setSelectedCellsAsGui(),
                tbl.removeSelectedRowsGui,
                tbl.setSelectedCellsAsFirst,
                lambda: win.componentController.scaleViewboxToSelectedIds(),
            ],
            [
                CNST.TOOL_TBL_SET_AS,
                CNST.TOOL_TBL_DEL_ROWS,
                CNST.TOOL_TBL_SET_SAME_AS_FIRST,
                CNST.TOOL_TBL_ZOOM_TO_COMPS,
            ],
        ):
            param.opts["ownerObj"] = win.tableView
            self.registerFunc(func, name=param.name, btnOpts=param)
        tbl.menu = self.toolsEditor.actionsMenuFromProcs(parent=tbl, nest=True)
        self.dock.addEditors([win.tableData.filter])
        super().attachWinRef(win)
        self.tableData = win.tableData
