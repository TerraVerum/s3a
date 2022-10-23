from __future__ import annotations

from utilitys import ParamEditorPlugin

from ..constants import PRJ_CONSTS as CNST
from ..models.s3abase import S3ABase
from ..shared import SharedAppSettings


class CompTablePlugin(ParamEditorPlugin):
    name = "Component Table"

    def __initEditorParams__(self, shared: SharedAppSettings):
        super().__initEditorParams__()
        self.dock.addEditors([shared.filter])

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
        super().attachWinRef(win)
        self.tableData = win.tableData
