from __future__ import annotations

import cv2 as cv
import numpy as np
import pandas as pd
from utilitys import ParamEditorPlugin, ParamContainer, fns

from ..constants import PRJ_CONSTS as CNST, REQD_TBL_FIELDS as RTF
from ..models import s3abase
from ..shared import SharedAppSettings
from ..structures import XYVertices, ComplexXYVertices


class MainImagePlugin(ParamEditorPlugin):
    name = __groupingName__ = "Application"
    _makeMenuShortcuts = False

    def __initEditorParams__(self, shared: SharedAppSettings, **kwargs):
        self.props = ParamContainer()
        shared.generalProps.registerProp(CNST.PROP_MIN_COMP_SZ, container=self.props)
        self.tableData = shared.tableData
        super().__initEditorParams__(shared=shared, **kwargs)

    def attachWinRef(self, win: s3abase.S3ABase):
        self._hookupCopier(win)
        self._hookupDrawActions(win)
        self._hookupSelectionTools(win)

        win.mainImg.addTools(self.toolsEditor)
        # No need for a dropdown menu
        self.dock = None
        super().attachWinRef(win)

    def _hookupDrawActions(self, win):
        disp = win.compDisplay

        def actHandler(verts, param):
            # When editing, only want to select if nothing is already started
            if (
                param not in [CNST.DRAW_ACT_REM, CNST.DRAW_ACT_ADD]
                or len(self.win.vertsPlg.region.regionData) == 0
            ):
                # Special case: Selection with point shape should be a point
                if (
                    self.win.mainImg.shapeCollection.curShapeParam
                    == CNST.DRAW_SHAPE_POINT
                ):
                    verts = verts.mean(0, keepdims=True)
                # Make sure to check vertices plugin regions since they suppress disp's regions for focused ids
                # Must be done first, since a no-find in disp's regions will deselect them
                with fns.makeDummySignal(win.compTbl, "sigSelectionChanged"):
                    # Second call should handle the true selection signal
                    disp.reflectSelectionBoundsMade(verts, self.win.vertsPlg.region)
                    disp.reflectSelectionBoundsMade(verts, clearExisting=False)

                nonUniqueIds = win.compTbl.idsRowsColsFromSelection(
                    excludeNoEditCols=False, warnNoneSelection=False
                )[:, 0]
                selection = pd.unique(nonUniqueIds)
                win.compTbl.sigSelectionChanged.emit(selection)

        acts = [
            CNST.DRAW_ACT_ADD,
            CNST.DRAW_ACT_REM,
            CNST.DRAW_ACT_SELECT,
            CNST.DRAW_ACT_PAN,
        ]
        win.mainImg.registerDrawAction(acts, actHandler)
        win.mainImg.registerDrawAction(CNST.DRAW_ACT_CREATE, self.createComponent)

    def _hookupCopier(self, win):
        mainImg = win.mainImg

        copier = mainImg.regionMover

        def startCopy():
            """
            Copies the selected components. They can be pasted by <b>double-clicking</b>
            on the destination location. When done copying, Click the *Clear ROI* tool change
            the current draw action.
            """
            copier.inCopyMode = True
            copier.sigMoveStarted.emit()

        def startMove():
            """
            Moves the selected components. They can be pasted by <b>double-clicking</b>
            on the destination location.
            """
            copier.inCopyMode = False
            copier.sigMoveStarted.emit()

        self.registerFunc(startMove, btnOpts=CNST.TOOL_MOVE_REGIONS)
        self.registerFunc(startCopy, btnOpts=CNST.TOOL_COPY_REGIONS)
        copier.sigMoveStopped.connect(win.mainImg.updateFocusedComp)

    def _hookupSelectionTools(self, win):
        disp = win.compDisplay
        self.registerFunc(
            disp.mergeSelectedComps,
            btnOpts=CNST.TOOL_MERGE_COMPS,
            ignoreKeys=["keepId"],
        )
        self.registerFunc(disp.splitSelectedComps, btnOpts=CNST.TOOL_SPLIT_COMPS)
        self.registerFunc(disp.removeSelectedCompOverlap, btnOpts=CNST.TOOL_REM_OVERLAP)

    @property
    def image(self):
        return self.win.mainImg.image

    def createComponent(self, roiVerts: XYVertices):
        verts = np.clip(roiVerts.astype(int), 0, self.image.shape[:2][::-1])

        if cv.contourArea(verts) < self.props[CNST.PROP_MIN_COMP_SZ]:
            # Use as selection instead of creation
            self.win.compDisplay.reflectSelectionBoundsMade(roiVerts[[0]])
            return

        # noinspection PyTypeChecker
        verts = ComplexXYVertices([verts])
        newComps = self.tableData.makeCompDf()
        newComps[RTF.VERTICES] = [verts]
        self.win.addAndFocusComps(newComps)
