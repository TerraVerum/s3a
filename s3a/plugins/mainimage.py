from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np
from utilitys import ParamContainer, ParamEditorPlugin, RunOpts, fns

from pyqtgraph.parametertree import InteractiveFunction
from ..constants import PRJ_CONSTS as CNST, REQD_TBL_FIELDS as RTF
from ..generalutils import ClassInteractiveFunction
from ..structures import ComplexXYVertices, XYVertices
from ..views.regions import MultiRegionPlot
from ..views.rois import PointROI

if TYPE_CHECKING:
    from ..models.s3abase import S3ABase
    from ..shared import SharedAppSettings
    from ..tabledata import TableData


class MainImagePlugin(ParamEditorPlugin):
    name = __groupingName__ = "Application"
    _makeMenuShortcuts = False
    tableData: TableData
    win: S3ABase

    def __initEditorParams__(self, shared: SharedAppSettings, **kwargs):
        self.props = ParamContainer()
        shared.generalProperties.registerProp(
            CNST.PROP_MIN_COMP_SZ, container=self.props
        )
        shared.colorScheme.registerFunc(
            self.win.mainImage.updateGridScheme, runOpts=RunOpts.ON_CHANGED
        )
        shared.colorScheme.registerFunc(
            PointROI.updateRadius,
            name="Point ROI Features",
            runOpts=RunOpts.ON_CHANGED,
        )

        if not isinstance(MultiRegionPlot.updateColors, InteractiveFunction):
            MultiRegionPlot.updateColors = ClassInteractiveFunction(
                MultiRegionPlot.updateColors
            )
            MultiRegionPlot.setBoundaryOnly = ClassInteractiveFunction(
                MultiRegionPlot.setBoundaryOnly
            )
            shared.colorScheme.registerFunc(
                MultiRegionPlot.updateColors,
                runOpts=RunOpts.ON_CHANGED,
                nest=False,
                container=MultiRegionPlot.props,
                labelColormap=dict(limits=fns.listAllPgColormaps() + ["None"]),
            )
            shared.generalProperties.registerFunc(
                MultiRegionPlot.setBoundaryOnly, runOpts=RunOpts.ON_CHANGED, nest=False
            )

        super().__initEditorParams__(shared=shared, **kwargs)
        self._cachedRegionIntersection = False

    def attachWinRef(self, win: S3ABase):
        self.tableData = win.tableData
        self._hookupCopier(win)
        self._hookupDrawActions(win)
        self._hookupSelectionTools(win)

        win.mainImage.addTools(self.toolsEditor)
        # No need for a dropdown menu
        self.dock = None
        super().attachWinRef(win)

    def _hookupDrawActions(self, win):
        disp = win.componentController

        def actHandler(verts, param):
            activeEdits = len(self.win.verticesPlugin.region.regionData["Vertices"]) > 0
            if (
                param in [CNST.DRAW_ACT_REM, CNST.DRAW_ACT_ADD]
                and not activeEdits
                and self.win.componentController.selectionIntersectsRegion(verts)
            ):
                warnings.warn(
                    "Made a selection on top of an existing component. It is ambiguous"
                    " whether the existing component should be selected or a new"
                    " component should be created on top. Use either 'Select' or"
                    " 'Create' action first",
                    UserWarning,
                    stacklevel=2,
                )
                return
            elif param in [CNST.DRAW_ACT_REM, CNST.DRAW_ACT_ADD] and activeEdits:
                # Don't make selection if edits are already in progress
                return
            # Special case: Selection with point shape should be a point
            if (
                self.win.mainImage.shapeCollection.shapeParameter
                == CNST.DRAW_SHAPE_POINT
            ):
                verts = verts.mean(0, keepdims=True)
            disp.reflectSelectionBoundsMade(verts)

        acts = [
            CNST.DRAW_ACT_ADD,
            CNST.DRAW_ACT_REM,
            CNST.DRAW_ACT_SELECT,
            CNST.DRAW_ACT_PAN,
        ]
        win.mainImage.registerDrawAction(acts, actHandler)
        # Create checks an edge case for selection, so no need to add to above acts
        win.mainImage.registerDrawAction(CNST.DRAW_ACT_CREATE, self.createComponent)

    def _hookupCopier(self, win):
        mainImage = win.mainImage

        copier = mainImage.regionMover

        def startCopy():
            """
            Copies the selected components. They can be pasted by
            <b>double-clicking</b> on the destination location. When done copying,
            Click the *Clear ROI* tool change the current draw action.
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
        copier.sigMoveStopped.connect(win.componentManager.updateFocusedComponent)

    def _hookupSelectionTools(self, window):
        disp = window.componentController
        self.registerFunc(
            disp.mergeSelectedComponents,
            btnOpts=CNST.TOOL_MERGE_COMPS,
            ignoreKeys=["keepId"],
        )
        self.registerFunc(disp.splitSelectedComponents, btnOpts=CNST.TOOL_SPLIT_COMPS)
        self.registerFunc(
            disp.removeSelectedComponentOverlap, btnOpts=CNST.TOOL_REM_OVERLAP
        )

    @property
    def image(self):
        return self.win.mainImage.image

    def createComponent(self, roiVertices: XYVertices):
        verts = np.clip(roiVertices.astype(int), 0, self.image.shape[:2][::-1])

        if cv.contourArea(verts) < self.props[CNST.PROP_MIN_COMP_SZ]:
            # Use as selection instead of creation
            self.win.componentController.reflectSelectionBoundsMade(roiVertices[[0]])
            return

        verts = ComplexXYVertices([verts]).simplify(
            self.win.verticesPlugin.props[CNST.PROP_REG_APPROX_EPS]
        )
        newComps = self.tableData.makeComponentDf()
        newComps[RTF.VERTICES] = [verts]
        self.win.addAndFocusComponents(newComps)
