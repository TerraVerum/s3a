from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np
from qtextras import ParameterContainer, RunOptions, fns

from .base import ParameterEditorPlugin
from ..constants import PRJ_CONSTS as CNST, REQD_TBL_FIELDS as RTF
from ..structures import ComplexXYVertices, XYVertices
from ..views.rois import PointROI

if TYPE_CHECKING:
    from ..models.s3abase import S3ABase
    from ..shared import SharedAppSettings
    from ..tabledata import TableData


class MainImagePlugin(ParameterEditorPlugin):
    createDock = False
    createProcessMenu = False
    tableData: TableData

    def __initSharedSettings__(self, shared: SharedAppSettings = None, **kwargs):
        self.props = ParameterContainer()
        shared.generalProperties.registerParameter(
            CNST.PROP_MIN_COMP_SZ, container=self.props
        )
        shared.generalProperties.registerFunction(
            PointROI.updateRadius,
            radius=dict(title="Point ROI Radius"),
            runOptions=RunOptions.ON_CHANGED,
            nest=False,
        )
        shared.colorScheme.registerFunction(
            self.window.mainImage.updateGridScheme,
            runOptions=RunOptions.ON_CHANGED,
            name="grid_scheme",
        )

        self._hookupRegionPlotProperties(self.window)

        super().__initSharedSettings__(shared=shared, **kwargs)

    def attachToWindow(self, window: S3ABase):
        self.tableData = window.tableData
        self._hookupCopier(window)
        self._hookupDrawActions(window)
        self._hookupSelectionTools(window)

        collection = window.mainImage.addTools(self)
        # "self" doesn't have a gui component, so if shortcuts aren't reassigned to
        # visible objects, they will never be activatable
        for options, button in collection.optionsButtonMap.items():
            shortcut = self.nameShortcutMap[options.name].key()
            self.registerObjectShortcut(button, shortcut, options.name, force=True)
        super().attachToWindow(window)

    def _hookupDrawActions(self, window):
        disp = window.componentController

        def actHandler(verts, param):
            activeEdits = (
                len(self.window.verticesPlugin.region.regionData["Vertices"]) > 0
            )
            if (
                param in [CNST.DRAW_ACT_REM, CNST.DRAW_ACT_ADD]
                and not activeEdits
                and self.window.componentController.selectionIntersectsRegion(verts)
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
                self.window.mainImage.shapeCollection.shapeParameter
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
        window.mainImage.registerDrawAction(acts, actHandler)
        # Create checks an edge case for selection, so no need to add to above acts
        window.mainImage.registerDrawAction(CNST.DRAW_ACT_CREATE, self.createComponent)

    def _hookupCopier(self, window):
        mainImage = window.mainImage

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

        self.registerFunction(startMove, runActionTemplate=CNST.TOOL_MOVE_REGIONS)
        self.registerFunction(startCopy, runActionTemplate=CNST.TOOL_COPY_REGIONS)
        copier.sigMoveStopped.connect(window.componentManager.updateFocusedComponent)

    def _hookupSelectionTools(self, window):
        disp = window.componentController
        self.registerFunction(
            disp.mergeSelectedComponents,
            runActionTemplate=CNST.TOOL_MERGE_COMPS,
            ignores=["keepId"],
        )
        self.registerFunction(
            disp.splitSelectedComponents, runActionTemplate=CNST.TOOL_SPLIT_COMPS
        )
        self.registerFunction(
            disp.removeSelectedComponentOverlap, runActionTemplate=CNST.TOOL_REM_OVERLAP
        )

    def _hookupRegionPlotProperties(self, window):
        scheme = window.sharedSettings.colorScheme
        general = window.sharedSettings.generalProperties

        regionPlot = self.window.componentController.regionPlot
        availableParams = list(regionPlot.props.parameters.values())
        colorsSig = inspect.signature(regionPlot.updateColors).parameters
        regionNamePath = (scheme.defaultParent, "Region Features")

        colorPropsParameter = fns.getParameterChild(
            scheme.rootParameter, *regionNamePath
        )
        generalPropsParameter = fns.getParameterChild(
            general.rootParameter, *regionNamePath
        )

        colorPropsParameter.addChildren(
            [p for p in availableParams if p.name() in colorsSig]
        )
        generalPropsParameter.addChildren(
            [p for p in availableParams if p.name() not in colorsSig]
        )

    @property
    def image(self):
        return self.window.mainImage.image

    def createComponent(self, roiVertices: XYVertices):
        verts = np.clip(roiVertices.astype(int), 0, self.image.shape[:2][::-1])

        if cv.contourArea(verts) < self.props[CNST.PROP_MIN_COMP_SZ]:
            # Use as selection instead of creation
            self.window.componentController.reflectSelectionBoundsMade(roiVertices[[0]])
            return

        verts = ComplexXYVertices([verts]).simplify(
            self.window.verticesPlugin.props[CNST.PROP_REG_APPROX_EPS]
        )
        newComps = self.tableData.makeComponentDf()
        newComps[RTF.VERTICES] = [verts]
        self.window.addAndFocusComponents(newComps)
