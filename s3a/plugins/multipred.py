from __future__ import annotations

import numpy as np
import pandas as pd
from utilitys import ProcessIO

from .base import ProcessorPlugin
from ..constants import PRJ_CONSTS as CNST, PRJ_ENUMS
from ..models.tablemodel import ComponentMgr
from ..shared import SharedAppSettings


class MultiPredictionsPlugin(ProcessorPlugin):
    name = "Multi-Predictions"

    mgr: ComponentMgr

    def __initEditorParams__(self, shared: SharedAppSettings):
        super().__initEditorParams__()
        self.procEditor = shared.multiPredClctn.createProcessorEditor(
            type(self), self.name + " Processor"
        )
        self.dock.addEditors([self.procEditor])

    def __init__(self):
        super().__init__()
        self.registerFunc(self.lastRunAnalytics)

    def attachWinRef(self, win):
        super().attachWinRef(win)
        self.mgr = win.compMgr
        self.mainImg = win.mainImg
        win.mainImg.toolsEditor.registerFunc(
            self.makePrediction,
            btnOpts=CNST.TOOL_MULT_PRED,
            ignoreKeys=["comps"],
        )

    def makePrediction(self, comps: pd.DataFrame = None, **runKwargs):
        if self.win.mainImg.image is None:
            return
        if comps is None:
            comps = self.win.exportableDf
        # It is possible for a previously selected id to be deleted before a redraw occurs, in which case the
        # selected id won't correspond to a valid index. Resolve using intersection with all components
        selectedIds = np.intersect1d(
            self.win.compDisplay.selectedIds, self.win.compMgr.compDf.index
        )
        vbRange = np.array(self.mainImg.getViewBox().viewRange()).T
        image = self.win.mainImg.image
        newComps = self.curProcessor.run(
            components=comps,
            fullComponents=comps,
            fullImage=image,
            image=image,
            viewbox=vbRange,
            selectedIds=selectedIds,
            **runKwargs,
        )
        if not isinstance(newComps, ProcessIO):
            newComps = ProcessIO(components=newComps)
        compsToAdd = newComps["components"]
        if not len(compsToAdd):
            return
        addType = runKwargs.get("addType") or newComps.get(
            "addType", PRJ_ENUMS.COMP_ADD_AS_NEW
        )
        return self.mgr.addComps(compsToAdd, addType)

    def lastRunAnalytics(self):
        raise NotImplementedError
