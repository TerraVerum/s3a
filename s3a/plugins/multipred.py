from __future__ import annotations

import numpy as np
import pandas as pd
from utilitys import ProcessIO

from .base import ProcessorPlugin
from ..constants import PRJ_CONSTS as CNST, PRJ_ENUMS, REQD_TBL_FIELDS as RTF
from ..models.tablemodel import ComponentMgr
from ..processing.algorithms import multipred
from ..shared import SharedAppSettings
from ..structures import ComplexXYVertices


class MultiPredictionsPlugin(ProcessorPlugin):
    name = "Multi-Predictions"

    mgr: ComponentMgr

    def __initEditorParams__(self, shared: SharedAppSettings):
        super().__initEditorParams__()
        self.procEditor = shared.multiPredClctn.createProcessorEditor(
            type(self), self.name + " Processor"
        )
        self.dock.addEditors([self.procEditor])
        self.pluginRunnerProc = shared.multiPredClctn.parseProcName(
            "Run Plugins", topFirst=False
        )

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
        self.updateRunnerLimits()
        win.sigPluginAdded.connect(self.updateRunnerLimits)

    def updateRunnerLimits(self):
        self.pluginRunnerProc.input.params["plugins"].setLimits(
            [
                p.name
                for p in self.win.clsToPluginMapping.values()
                if hasattr(p, "runOnComponent")
            ]
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
        result = self.curProcessor.run(
            components=comps,
            fullComponents=comps,
            fullImage=image,
            image=image,
            viewbox=vbRange,
            selectedIds=selectedIds,
            **runKwargs,
        )
        if not isinstance(result, ProcessIO):
            result = ProcessIO(components=result)
        compsToAdd = result["components"]
        if not len(compsToAdd):
            return
        if "plugins" in result:
            compsToAdd = self.applyPluginRunners(compsToAdd, result["plugins"])

        addType = runKwargs.get("addType") or result.get(
            "addType", PRJ_ENUMS.COMP_ADD_AS_MERGE
        )
        return self.mgr.addComps(compsToAdd, addType)

    def applyPluginRunners(self, components: pd.DataFrame, plugins: list[str]):
        if not plugins:
            return components
        # Don't run on empty components, since these only exist to indicate deletion
        # But keep them recorded to handle them later on
        emptyComps = []
        for pluginName in plugins:
            for plugin in self.win.clsToPluginMapping.values():
                if plugin.name == pluginName:
                    dispatched = multipred.ProcessDispatcher(plugin.runOnComponent)
                    emptyIdxs = (
                        components[RTF.VERTICES]
                        .apply(ComplexXYVertices.isEmpty)
                        .to_numpy(bool)
                    )
                    emptyComps.append(components[emptyIdxs])
                    components = dispatched(components=components[~emptyIdxs])[
                        "components"
                    ]
                    continue
        return pd.concat([components, *emptyComps])

    def lastRunAnalytics(self):
        raise NotImplementedError
