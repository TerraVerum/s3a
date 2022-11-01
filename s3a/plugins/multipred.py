from __future__ import annotations

import numpy as np
import pandas as pd
from utilitys import ProcessIO

from .base import ProcessorPlugin
from ..constants import (
    CONFIG_DIR,
    MULTI_PREDICTIONS_DIR,
    PRJ_CONSTS as CNST,
    PRJ_ENUMS,
    REQD_TBL_FIELDS as RTF,
)
from ..models.tablemodel import ComponentManager
from ..parameditors.algcollection import AlgorithmCollection
from ..processing.algorithms import multipred
from ..shared import SharedAppSettings
from ..structures import ComplexXYVertices


class MultiPredictionsPlugin(ProcessorPlugin):
    name = "Multi-Predictions"

    manager: ComponentManager

    def __initEditorParams__(self, shared: SharedAppSettings):
        super().__initEditorParams__()

        self.multiPredictionCollection = AlgorithmCollection(
            saveDir=MULTI_PREDICTIONS_DIR, template=CONFIG_DIR / "multipred.yml"
        )
        self.procEditor = self.multiPredictionCollection.createProcessorEditor(
            type(self), self.name + " Processor"
        )
        self.dock.addEditors([self.procEditor])
        self.pluginRunnerProc = self.multiPredictionCollection.parseProcessName(
            "Run Plugins", topFirst=False
        )

    def __init__(self):
        super().__init__()
        self.registerFunc(self.lastRunAnalytics)

    def attachWinRef(self, win):
        super().attachWinRef(win)
        self.manager = win.componentManager
        self.mainImage = win.mainImage
        win.mainImage.toolsEditor.registerFunc(
            self.makePrediction,
            btnOpts=CNST.TOOL_MULT_PRED,
            ignoreKeys=["components"],
        )
        self.updateRunnerLimits()
        win.sigPluginAdded.connect(self.updateRunnerLimits)

    def updateRunnerLimits(self):
        self.pluginRunnerProc.input.parameters["plugins"].setLimits(
            [
                p.name
                for p in self.win.classPluginMap.values()
                if hasattr(p, "runOnComponent")
            ]
        )

    def makePrediction(self, components: pd.DataFrame = None, **runKwargs):
        if self.win.mainImage.image is None:
            return
        if components is None:
            components = self.win.componentDf
        # It is possible for a previously selected id to be deleted before a redraw
        # occurs, in which case the selected id won't correspond to a valid index.
        # Resolve using intersection with all components
        selectedIds = np.intersect1d(
            self.win.componentController.selectedIds,
            self.win.componentManager.compDf.index,
        )
        vbRange = np.array(self.mainImage.getViewBox().viewRange()).T
        image = self.win.mainImage.image
        result = self.currentProcessor.activate(
            components=components,
            fullComponents=components,
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
            "addType", PRJ_ENUMS.COMPONENT_ADD_AS_MERGE
        )
        return self.manager.addComponents(compsToAdd, addType)

    def applyPluginRunners(self, components: pd.DataFrame, plugins: list[str]):
        if not plugins:
            return components
        # Don't run on empty components, since these only exist to indicate deletion
        # But keep them recorded to handle them later on
        emptyComps = []
        for pluginName in plugins:
            for plugin in self.win.classPluginMap.values():
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
