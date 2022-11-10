from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ProcessorPlugin
from ..constants import (
    CONFIG_DIR,
    MENU_OPTS_DIR,
    PRJ_CONSTS as CNST,
    PRJ_ENUMS,
    REQD_TBL_FIELDS as RTF,
)
from ..models.tablemodel import ComponentManager
from ..parameditors.algcollection import AlgorithmCollection
from ..processing.algorithms import multipred
from ..structures import ComplexXYVertices


class MultiPredictionsPlugin(ProcessorPlugin):
    manager: ComponentManager

    def __init__(self):
        clctn = AlgorithmCollection(
            name="Multi Predictions", template=CONFIG_DIR / f"multipred.yml"
        )
        clctn.addAllModuleProcesses(multipred)
        super().__init__(clctn, MENU_OPTS_DIR)

        self.registerFunction(
            self.lastRunAnalytics, runActionTemplate=CNST.TOOL_PROC_ANALYTICS
        )
        self.pluginRunnerFunction = self.algorithmCollection.parseProcessName(
            "Run Plugins", topFirst=False
        )

    def attachToWindow(self, window):
        super().attachToWindow(window)
        self.manager = window.componentManager
        self.mainImage = window.mainImage
        window.mainImage.toolsEditor.registerFunction(
            self.makePrediction, runActionTemplate=CNST.TOOL_MULT_PRED
        )
        self.updateRunnerLimits()
        window.sigPluginAdded.connect(self.updateRunnerLimits)

    def updateRunnerLimits(self):
        newLimits = [
            p.name
            for p in self.window.classPluginMap.values()
            if hasattr(p, "runOnComponent")
        ]
        self.pluginRunnerFunction.input.parameters["plugins"].setLimits(newLimits)

    def makePrediction(self, components: pd.DataFrame = None, **runKwargs):
        if self.window.mainImage.image is None:
            return
        if components is None:
            components = self.window.componentDf
        # It is possible for a previously selected id to be deleted before a redraw
        # occurs, in which case the selected id won't correspond to a valid index.
        # Resolve using intersection with all components
        selectedIds = np.intersect1d(
            self.window.componentController.selectedIds,
            self.window.componentManager.compDf.index,
        )
        vbRange = np.array(self.mainImage.getViewBox().viewRange()).T
        image = self.window.mainImage.image
        result = self.currentProcessor.activate(
            components=components,
            fullComponents=components,
            fullImage=image,
            image=image,
            viewbox=vbRange,
            selectedIds=selectedIds,
            **runKwargs,
        )
        if not isinstance(result, dict):
            result = dict(components=result)
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
            for plugin in self.window.classPluginMap.values():
                if plugin.name == pluginName and hasattr(plugin, "runOnComponent"):
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
