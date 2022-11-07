from __future__ import annotations

import numpy as np
import pandas as pd
from qtextras import nameFormatter

from .base import ProcessorPlugin
from ..constants import (
    CONFIG_DIR,
    MENU_OPTS_DIR,
    MULTI_PREDICTIONS_DIR,
    PRJ_CONSTS as CNST,
    PRJ_ENUMS,
    REQD_TBL_FIELDS as RTF,
)
from ..models.tablemodel import ComponentManager
from ..parameditors.algcollection import AlgorithmCollection, AlgorithmEditor
from ..processing.algorithms import multipred
from ..shared import SharedAppSettings
from ..structures import ComplexXYVertices


class MultiPredictionsPlugin(ProcessorPlugin):
    manager: ComponentManager

    def __initSharedSettings__(self, shared: SharedAppSettings = None, **kwargs):
        super().__initSharedSettings__(shared, **kwargs)

        _, self.processEditorMenu = self.processEditor.createWindowDock(
            self.window, self.processEditor.name
        )

    def __init__(self):
        super().__init__()
        self.registerFunction(self.lastRunAnalytics)

        self.multiPredictionCollection = AlgorithmCollection(
            name="Multi Predictions",
            directory=MULTI_PREDICTIONS_DIR,
            template=CONFIG_DIR / "multipred.yml",
        )
        self.multiPredictionCollection.addAllModuleProcesses(multipred)
        self.processEditor = AlgorithmEditor(
            self.multiPredictionCollection,
            name=self.name + " Processor",
            directory=MENU_OPTS_DIR / nameFormatter(type(self).__name__).lower(),
        )
        self.pluginRunnerProc = self.multiPredictionCollection.parseProcessName(
            "Run Plugins", topFirst=False
        )

    def attachToWindow(self, window):
        super().attachToWindow(window)
        beforeAction = self.menu.actions()[0] if len(self.menu.actions()) else None
        self.menu.insertMenu(beforeAction, self.processEditorMenu)
        self.manager = window.componentManager
        self.mainImage = window.mainImage
        window.mainImage.toolsEditor.registerFunction(
            self.makePrediction,
            runActionTemplate=CNST.TOOL_MULT_PRED,
            ignores=["components"],
        )
        self.updateRunnerLimits()
        window.sigPluginAdded.connect(self.updateRunnerLimits)

    def updateRunnerLimits(self):
        newLimits = [
            p.name
            for p in self.window.classPluginMap.values()
            if hasattr(p, "runOnComponent")
        ]
        self.pluginRunnerProc.input.parameters["plugins"].setLimits(newLimits)

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
