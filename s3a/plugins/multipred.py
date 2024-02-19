from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd
from qtextras import bindInteractorOptions as bind, fns

from .base import ProcessorPlugin
from ..constants import (
    CONFIG_DIR,
    MENU_OPTS_DIR,
    PRJ_CONSTS as CNST,
    PRJ_ENUMS,
    REQD_TBL_FIELDS as RTF,
)
from ..generalutils import concatAllowEmpty, coerceDfTypes
from ..models.tablemodel import ComponentManager
from ..parameditors.algcollection import AlgorithmCollection
from ..processing.algorithms import multipred


class MultiPredictionsPlugin(ProcessorPlugin):
    manager: ComponentManager

    def __init__(self):
        clctn = AlgorithmCollection(
            name="Multi Predictions",
            template=CONFIG_DIR / f"multipred.yml",
            directory=MENU_OPTS_DIR,
        )
        super().__init__(clctn)

        self.registerFunction(
            self.lastRunAnalytics, runActionTemplate=CNST.TOOL_PROC_ANALYTICS
        )
        runnerName = self.algorithmCollection.addFunction(
            self.applyPluginRunners,
            name=fns.nameFormatter("run_plugins"),
        )
        self.pluginRunnerFunction = self.algorithmCollection.parseProcessName(
            runnerName
        )
        self.pluginRunnerFunction.cachable = False

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
        if "plugins" in self.pluginRunnerFunction.input.parameters:
            self.pluginRunnerFunction.input.parameters["plugins"].setOpts(
                limits=newLimits
            )
        else:
            # Change bound limits for this function
            boundAttrs = type(self).applyPluginRunners.__interactor_bind_options__
            boundAttrs["plugins"]["limits"] = newLimits

    def makePrediction(self, components: pd.DataFrame = None, **runKwargs):
        if self.window.mainImage.image is None:
            return
        if components is None:
            components = self.window.componentManager.compDf
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

        addType = runKwargs.get("addType") or result.get(
            "addType", PRJ_ENUMS.COMPONENT_ADD_AS_MERGE
        )
        compsToAdd = coerceDfTypes(compsToAdd, components.columns)
        return self.manager.addComponents(compsToAdd, addType)

    @bind(
        plugins=dict(type="checklist", value=[RTF.VERTICES.name], limits=[]),
        verticesAs=dict(
            type="list",
            limits=["foreground", "background", "none"],
            tip="Whether to treat component vertices as foreground information, "
            "background information, or to disregard them during plugin processing",
            title=fns.nameFormatter("treat_vertices_as"),
        ),
    )
    def applyPluginRunners(
        self,
        components: pd.DataFrame,
        plugins: list[str],
        verticesAs: t.Literal["foreground", "background", "none"] = "background",
    ):
        if not plugins or not len(components):
            return components
        # Don't run on empty components, since these only exist to indicate deletion
        # But keep them recorded to handle them later on
        # Add a sentinel dataframe to avoid error from empty components
        emptyComps = [components.iloc[0:0]]
        usePlugins = [
            p
            for p in self.window.classPluginMap.values()
            if p.name in plugins and hasattr(p, "runOnComponent")
        ]
        for plugin in usePlugins:
            dispatched = multipred.ProcessDispatcher(plugin.runOnComponent)
            components, empty = self._handleDispatchedComponents(
                components, dispatched, verticesAs=verticesAs
            )
            if len(empty):
                emptyComps.append(empty)
        return dict(components=concatAllowEmpty([components, *emptyComps]))

    def _handleDispatchedComponents(
        self,
        components: pd.DataFrame,
        dispatched: multipred.ProcessDispatcher,
        **kwargs,
    ):
        if not len(components):
            return components
        result = dispatched(components=components, **kwargs)
        newComponents = result["components"]
        emptyIdxs = (newComponents[RTF.VERTICES].apply(len) < 1).to_numpy(bool)
        return newComponents[~emptyIdxs], newComponents[emptyIdxs]

    def lastRunAnalytics(self):
        raise NotImplementedError
