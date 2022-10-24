from __future__ import annotations

from typing import Union

import pandas as pd
from utilitys import NestedProcWrapper, ParamEditorPlugin

from ..constants import PRJ_CONSTS
from ..parameditors import algcollection


class ProcessorPlugin(ParamEditorPlugin):
    procEditor: algcollection.AlgParamEditor = None
    """
    Most table field plugins will use some sort of processor to infer field data.
    This property holds spawned collections. See :class:`XYVerticesPlugin` for
    an example.
    """

    @property
    def currentProcessor(self):
        return self.procEditor.currentProcessor

    @currentProcessor.setter
    def currentProcessor(self, newProcessor: Union[str, NestedProcWrapper]):
        self.procEditor.changeActiveProcessor(newProcessor)


class TableFieldPlugin(ProcessorPlugin):
    mainImage = None
    """
    Holds a reference to the focused image and set when the s3a reference is set. 
    This is useful for most table field plugins, since mainImage will hold a reference to 
    the component series that is modified by the plugins.
    """

    _active = False

    _makeMenuShortcuts = False

    # @property
    # def parentMenu(self):
    #   return self.win.tableFieldToolbar

    def attachWinRef(self, win):
        super().attachWinRef(win)
        self.mainImage = win.mainImage
        self.componentManager = win.componentManager
        win.sigRegionAccepted.connect(self.acceptChanges)
        self.componentManager.sigUpdatedFocusedComponent.connect(
            self.updateFocusedComponent
        )
        self.active = True
        self.registerFunc(
            self.processorAnalytics, btnOpts=PRJ_CONSTS.TOOL_PROC_ANALYTICS
        )

    def processorAnalytics(self):
        proc = self.currentProcessor
        try:
            proc.processor.stageSummaryGui()
        except NotImplementedError:
            raise TypeError(
                f"Processor type {type(proc)} does not implement summary analytics."
            )

    def updateFocusedComponent(self, component: pd.Series = None):
        """
        This function is called when a new component is created or the focused image is
        updated from the main view. See :meth:`ComponentManager.updateFocusedComponent` for
        parameters.
        """
        pass

    def acceptChanges(self):
        """
        This must be overloaded by each plugin so the set component data is properly
        stored in the focused component. Essentially, any changes made by this plugin
        are saved after a call to this method.
        """
        raise NotImplementedError

    @property
    def active(self):
        """Whether this plugin is currently in use by the focused image."""
        return self._active

    @active.setter
    def active(self, newActive: bool):
        if newActive == self._active:
            return
        if newActive:
            self._onActivate()
        else:
            self._onDeactivate()
        self._active = newActive

    def _onActivate(self):
        """Overloaded by plugin classes to set up the plugin for use"""

    def _onDeactivate(self):
        """
        Overloaded by plugin classes to tear down when the plugin is no longer in
        use
        """
