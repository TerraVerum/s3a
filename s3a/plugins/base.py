from __future__ import annotations

import contextlib
import typing as t
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
from pyqtgraph.parametertree import InteractiveFunction
from pyqtgraph.Qt import QtWidgets
from qtextras import ParameterEditor, fns

from ..constants import MENU_OPTS_DIR, PRJ_CONSTS
from ..graphicsutils import reorderMenuActions
from ..parameditors import algcollection
from ..parameditors.algcollection import AlgorithmCollection, AlgorithmEditor
from ..processing import PipelineParameter

if t.TYPE_CHECKING:
    from ..models.s3abase import S3ABase
    from ..models.tablemodel import ComponentManager
    from ..shared import SharedAppSettings
    from ..views.s3agui import S3A

_UNSET_NAME = object()


class ParameterEditorPlugin(ParameterEditor):
    window: S3A = None
    directoryParent: str | None = MENU_OPTS_DIR
    menuTitle: str = None

    createDock = False
    createProcessMenu = True

    dock: QtWidgets.QDockWidget | None = None
    menu: QtWidgets.QMenu | None = None

    def __initSharedSettings__(self, shared: SharedAppSettings = None, **kwargs):
        """
        Overload this method to add parameters to the editor. This method is called
        when the plugin is attached to the window.
        """
        pass

    def __init__(self, *, name: str = None, directory: str = None, **kwargs):
        defaultName = self.name or self.__class__.__name__.replace("Plugin", "")
        if name is None:
            name = fns.nameFormatter(defaultName)
        if directory is None and self.directoryParent is not None:
            directory = Path(self.directoryParent) / name.lower()
        super().__init__(name=name, directory=directory)
        self.registeredEditors: list[ParameterEditor] = []
        if self.createDock:
            self.registeredEditors.append(self)

    def attachToWindow(self, window: S3A | S3ABase):
        self.window = window
        self.menuTitle = self._resolveMenuTitle(self.name)
        if self.createDock:
            self.dock, self.menu = self.createWindowDock(
                window, self.menuTitle, createProcessMenu=self.createProcessMenu
            )
        elif self.createProcessMenu:
            self.menu = self.createActionsFromFunctions(QtWidgets.QMenu(self.menuTitle))

        if self.menu:
            window.menuBar().addMenu(self.menu)

        # Temporarily set the default name path for where shared parameters get registered
        with self.sharedDefaultParentContext():
            self.__initSharedSettings__(shared=window.sharedSettings)

    def registerPopoutFunctions(
        self,
        functionList: t.Sequence[t.Callable],
        nameList: t.Sequence[str] = None,
        groupName: str = None,
        runActionTemplate: dict = None,
        menu: QtWidgets.QMenu = None,
    ):
        if groupName is None and runActionTemplate is None:
            raise ValueError("Must provide either group name or action options")
        if groupName is None:
            groupName = runActionTemplate["name"]

        groupParameter = fns.getParameterChild(
            self.rootParameter, groupName, groupOpts=dict(type="_actiongroup")
        )
        function = InteractiveFunction(lambda: fns.parameterDialog(groupParameter))
        self.registerFunction(
            function, name=groupName, runActionTemplate=runActionTemplate, runOptions=[]
        )

        if nameList is None:
            nameList = [None] * len(functionList)

        # Don't allow registration listeners to find children; they are a subset
        # of the already-registed parent
        with fns.makeDummySignal(self, "sigFunctionRegistered"):
            for title, func in zip(nameList, functionList):
                self.registerFunction(func, name=title, parent=groupParameter)

        if menu is not None:
            act = menu.addAction(groupName, function)
            if runActionTemplate is not None and runActionTemplate.get("shortcut"):
                act.setShortcut(runActionTemplate["shortcut"])
        return function

    def _resolveMenuTitle(self, name: str = None, ensureShortcut=True):
        name = self.menuTitle or name
        if ensureShortcut and "&" not in name:
            name = f"&{name}"
        return name

    @contextlib.contextmanager
    def sharedDefaultParentContext(self, name: str = None):
        if name is None:
            name = self.name
        attrs = self.window.sharedSettings
        with ExitStack() as stack:
            for editor in [attrs.colorScheme, attrs.generalProperties]:
                stack.enter_context(fns.overrideAttr(editor, "defaultParent", name))
            yield

    def createDockWithoutFunctionMenu(self, editor: ParameterEditor, reorder=True):
        """
        Convenience function to call ``createWindowDock`` with
        ``createProcessMenu=False`` and optionally ensure the "show" action is first in
        self's menu
        """
        if self.menu is None:
            raise ValueError(
                "`self.menu` must exist before creating dock. Perhaps you made this "
                "function call before `super().attachToWindow(window)`?"
            )
        editor.createWindowDock(
            self.window, editor.name, createProcessMenu=False, menu=self.menu
        )
        if reorder:
            reorderMenuActions(self.menu, oldIndex=-1, newIndex=0)


class ProcessorPlugin(ParameterEditorPlugin):
    processEditor: algcollection.AlgorithmEditor = None
    """
    Most table field plugins will use some sort of processor to infer field data.
    This property holds spawned collections. See :class:`VerticesPlugin` for
    an example.
    """

    def __init__(
        self,
        algorithmCollection: AlgorithmCollection = None,
        processorSuffix: str = ".alg",
        **kwargs,
    ):
        if algorithmCollection is None:
            algorithmCollection = AlgorithmCollection()
        super().__init__(**kwargs)
        self.algorithmCollection = algorithmCollection

        procName = f"{self.name} Processor"
        if algorithmCollection.directory:
            procDir = Path(algorithmCollection.directory) / procName.lower()
            procDir.mkdir(exist_ok=True)
        else:
            procDir = None

        self.processEditor = AlgorithmEditor(
            self.algorithmCollection,
            name=procName,
            directory=procDir,
            suffix=processorSuffix,
        )

    @property
    def currentProcessor(self) -> PipelineParameter:
        return self.processEditor.currentProcessor

    @currentProcessor.setter
    def currentProcessor(self, newProcessor: str | PipelineParameter):
        self.processEditor.changeActiveProcessor(newProcessor)

    def attachToWindow(self, window: S3A | S3ABase):
        super().attachToWindow(window)
        self.createDockWithoutFunctionMenu(self.processEditor)
        self.registeredEditors.append(self.processEditor)


class TableFieldPlugin(ProcessorPlugin):
    mainImage = None
    """
    Holds a reference to the focused image and set when the s3a reference is set. 
    This is useful for most table field plugins, since mainImage will hold a reference to 
    the component series that is modified by the plugins.
    """
    componentManager: ComponentManager = None
    """
    Holds a reference to the focused image and set when the s3a reference is set.
    Offers a convenient way to access component data.
    """

    _active = False

    _makeMenuShortcuts = False

    def attachToWindow(self, window: S3A):
        super().attachToWindow(window)
        self.mainImage = window.mainImage
        self.componentManager = window.componentManager
        window.sigRegionAccepted.connect(self.acceptChanges)
        self.componentManager.sigUpdatedFocusedComponent.connect(
            self.updateFocusedComponent
        )
        self.active = True
        self.registerFunction(
            self.processorAnalytics, runActionTemplate=PRJ_CONSTS.TOOL_PROC_ANALYTICS
        )

    def processorAnalytics(self):
        proc = self.currentProcessor
        if hasattr(proc, "stageSummaryGui"):
            proc.stageSummaryGui()
        else:
            raise TypeError(
                f"Processor type {type(proc)} does not implement summary analytics."
            )

    def updateFocusedComponent(self, component: pd.Series = None):
        """
        This function is called when a new component is created or the focused image is
        updated from the main view. See :meth:`ComponentManager.updateFocusedComponent`
        for parameters.
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
