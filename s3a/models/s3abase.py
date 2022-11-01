import argparse
import inspect
import os.path
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Type, Union
from warnings import warn

import numpy as np
import pandas as pd
from pyqtgraph.Qt import QtCore, QtWidgets
from utilitys import (
    ActionStack,
    DeferredActionStackMixin as DASM,
    EditorPropsMixin,
    ParamContainer,
    ParamEditor,
    ParamEditorDockGrouping,
    ParamEditorPlugin,
    RunOpts,
    fns,
)

from .. import defaultIo
from ..constants import PRJ_CONSTS, PRJ_ENUMS, REQD_TBL_FIELDS
from ..controls.tableviewproxy import ComponentController, ComponentSorterFilter
from ..logger import getAppLogger
from ..models.tablemodel import ComponentManager
from ..parameditors.appstate import AppStateEditor
from ..plugins import EXTERNAL_PLUGINS, INTERNAL_PLUGINS, tablefield
from ..plugins.file import FilePlugin
from ..plugins.misc import RandomToolsPlugin
from ..shared import SharedAppSettings
from ..structures import FilePath, NChanImg
from ..tabledata import TableData
from ..views.imageareas import MainImage
from ..views.tableview import ComponentTableView

__all__ = ["S3ABase"]


class S3ABase(DASM, QtWidgets.QMainWindow):
    """
    Top-level widget for producing component bounding regions from an input image.
    """

    sigRegionAccepted = QtCore.Signal()
    sigPluginAdded = QtCore.Signal(object)  # Plugin object
    __groupingName__ = "S3A Window"

    scope = ExitStack()
    """
    Allows each instance of s3a to act like a "scope" for all objecats instantiated 
    within. Keeps multiple instances of separate S3A pieces from e.g. sharing the same 
    undo buffer. This is managed by __new__. 
    """

    sharedAttrs: SharedAppSettings
    """App-level properties that many moving pieces use"""

    def __new__(cls, *args, **kwargs):
        cls.scope.close()
        cls.scope, newAttrs = cls.createScope(cls.scope, returnAttributes=True)
        newAttrs: SharedAppSettings
        obj = super().__new__(cls, *args, **kwargs)
        obj.sharedAttrs = newAttrs
        return obj

    @staticmethod
    def createScope(scope: ExitStack = None, returnAttributes=False):
        if scope is None:
            scope = ExitStack()
        newAttrs = SharedAppSettings()
        scope.enter_context(EditorPropsMixin.setEditorPropertyOpts(shared=newAttrs))
        scope.enter_context(DASM.setStack(ActionStack()))
        if returnAttributes:
            return scope, newAttrs
        return scope

    def __init__(self, parent=None, **startupSettings):
        super().__init__(parent)

        self.props = ParamContainer()
        self.sharedAttrs.generalProperties.registerProps(
            [PRJ_CONSTS.EXP_ONLY_VISIBLE, PRJ_CONSTS.INCLUDE_FNAME_PATH],
            container=self.props,
        )

        self.classPluginMap: Dict[Type[ParamEditorPlugin], ParamEditorPlugin] = {}
        """
        Maintains a record of all plugins added to this window. Only up to one instance
        of each plugin class is expected.
        """

        self.docks: List[QtWidgets.QDockWidget] = []
        """List of docks from added plugins"""

        self.tableFieldToolbar = QtWidgets.QToolBar("Table Field Plugins")
        self.generalToolbar = QtWidgets.QToolBar("General")

        self.mainImage = MainImage(toolbar=self.generalToolbar)
        PRJ_CONSTS.TOOL_ACCEPT_FOC_REGION.opts["ownerObj"] = self.mainImage
        attrs = self.sharedAttrs
        self.mainImage.toolsEditor.registerFunc(
            self.acceptFocusedRegion, btnOpts=PRJ_CONSTS.TOOL_ACCEPT_FOC_REGION
        )
        _, param = attrs.generalProperties.registerFunc(
            self.actionStack.resizeStack,
            runOpts=RunOpts.ON_CHANGED,
            maxLength={
                **PRJ_CONSTS.PROP_UNDO_BUF_SZ.toPgDict(),
                "title": PRJ_CONSTS.PROP_UNDO_BUF_SZ.name,
            },
            returnParam=True,
            nest=False,
            container=self.props,
        )

        attrs.shortcuts.registerShortcut(
            PRJ_CONSTS.TOOL_CLEAR_ROI,
            self.mainImage.clearCurrentRoi,
            overrideOwnerObj=self.mainImage,
        )

        self.tableData = TableData(makeFilter=True)
        self.componentManager = ComponentManager(self.tableData)

        self.tableView = ComponentTableView()
        self.componentController = ComponentController(
            self.componentManager, self.mainImage, self.tableView
        )

        self.tableView.setSortingEnabled(True)
        self.tableView.setAlternatingRowColors(True)
        # Allow filtering/sorting
        self.sortFilterProxy = ComponentSorterFilter(self.componentManager)
        self.tableView.setModel(self.sortFilterProxy)

        self.hasUnsavedChanges = False
        self.sourceImagePath: Optional[Path] = None

        self.appStateEditor = AppStateEditor(
            self.sharedAttrs.quickLoader, self, name="App State Editor"
        )

        # -----
        # INTERFACE WITH QUICK LOADER / PLUGINS
        # -----
        toAdd = INTERNAL_PLUGINS() + EXTERNAL_PLUGINS()
        # Insert "settings" and "shortcuts" in a more logical location (after file + edit)
        toAdd = (
            toAdd[:2]
            + [self.sharedAttrs.settingsPlugin, self.sharedAttrs.shortcutsPlugin]
            + toAdd[2:]
        )
        for plg in toAdd:
            if inspect.isclass(plg):
                self.addPlugin(plg)
            else:
                self._addPluginObject(plg)

        # Create links for commonly used plugins
        # noinspection PyTypeChecker
        self.filePlugin: FilePlugin = self.classPluginMap[FilePlugin]
        self.tableData.sigConfigUpdated.connect(self.resetTableFields)

        # noinspection PyTypeChecker
        self.verticesPlugin: tablefield.VerticesPlugin = self.classPluginMap[
            tablefield.VerticesPlugin
        ]
        # noinspection PyTypeChecker
        self.miscPlugin: RandomToolsPlugin = self.classPluginMap[RandomToolsPlugin]
        self.componentIo = self.filePlugin.projectData.componentIo

        # Connect signals
        # -----
        # COMPONENT MANAGER
        # -----
        def handleUpdate(*_args):
            self.hasUnsavedChanges = True

        self.componentManager.sigComponentsChanged.connect(handleUpdate)

        # -----
        # MAIN IMAGE
        # -----
        def handleComponentsChanged(changedDict: dict):
            ser = self.componentManager.focusedComponent
            focusedId = ser[REQD_TBL_FIELDS.ID]
            if focusedId in changedDict["deleted"]:
                self.componentController.selectRowsById([])
                self.changeFocusedComponent()
            elif focusedId in changedDict["changed"]:
                self.changeFocusedComponent(self.componentManager.compDf.loc[focusedId])

        self.componentManager.sigComponentsChanged.connect(handleComponentsChanged)

        self.filePlugin.projectData.sigAnnotationsAdded.connect(
            self._maybeLoadActiveAnnotation
        )

        # -----
        # COMPONENT TABLE
        # -----
        self.componentController.sigComponentsSelected.connect(
            lambda newComps: self.changeFocusedComponent(newComps.index)
        )

        # -----
        # MISC
        # -----
        self.saveAllEditorDefaults()

    def resetTableFields(self):
        """
        When table fields change, the displayed columns must change and the view
        must be made aware. Ensure this occurs here
        """
        # Even if the field names are the same, e.g. classes may added or default
        # values could be changed. So, reset the cell editor delegates no matter what
        # Start by adding any potentially new plugins
        for plg in self.filePlugin.projectData.spawnedPlugins:
            self._addPluginObject(plg)
        self.tableView.setColDelegates()
        self.tableView.popup.reflectDelegateChange()
        # Make sure this is necessary, first
        for mgr in self.componentManager, self.tableView.popup.tbl.manager:
            if mgr.columnTitles == list([f.name for f in self.tableData.allFields]):
                # Fields haven't changed since last reset. Types could be different,
                # but nothing will break. So, the table doesn't have to be completely
                # reset
                return

            mgr.beginResetModel()
            mgr.removeComponents()
            mgr.resetFields()
            mgr.endResetModel()

    def saveAllEditorDefaults(self):
        for editor in self.docks:
            if isinstance(editor, ParamEditorDockGrouping):
                for subEditor in editor.editors:
                    subEditor.saveCurStateAsDefault()
            else:
                editor.saveCurStateAsDefault()

    @DASM.undoable("Accept Focused Region")
    def acceptFocusedRegion(self):
        """
        Applies the focused image vertices to the corresponding component in the
        table
        """
        # If the component was deleted
        mgr = self.componentManager
        focusedId = self.componentManager.focusedComponent[REQD_TBL_FIELDS.ID]
        exists = focusedId in mgr.compDf.index
        if not exists and focusedId != REQD_TBL_FIELDS.ID.value:
            # Could be a brand new component, allow in that case
            warn("Cannot accept region as this component was deleted.", UserWarning)
            return

        self.sigRegionAccepted.emit()

        ser = self.componentManager.focusedComponent
        if ser[REQD_TBL_FIELDS.VERTICES].isEmpty():
            # Component should be erased. Since new components will not match existing
            # IDs the same function will work regardless of whether this was new or
            # existing
            self.componentManager.removeComponents([ser[REQD_TBL_FIELDS.ID]])
            return

        if exists:
            undo = self._acceptFocusedExisting(ser)
        else:
            undo = self._acceptFocusedNew(ser)
        self.componentController.selectRowsById([])
        yield
        undo()

    def _acceptFocusedNew(self, focusedComponent: pd.Series):
        # New, make a brand new table entry
        compAsDf = fns.serAsFrame(focusedComponent)
        newIds = self.componentManager.addComponents(compAsDf)["added"]
        compAsDf[REQD_TBL_FIELDS.ID] = newIds
        compAsDf = compAsDf.set_index(REQD_TBL_FIELDS.ID, drop=False)

        def undo():
            self.componentManager.removeComponents(newIds)
            # Make sure the old, previously existing outline re-exists at this point
            self.verticesPlugin.updateRegionFromDf(compAsDf)

        return undo

    def _acceptFocusedExisting(self, focusedComponent: pd.Series):
        oldComp = self.componentManager.compDf.loc[
            [focusedComponent[REQD_TBL_FIELDS.ID]]
        ].copy()
        modifiedDf = fns.serAsFrame(focusedComponent)
        self.componentManager.addComponents(
            modifiedDf, addType=PRJ_ENUMS.COMPONENT_ADD_AS_MERGE
        )

        def undo():
            self.addAndFocusComponents(
                oldComp, addType=PRJ_ENUMS.COMPONENT_ADD_AS_MERGE
            )
            self.componentManager.updateFocusedComponent(focusedComponent)

        return undo

    def clearBoundaries(self):
        """Removes all components from the component table"""
        self.componentManager.removeComponents()

    def addPlugin(self, pluginCls: Type[ParamEditorPlugin], *args, **kwargs):
        """
        From a class inheriting the ``PrjParamEditorPlugin``, creates a plugin object
        that will appear in the S3A toolbar. An entry is created with dropdown options
        for each editor in ``pluginCls``'s ``editors`` attribute.

        Parameters
        ----------
        pluginCls
            Class containing plugin actions
        args
            Passed to class constructor
        kwargs
            Passed to class constructor
        """
        if pluginCls in self.classPluginMap:
            getAppLogger(__name__).info(
                f"Ignoring {pluginCls} since it was previously added", UserWarning
            )

        plugin: ParamEditorPlugin = pluginCls(*args, **kwargs)
        return self._addPluginObject(plugin)

    def _addPluginObject(self, plugin: ParamEditorPlugin, overwriteExisting=False):
        """
        Adds already intsantiated plugin. Discourage public use of this API since most
        plugin use should be class-based until window registration. This mainly
        provides for adding spawned plugins from prject data
        """
        pluginCls = type(plugin)
        if not overwriteExisting and pluginCls in self.classPluginMap:
            return None
        self.classPluginMap[pluginCls] = plugin
        if plugin.dock is not None and plugin.dock not in self.docks:
            self.docks.append(plugin.dock)
        # Many plugins register functions when attaching win
        with ParamEditor.setBaseRegisterPath(plugin.__groupingName__):
            plugin.attachWinRef(self)
        if plugin.dock:
            plugin.dock.setParent(self)
        self.sigPluginAdded.emit(plugin)
        return plugin

    @DASM.undoable("Change Main Image")
    def setMainImage(
        self, file: FilePath = None, imgData: NChanImg = None, clearExistingComps=True
    ):
        """
        * If file is None, the main and focused images are blacked out.
        * If only file is provided, it is assumed to be an image. The image data
        will be populated by reading in that file.
        * If both file and imageData are provided, then imageData is used to populate the
        image, and file is assumed to be the file associated with that data.

        Parameters
        ----------
        file
            Filename either to load or that corresponds to imageData
        imgData
            N-Channel numpy image
        clearExistingComps
            If True, erases all existing components on image load. Else, they are
            retained.
        """
        oldFile = self.sourceImagePath
        oldData = self.mainImage.image
        if file is not None:
            file = Path(file).resolve()
        if file == self.sourceImagePath:
            return

        self.saveCurrentAnnotation()

        if imgData is not None:
            self.mainImage.setImage(imgData)
        else:
            # Alpha channel usually causes difficulties with image proesses
            self.mainImage.setImage(file, stripAlpha=True)
        self.sourceImagePath = file

        self.clearBoundaries()
        self.mainImage.plotItem.vb.autoRange()
        if file is not None:
            # Add image data if the file doesn't exist
            data = None if file.exists() else self.mainImage.image
            self.filePlugin.projectData.addImage(file, data)
        self.loadNewAnnotations()
        infoName = (file and file.name) or None
        getAppLogger(__name__).info(f"Changed main image to {infoName}")
        yield
        self.setMainImage(oldFile, oldData, clearExistingComps)

    def saveCurrentAnnotation(self):
        sourceImagePath = self.sourceImagePath
        if sourceImagePath is None:
            return
        srcImg_proj = self.filePlugin.imagesPath / sourceImagePath.name
        if not srcImg_proj.exists() or srcImg_proj != sourceImagePath:
            # Either the image didn't exist (i.e. was programmatically generated) or
            # doesn't yet belong to the project
            self.filePlugin.addImage(
                sourceImagePath,
                data=self.mainImage.image,
                copyToProject=True,
                allowOverwrite=True,
            )
        # srcImg_proj is guaranteed to exist at this point
        # Suppress to avoid double-loading due to `sigAnnotationsAdded`
        with self.filePlugin.projectData.suppressSignals():
            self.filePlugin.addAnnotation(
                data=self.componentDf, image=srcImg_proj, overwriteOld=True
            )
        # Now all added components should be forced to belong to this image
        names = self.componentManager.compDf[
            [REQD_TBL_FIELDS.IMAGE_FILE, REQD_TBL_FIELDS.ID]
        ].copy()
        names.loc[:, REQD_TBL_FIELDS.IMAGE_FILE] = self.sourceImagePath.name
        self.componentManager.addComponents(
            names, addType=PRJ_ENUMS.COMPONENT_ADD_AS_MERGE
        )
        self.sourceImagePath = srcImg_proj
        self.hasUnsavedChanges = False
        getAppLogger(__name__).info("Saved current annotation")

    def loadNewAnnotations(self, imagePath: FilePath = None):
        if imagePath is None:
            imagePath = self.sourceImagePath
        if imagePath is None:
            return
        imgAnns = self.filePlugin.imageAnnotationMap.get(imagePath, None)
        if imgAnns is not None:
            self.componentManager.addComponents(
                self.componentIo.importByFileType(
                    imgAnns, imageShape=self.mainImage.image.shape
                ),
                addType=PRJ_ENUMS.COMPONENT_ADD_AS_MERGE,
            )
            # 'hasUnsavedChanges' will be true after this, even though the changes are
            # saved.
            self.hasUnsavedChanges = False

    def _maybeLoadActiveAnnotation(self, addedAnnotations: List[Path]):
        """
        When annotations are added to a project while an image is active, that image
        will not receive the new annotations. This function looks through recently
        added annotations, checks if any match the current image, and loads them in if so
        """
        # No worries if no main image is loaded
        if self.sourceImagePath is None:
            return
        srcImgName = self.sourceImagePath.name
        for annName in addedAnnotations:
            if annName.stem == srcImgName:
                self.loadNewAnnotations(self.sourceImagePath)
                # Not possible for multiple added annotations to have the same name,
                # otherwise they would overwrite, so it's safe to break here
                break

    @fns.dynamicDocstring(filters=defaultIo.ioFileFilter(PRJ_ENUMS.IO_EXPORT))
    def exportCurrentAnnotation(self, outputPath: Union[str, Path], **kwargs):
        """
        Exports current image annotations to a file. This may be more convenient than
        exporting an entire project if just the current annotations are needed

        Parameters
        ----------
        outputPath
            Where to export. The file extension determines the save type
            title: Output File
            pType: filepicker
            existing: False
            fileFilter: {filters}
        **kwargs
            Passed to the exporter
        """
        outputPath = Path(outputPath)
        self.componentIo.exportByFileType(
            self.componentDf,
            outputPath,
            imageShape=self.mainImage.image.shape,
            **kwargs,
        )
        msgPath = os.path.join(outputPath.parent.name, outputPath.name)
        getAppLogger(__name__).attention(f"Exported current annotation to {msgPath}")

    @property
    def componentDf(self):
        """
        Dataframe from manager with populated information for main image name and
        potentially filtered to only visible components (if requested by the user)
        """
        displayIds = self.componentController.displayedIds
        srcImgFname = self.sourceImagePath
        if self.props[PRJ_CONSTS.EXP_ONLY_VISIBLE] and displayIds is not None:
            exportIds = displayIds
        else:
            exportIds = self.componentManager.compDf.index
        exportDf: pd.DataFrame = self.componentManager.compDf.loc[exportIds].copy()
        if not self.props[PRJ_CONSTS.INCLUDE_FNAME_PATH] and srcImgFname is not None:
            # Only use the file name, not the whole path
            srcImgFname = srcImgFname.name
        elif srcImgFname is not None:
            srcImgFname = str(srcImgFname)
        # Assign correct export name for only new components
        overwriteIdxs = (
            exportDf[REQD_TBL_FIELDS.IMAGE_FILE] == REQD_TBL_FIELDS.IMAGE_FILE.value
        )
        # TODO: Maybe the current file will match the current file indicator. What
        #  happens then?
        exportDf.loc[overwriteIdxs, REQD_TBL_FIELDS.IMAGE_FILE] = srcImgFname
        # Ensure ids are sequential
        seqIds = np.arange(len(exportDf))
        exportDf.index = exportDf[REQD_TBL_FIELDS.ID] = seqIds
        return exportDf

    def openAnnotations(self, fileName: str, loadType=PRJ_ENUMS.COMPONENT_ADD_AS_NEW):
        pathFname = Path(fileName)
        if self.mainImage.image is None:
            raise IOError("Cannot load components when no main image is set.")
        fType = pathFname.suffix[1:]
        if not any(fType in typ for typ in self.componentIo.importTypes):
            raise IOError(
                f"Extension {fType} is not recognized. Must be one of:\n"
                + self.componentIo.ioFileFilter()
            )
        newComps = self.componentIo.importByFileType(
            fileName, self.mainImage.image.shape
        )
        self.componentManager.addComponents(newComps, loadType)

    @DASM.undoable("Create New Component")
    def addAndFocusComponents(
        self, components: pd.DataFrame, addType=PRJ_ENUMS.COMPONENT_ADD_AS_NEW
    ):
        # Capture undo action here since current stack might be disabled
        dummyStack = ActionStack()
        undoableAddComps = dummyStack.undoable()(ComponentManager.addComponents)
        changeDict = undoableAddComps(self.componentManager, components, addType)
        # Focus is performed by comp table
        # Arbitrarily choose the last possible component
        changeList = np.concatenate([changeDict["added"], changeDict["changed"]])
        oldFocused = self.componentManager.focusedComponent[REQD_TBL_FIELDS.ID]
        # Nothing to undo if there were no changes
        if len(changeList) > 0:
            self.componentController.selectRowsById([changeList[-1]])
        yield changeDict
        # Explicitly call the captured "undo" from adding components if needed
        if len(dummyStack.actions):
            dummyStack.undo()
        self.componentController.selectRowsById([oldFocused])

    def changeFocusedComponent(self, ids: Union[int, Sequence[int]] = None):
        # TODO: More robust scenario if multiple components are in the dataframe
        #   For now, treat ambiguity by not focusing anything
        if np.isscalar(ids):
            ids = [ids]
        if (
            self.mainImage.imgItem.image is None
            or ids is None
            or len(ids) != 1
            or ids[0] not in self.componentManager.compDf.index
        ):
            self.componentManager.updateFocusedComponent()
        else:
            newComp: pd.Series = self.componentManager.compDf.loc[ids[0]]
            self.componentManager.updateFocusedComponent(newComp)

    # Stolen and adapted for python from https://stackoverflow.com/a/42910109/9463643
    # noinspection PyTypeChecker
    def addTabbedDock(
        self, area: QtCore.Qt.DockWidgetArea, dockwidget: QtWidgets.QDockWidget
    ):
        curAreaWidgets = [
            d
            for d in self.findChildren(QtWidgets.QDockWidget)
            if self.dockWidgetArea(d) == area
        ]
        try:
            self.tabifyDockWidget(curAreaWidgets[-1], dockwidget)
        except IndexError:
            # First dock in area
            self.addDockWidget(area, dockwidget)

    def updateCliOptions(self, parser: argparse.ArgumentParser = None):
        """
        Adds quick loader and app state options to a parser, or creates a new parser if
        one is not passed
        """
        if parser is None:
            parser = argparse.ArgumentParser("S3A")
        ql = self.appStateEditor.quickLoader
        for editor in ql.listModel.uniqueEditors:
            states = ql.listModel.getParamStateFiles(editor.saveDir, editor.fileType)
            formatted = [f'"{s}"' for s in states]
            parser.add_argument(
                f'--{editor.name.lower().replace(" ", "")}', choices=formatted
            )
        for loader in self.appStateEditor.stateFuncsDf.index:
            parser.add_argument(f"--{loader}", type=str)
