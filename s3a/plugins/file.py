from __future__ import annotations

import inspect
import os
import pydoc
import shutil
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Optional, Set, List, Dict, Sequence, Union, Tuple, Type
from warnings import warn

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree.Parameter import PARAM_TYPES
from utilitys import (
    CompositionMixin,
    AtomicProcess,
    fns,
    ParamEditorPlugin,
    ParamEditor,
)

from ..compio import ComponentIO, defaultIo
from ..compio.base import AnnotationExporter
from ..constants import (
    APP_STATE_DIR,
    PROJECT_FILE_TYPE,
    PROJECT_BASE_TEMPLATE,
    PRJ_ENUMS,
    PRJ_CONSTS as CNST,
    REQD_TBL_FIELDS,
)
from ..generalutils import hierarchicalUpdate, cvImsaveRgb
from ..graphicsutils import DropList
from ..logger import getAppLogger
from ..structures import FilePath, NChanImg


def absolutePath(p: Optional[Path]):
    """
    Bug in ``Path.resolve`` means it doesn't actually return an absolute path on Windows
    for a non-existent path While we're here, return None nicely without raising error
    """
    if p is None:
        return None
    return Path(os.path.abspath(p))


class FilePlugin(CompositionMixin, ParamEditorPlugin):
    name = "File"

    def __init__(self, startupName: FilePath = None, startupCfg: dict = None):
        super().__init__()
        self.projectData = self.exposes(ProjectData(startupName, startupCfg))
        self.autosaveTimer = QtCore.QTimer()

        self.registerFunc(self.save, btnOpts=CNST.TOOL_PROJ_SAVE)
        self.registerFunc(self.showProjectImagesGui, btnOpts=CNST.TOOL_PROJ_OPEN_IMG)
        self.menu.addSeparator()

        self.registerFunc(self.createGui, btnOpts=CNST.TOOL_PROJ_CREATE)
        self.registerFunc(self.openGui, btnOpts=CNST.TOOL_PROJ_OPEN)

        self.registerPopoutFuncs(
            [self.updateProjectProperties, self.addImagesGui, self.addAnnotationsGui],
            ["Update Project Properties", "Add Images", "Add Annotations"],
            btnOpts=CNST.TOOL_PROJ_SETTINGS,
        )

        self.registerFunc(
            lambda: self.win.setMainImageGui(), btnOpts=CNST.TOOL_PROJ_ADD_IMG
        )
        self.registerFunc(
            lambda: self.win.openAnnotationGui(), btnOpts=CNST.TOOL_PROJ_ADD_ANN
        )

        self.registerPopoutFuncs(
            [self.startAutosave, self.stopAutosave], btnOpts=CNST.TOOL_AUTOSAVE
        )

        self._projectImagePane = ProjectImagePane()
        self._projectImagePane.sigImageSelected.connect(
            lambda imgFname: self.win.setMainImage(imgFname)
        )
        self.projectData.sigConfigLoaded.connect(
            lambda: self._projectImagePane.setRootDirectory(
                str(self.projectData.imagesPath)
            )
        )

        def onCfgLoad():
            self._updateProjectLabel()
            if self.win:
                # Other arguments are consumed by app state editor
                state = self.win.appStateEditor
                if state.loading:
                    hierarchicalUpdate(state.startupSettings, self.projectData.startup)
                else:
                    # Opening a project after s3a is already loaded
                    state.loadParamValues(stateDict={}, **self.projectData.startup)

        self.projectData.sigConfigLoaded.connect(onCfgLoad)

        self.projNameLbl = QtWidgets.QLabel()

        useDefault = startupName is None and startupCfg is None
        self._createDefaultProject(useDefault)

    def _updateProjectLabel(self):
        self.projNameLbl.setText(f"Project: {self.projectData.configPath.name}")
        self.projNameLbl.setToolTip(str(self.projectData.configPath))

    def _buildIoOptions(self):
        """
        Builds export option parameters for user interaction. Assumes export popout
        funcs have already been created
        """
        componentIo = self.projectData.componentIo
        exportOptsParam = fns.getParamChild(
            self.toolsEditor.params, CNST.TOOL_PROJ_EXPORT.name, "Export Options"
        )
        # Use a wrapper to easily get hyperparams created
        for name, fn in inspect.getmembers(
            componentIo, lambda el: isinstance(el, AnnotationExporter)
        ):
            metadata = fn.optionsMetadata()
            for child in metadata.values():
                if child["value"] is None or child["type"] not in PARAM_TYPES:
                    # Not representable
                    continue
                fns.getParamChild(exportOptsParam, chOpts=child)
        # Add catch-all that will be literally evaluated later
        exportOptsParam.addChild(
            dict(name="extra", type="text", value="", expanded=False)
        )
        return exportOptsParam

    def attachWinRef(self, win):
        super().attachWinRef(win)
        self.projectData.componentIo.tableData = win.tableData
        win.statusBar().addPermanentWidget(self.projNameLbl)

        def handleExport(_dir):
            saveImg = win.sourceImagePath
            ret = str(self.projectData.configPath)
            if not saveImg:
                self.projectData.startup.pop("image", None)
                return ret
            if saveImg and saveImg.parent == self.projectData.imagesPath:
                saveImg = saveImg.name
            self.projectData.startup["image"] = str(saveImg)
            self.save()
            return ret

        win.appStateEditor.addImportExportOptions("project", self.open, handleExport, 0)

        def receiveAutosave(autosaveArg: Union[bool, FilePath, dict]):
            """Loads autosave configuration from file and starts autosaving"""
            if isinstance(autosaveArg, bool):
                if not autosaveArg:
                    # --autosave False means don't autosave
                    self.autosaveTimer.stop()
                    return
                cfg = {}
            elif isinstance(autosaveArg, dict):
                cfg = autosaveArg
            else:
                cfg = fns.attemptFileLoad(autosaveArg)
            self.startAutosave(**cfg)

        def exportAutosave(savePath: Path):
            """
            Stores current autosave configuration at the specified location,
            if autosave is running
            """
            if not self.autosaveTimer.isActive():
                return None

            cfg = {}
            for proc, params in self.toolsEditor.procToParamsMapping.items():
                if proc.name == "Start Autosave":
                    cfg = fns.paramValues(params).pop("Start Autosave", {})
                    break

            saveName = str(savePath / "autosave.params")
            fns.saveToFile(cfg, saveName)
            return saveName

        win.appStateEditor.addImportExportOptions(
            "autosave", receiveAutosave, exportAutosave
        )

        def startImage(imgName: str):
            imgName = Path(imgName)
            if not imgName.is_absolute():
                imgName = self.projectData.imagesPath / imgName
            if not imgName.exists():
                return
            addedName = self.projectData.addImage(imgName)
            self.win.setMainImage(addedName or imgName)

        win.appStateEditor.addImportExportOptions(
            "image", startImage, lambda *args: None, 1
        )

        def exportWrapper(func):
            def wrapper(**kwargs):
                initial = {**self.exportOptsParam}
                # Fixup the special "extra" parameter
                extra = initial.pop("extra")
                if extra:
                    try:
                        newOpts = eval(f"dict({extra})", {}, {})
                        initial.update(newOpts)
                    except Exception as ex:
                        warn(f"Could not parse extra arguments:\n{ex}")
                initial.update(kwargs)
                return func(**initial)

            return wrapper

        doctoredCur = AtomicProcess(
            exportWrapper(win.exportCurrentAnnotation),
            "Current Annotation",
            outFname="",
            docFunc=win.exportCurrentAnnotation,
        )
        doctoredAll = AtomicProcess(
            exportWrapper(self.projectData.exportAnnotations),
            docFunc=self.projectData.exportAnnotations,
        )
        self.registerPopoutFuncs(
            [self.projectData.exportProject, doctoredAll, doctoredCur],
            ["Project", "All Annotations", "Current Annotation"],
            btnOpts=CNST.TOOL_PROJ_EXPORT,
        )
        self._projectImagePane.hide()
        self._updateProjectLabel()
        win.addTabbedDock(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._projectImagePane
        )
        self.exportOptsParam = self._buildIoOptions()

    def _createDefaultProject(self, setAsCur=True):
        defaultName = APP_STATE_DIR / PROJECT_FILE_TYPE
        # Delete default prj on startup, if not current
        if not setAsCur:
            for prj in defaultName.glob(f"*.{PROJECT_FILE_TYPE}"):
                prj.unlink()
        parent = self.projectData if setAsCur else None
        if setAsCur or not defaultName.exists():
            # Otherwise, no action needed
            self.projectData.create(name=defaultName, parent=parent)

    def open(self, configPath: FilePath = None, configDict: dict = None):
        if configPath is None and configDict is None:
            # Do nothing, happens when e.g. intentionally not loading a project
            return
        _, configDict = fns.resolveYamlDict(configPath, configDict)
        if configPath is not None and (
            absolutePath(configPath) != self.projectData.configPath
            or not pg.eq(configDict, self.projectData.config)
        ):
            self.win.setMainImage(None)
            self.projectData.loadConfig(configPath, configDict, force=True)

    def openGui(self):
        fname = fns.popupFilePicker(
            None, "Select Project File", f"S3A Project (*.{PROJECT_FILE_TYPE})"
        )
        with pg.BusyCursor():
            self.open(fname)

    def save(self):
        self.win.saveCurrentAnnotation()
        self.projectData.saveConfig()
        getAppLogger(__name__).attention("Saved project")

    @fns.dynamicDocstring(ioTypes=["<Unchanged>"] + list(defaultIo.roundTripTypes))
    def updateProjectProperties(
        self, tableConfig: FilePath = None, annotationFormat: str = None
    ):
        """
        Updates the specified project properties, for each one that is provided

        Parameters
        ----------
        tableConfig
            pType: filepicker
        annotationFormat
            How to save annotations internally. Note that altering this value may
            alter the speed of saving and loading annotations.
            pType: list
            limits: {ioTypes}
        """
        if tableConfig is not None:
            tableConfig = Path(tableConfig)
            self.projectData.tableData.loadConfig(tableConfig)
        if (
            annotationFormat is not None
            and annotationFormat in self.projectData.componentIo.importTypes
            and annotationFormat in self.projectData.componentIo.exportTypes
        ):
            self.projectData.config["annotation-format"] = annotationFormat

    @fns.dynamicDocstring(ioTypes=list(defaultIo.exportTypes))
    def startAutosave(
        self, interval=5, backupFolder="", baseName="autosave", annotationFormat="csv"
    ):
        """
        Saves the current annotation set evert *interval* minutes

        Parameters
        ----------
        interval
            Interval in minutes between saves
            limits: [1, None]
        backupFolder
            If provided, annotations are saved here sequentially afte reach *interval*
            minutes. Each output is named `[Parent Folder]/[base name]_[counter].[
            export type]`, where `counter` is the current save file number.
            pType: filepicker
            asFolder: True
        baseName
            What to name the saved annotation file
        annotationFormat
            File format for backups
            pType: list
            limits: {ioTypes}
        """
        self.autosaveTimer.stop()
        self.autosaveTimer = QtCore.QTimer()
        saveToBackup = len(str(backupFolder)) > 0
        if not saveToBackup:
            getAppLogger(__name__).attention(
                "No backup folder selected. Will save to project without saving to "
                "backup directory."
            )
        backupFolder = Path(backupFolder)
        backupFolder.mkdir(exist_ok=True, parents=True)
        # LGTM false positive, used in closure below
        lastSavedDf = self.win.componentDf.copy()  # lgtm [py/unused-local-variable]
        # Qtimer expects ms, turn mins->s->ms
        # Figure out where to start the counter
        globExpr = lambda: backupFolder.glob(f"{baseName}*.{annotationFormat}")
        existingFiles = list(globExpr())
        if len(existingFiles) == 0:
            counter = 0
        else:
            counter = (
                max(map(lambda fname: int(fname.stem.rsplit("_")[1]), existingFiles))
                + 1
            )

        def saveAndIncrementCounter():
            nonlocal counter, lastSavedDf
            baseSaveNamePlusFolder = (
                backupFolder / f"{baseName}_{counter}.{annotationFormat}"
            )
            counter += 1
            curDf = self.win.componentDf
            # A few things can go wrong during the comparison
            # noinspection PyBroadException
            try:
                isEqual = curDf.equals(lastSavedDf)
            except Exception:
                isEqual = False
            if not isEqual:
                self.win.exportCurrentAnnotation(baseSaveNamePlusFolder)
                lastSavedDf = curDf.copy()

        if saveToBackup:
            self.autosaveTimer.timeout.connect(saveAndIncrementCounter)
        self.autosaveTimer.timeout.connect(self.win.saveCurrentAnnotation)
        self.autosaveTimer.start(int(interval * 60 * 1000))
        getAppLogger(__name__).attention(f"Started autosaving")

    def stopAutosave(self):
        self.autosaveTimer.stop()
        getAppLogger(__name__).attention(f"Stopped autosaving")

    def showProjectImagesGui(self):
        if len(self.projectData.images) == 0:
            getAppLogger(__name__).warning(
                "This project does not have any images yet. You can add them either in\n"
                "`Project Settings... > Add Image Files` or\n"
                "`File > Add New Image`."
            )
            return
        self._projectImagePane.show()
        self._projectImagePane.raise_()

    def addImagesGui(self):
        wiz = QtWidgets.QWizard()
        page = NewProjectWizard.createFilePage("Images", wiz)
        wiz.addPage(page)
        if wiz.exec_():
            for file in page.fileList.files:
                self.projectData.addImageByPath(file)

    def addAnnotationsGui(self):
        wiz = QtWidgets.QWizard()
        page = NewProjectWizard.createFilePage("Annotations", wiz)
        wiz.addPage(page)
        if wiz.exec_():
            # Make sure if a file is added on top of the current image, recent changes
            # don't get lost
            self.win.saveCurrentAnnotation()
            for file in page.fileList.files:
                self.projectData.addAnnotationByPath(file)

    def createGui(self):
        wiz = NewProjectWizard(self)
        ok = wiz.exec_()
        if not ok:
            return
        parsedLists = {}
        for k, val in wiz.fileLists.items():
            parsedLists[k] = val.files
        # Since insertion order is preserved the extraction can be done without keys
        images, annotations = parsedLists.values()
        settings = wiz.projSettings
        projName = settings["Name"]
        prevTemplate = settings["Template Project"]
        if prevTemplate is not None and len(prevTemplate) > 0:
            baseCfg = fns.attemptFileLoad(prevTemplate)
        else:
            baseCfg = {}
        projPath = Path(wiz.projSettings["Location"]) / projName
        outPrj = self.projectData.create(name=projPath, cfg=baseCfg)
        if prevTemplate:
            prevTemplateLoc = Path(prevTemplate).parent
            if settings["Keep Existing Images"]:
                outPrj.addImageFolder(prevTemplateLoc / "images")
            if settings["Keep Existing Annotations"]:
                outPrj.addImageFolder(prevTemplateLoc / "annotations")
        for image in images:
            outPrj.addImageByPath(image)
        for ann in annotations:
            outPrj.addAnnotation(ann)

        self.open(outPrj.configPath)


class NewProjectWizard(QtWidgets.QWizard):
    def __init__(self, project: FilePlugin, parent=None) -> None:
        super().__init__(parent)
        self.project = project
        self.fileLists: Dict[str, DropList] = {}

        # -----
        # PROJECT SETTINGS
        # -----
        page = QtWidgets.QWizardPage(self)
        page.setTitle("Project Settings")
        settings = [
            dict(name="Name", type="str", value="new-project"),
            dict(name="Location", type="filepicker", value=".", asFolder=True),
            dict(
                name="Template Project",
                type="filepicker",
                value=None,
                tip="Path to existing project config file. This will serve as a template "
                "for the newly created project, except for overridden settings",
            ),
            dict(
                name="Keep Existing Images",
                type="bool",
                value=True,
                tip="Whether to keep images specified in the existing config",
            ),
            dict(
                name="Keep Existing Annotations",
                type="bool",
                value=True,
                tip="Whether to keep annotations specified in the existing config",
            ),
        ]
        # Use ParamEditor for speedy tree building
        editor = ParamEditor(paramList=settings, saveDir=None)
        tree = editor.tree
        self.projSettings = editor.params
        self.nameToPageMapping: Dict[str, QtWidgets.QWizardPage] = {}
        layout = QtWidgets.QVBoxLayout()
        page.setLayout(layout)
        layout.addWidget(tree)
        self.addPage(page)

        # -----
        # SELECTING PROJECT FILES
        # -----
        for fType in ["Images", "Annotations"]:
            page = self.createFilePage(fType, self)
            self.fileLists[fType] = page.fileList
            self.addPage(page)
            self.nameToPageMapping[fType] = page

    @staticmethod
    def createFilePage(name: str, wizard=None):
        page = QtWidgets.QWizardPage(wizard)
        page.setTitle(name)
        curLayout = QtWidgets.QVBoxLayout()
        page.setLayout(curLayout)
        curLayout.addWidget(
            QtWidgets.QLabel(
                f"New project {name.lower()} are shown below. Use the buttons"
                " or drag and drop to add files."
            )
        )
        flist = DropList(wizard)
        page.fileList = flist
        fileBtnLayout = QtWidgets.QHBoxLayout()
        curLayout.addWidget(flist)
        for title in f"Add Files", f"Add Folder":
            selectFolder = "Folder" in title
            btn = QtWidgets.QPushButton(title, wizard)
            btn.clicked.connect(
                partial(getFileListGui, wizard, flist, title, selectFolder)
            )
            fileBtnLayout.addWidget(btn)
        curLayout.addLayout(fileBtnLayout)
        return page


def getFileListGui(wizard, _flist: DropList, _title: str, _selectFolder=False):
    dlg = QtWidgets.QFileDialog()
    dlg.setModal(True)
    getFn = lambda *args, **kwargs: dlg.getOpenFileNames(
        *args, **kwargs, options=dlg.DontUseNativeDialog
    )[0]
    if _selectFolder:
        getFn = lambda *args, **kwargs: [
            dlg.getExistingDirectory(*args, **kwargs, options=dlg.DontUseNativeDialog)
        ]
    files = getFn(wizard, _title)
    _flist.addItems(files)


class ProjectImagePane(QtWidgets.QDockWidget):
    sigImageSelected = QtCore.Signal(str)  # Selected image name
    sigDeleteRequested = QtCore.Signal(str)  # Image name to delete

    def __init__(self, parent=None, rootDir: FilePath = None):
        super().__init__(parent)
        self.setWindowTitle("Project Images")
        self.setObjectName("Project Images")
        self.setFeatures(
            self.DockWidgetMovable | self.DockWidgetFloatable | self.DockWidgetClosable
        )
        wid = QtWidgets.QWidget()
        self.setWidget(wid)

        self.rootDir = None

        layout = QtWidgets.QVBoxLayout()
        wid.setLayout(layout)

        self.fileModel = QtWidgets.QFileSystemModel(self)
        self.fileModel.setNameFilterDisables(False)

        self.fileViewer = QtWidgets.QTreeView(self)
        self.fileViewer.setModel(self.fileModel)
        self.fileViewer.setSortingEnabled(True)

        self.completer = QtWidgets.QLineEdit()
        self.completer.setPlaceholderText("Type to filter")
        self.completer.textChanged.connect(self._filterFiles)

        self.fileViewer.activated.connect(self._emitFileName)

        layout.addWidget(self.completer)
        layout.addWidget(self.fileViewer)

    def _emitFileName(self, idx: QtCore.QModelIndex):
        self.sigImageSelected.emit(str(self.rootDir / idx.siblingAtColumn(0).data()))

    def _filterFiles(self, text):
        self.fileModel.setNameFilters([f"*{text}*"])

    def setRootDirectory(self, rootPath: FilePath):
        self.rootDir = Path(rootPath).absolute()
        rootPath = str(self.rootDir)
        rootIdx = self.fileModel.index(rootPath)
        if not rootIdx.isValid():
            raise ValueError(f"Invalid root directory specified: {rootPath}")
        self.fileModel.setRootPath(rootPath)
        self.fileViewer.setRootIndex(rootIdx)


class ProjectData(QtCore.QObject):
    sigConfigLoaded = QtCore.Signal()

    sigImagesAdded = QtCore.Signal(object)
    """List[Path] of added images"""
    sigImagesRemoved = QtCore.Signal(object)
    """List[Path] of removed images"""
    sigImagesMoved = QtCore.Signal(object)
    """
    List[(oldPath, NewPath)] Used mainly when images from outside the project are 
    annotated. In that case, images are copied to inside the project, and this signal 
    will be emitted.
    """

    sigAnnotationsAdded = QtCore.Signal(object)
    """List[Path]"""
    sigAnnotationsRemoved = QtCore.Signal(object)
    """List[Path]"""

    def __init__(self, configPath: FilePath = None, cfgDict: dict = None):
        super().__init__()
        self.componentIo = ComponentIO()
        self.templateName = PROJECT_BASE_TEMPLATE
        self.config = fns.attemptFileLoad(self.templateName)
        self.configPath: Optional[Path] = None
        self.images: List[Path] = []
        self.imageFolders: Set[Path] = set()
        self.imageAnnotationMap: Dict[Path, Path] = {}
        """Records annotations belonging to each image"""
        self.spawnedPlugins: List[ParamEditorPlugin] = []
        """
        Plugin instances stored separately from plugin-cfg to maintain serializability 
        of ``self.config`` 
        """

        self._suppressSignals = False
        """If this is *True*, no signals will be emitted """
        self.watcher = QtCore.QFileSystemWatcher()
        self.watcher.directoryChanged.connect(self._handleLocationChange)

        if configPath is not None or cfgDict is not None:
            self.loadConfig(configPath, cfgDict)

    def _handleLocationChange(self):
        imgs = list(self.imagesPath.glob("*.*"))
        # Images already in the project will be ignored on add
        newImgs = []
        for img in imgs:
            new = self.addImage(img)
            if new:
                newImgs.append(img)
        if newImgs:
            self.sigImagesAdded.emit(newImgs)
        # Handle removals
        delImgs = []
        delIdxs = []
        for ii, img in enumerate(self.images):
            if img.parent == self.imagesPath and img not in imgs:
                delIdxs.append(ii)
                delImgs.append(img)
        for idx in delIdxs:
            del self.images[idx]
        self.sigImagesRemoved.emit(delImgs)

        anns = list(self.annotationsPath.glob(f'*.{self.config["annotation-format"]}'))
        # Images already in the project will be ignored on add
        # Assume new annotations here are already formatted properly
        for ann in anns:
            if ann not in self.imageAnnotationMap.values():
                self.addFormattedAnnotation(ann)
        # Convert to list to avoid "dictionary changed size on iteration" error
        for ann in list(self.imageAnnotationMap.values()):
            if ann not in anns:
                self.removeAnnotation(ann)

    @property
    def location(self):
        return self.configPath.parent if self.configPath else None

    @property
    def imagesPath(self):
        return self.location / "images"

    @property
    def annotationsPath(self):
        return self.location / "annotations"

    @property
    def startup(self):
        return self.config["startup"]

    @property
    def pluginConfig(self) -> Dict[str, str]:
        return self.config["plugin-cfg"]

    @property
    def tableData(self):
        return self.componentIo.tableData

    def clearImagesAndAnnotations(self):
        oldImgs = self.images.copy()
        for lst in self.images, self.imageAnnotationMap, self.imageFolders:
            lst.clear()
        self._maybeEmit(self.sigImagesRemoved, oldImgs)

    def loadConfig(self, configPath: FilePath, configDict: dict = None, force=False):
        """
        Parameters
        ----------
        Loads the specified project configuration by name (if a file) or dict (if
        programmatically created)

        configPath
            Name of file to open. If `configDict` is provided instead, it will be
            saved here.
        configDict
            If provided, this config is used and saved to `configPath` instead of
            using the file.
        force
            If *True*, the new config will be loaded even if it is the same name
            as the current config
        """
        _, baseCfgDict = fns.resolveYamlDict(self.templateName)
        configPath, configDict = fns.resolveYamlDict(configPath, configDict)
        configPath = absolutePath(configPath)
        if not force and self.configPath == configPath:
            return None

        hierarchicalUpdate(baseCfgDict, configDict, uniqueListElements=True)

        loadPrjPlugins = baseCfgDict.get("plugin-cfg", {})
        newPlugins = {
            k: v for (k, v) in loadPrjPlugins.items() if k not in self.pluginConfig
        }
        removedPlugins = set(self.pluginConfig).difference(loadPrjPlugins)
        if removedPlugins:
            removedPluginsStr = ", ".join(removedPlugins)
            raise ValueError(
                f"The previous project loaded custom plugins, which cannot easily be "
                f"removed. To load a new project without plugin(s) {removedPluginsStr}, "
                f"close and re-open S3A with the new project instance instead. "
                f"Alternatively, add these missing plugins to the project you wish to "
                f"add."
            )
        warnPlgs = []
        for plgName, plgPath in newPlugins.items():
            # noinspection PyTypeChecker
            pluginCls: Type[ParamEditorPlugin] = pydoc.locate(plgPath)
            if pluginCls:
                # False Positive
                # noinspection PyCallingNonCallable
                self.spawnedPlugins.append(pluginCls())
            elif not pluginCls:
                warnPlgs.append(plgPath)
        if warnPlgs:
            warn(
                f"Some project plugins were specified, but could not be found:\n"
                f"{warnPlgs}",
                UserWarning,
            )

        self.configPath = configPath
        cfg = self.config = baseCfgDict

        self.clearImagesAndAnnotations()

        tableInfo = cfg.get("table-config", None)
        if isinstance(tableInfo, str):
            tableDict = None
            tableName = tableInfo
        else:
            if tableInfo is None:
                tableInfo = {}
            tableDict = tableInfo
            tableName = configPath
        tableName = Path(tableName)
        if not tableName.is_absolute() and configPath:
            tableName = self.location / tableName
        self.tableData.loadConfig(tableName, tableDict, force=True)

        if configPath:
            self._hookupProjectDirectoryInfo()

        self.sigConfigLoaded.emit()
        return self.configPath

    def _hookupProjectDirectoryInfo(self):
        """
        For projects that have backing files (i.e. the config is a file instead of a
        dict in memory), this function handles creating images/annotations dirs,
        adding project images/annotations, and adding file watchers
        """
        self.annotationsPath.mkdir(exist_ok=True)
        self.imagesPath.mkdir(exist_ok=True)

        allAddedImages = []
        with self.suppressSignals():
            allAddedImages.extend(
                self.addImageFolder(self.imagesPath, copyToProject=False)
            )
            # Leave out for now due to the large number of problems with lacking
            # idempotence allAddedImages.extend(self._addConfigImages())
        self._maybeEmit(self.sigImagesAdded, allAddedImages)

        with self.suppressSignals():
            for file in self.annotationsPath.glob(
                f'*.{self.config["annotation-format"]}'
            ):
                self.addFormattedAnnotation(file)
            # Leave out for now due to the large number of problems with lacking
            # idempotence self._addConfigAnnotations()

        self._maybeEmit(
            self.sigAnnotationsAdded, list(self.imageAnnotationMap.values())
        )

        dirs = self.watcher.directories()
        if dirs:
            self.watcher.removePaths(dirs)
        self.watcher.addPaths([str(self.imagesPath), str(self.annotationsPath)])
        for ioType in PRJ_ENUMS.IO_EXPORT, PRJ_ENUMS.IO_IMPORT:
            self.componentIo.updateOptions(ioType, source=self.imagesPath)

    @classmethod
    def create(
        cls,
        *,
        name: FilePath = f"./{PROJECT_FILE_TYPE}",
        config: dict = None,
        parent: ProjectData = None,
    ):
        """
        Creates a new project with the specified settings in the specified directory.

        Parameters
        ----------
        name
            Project Name. The parent directory of this name indicates the directory
            in which to create the project
            pType: filepicker
        config
            see ``ProjectData.loadConfig`` for information
        parent
            Associated ProjectData instance for a non-classmethod version of this function
        """
        name = Path(name)
        name = name / f"{name.name}.{PROJECT_FILE_TYPE}"
        location = name.parent
        location.mkdir(exist_ok=True, parents=True)
        if parent is None:
            parent = cls()

        if not name.exists() and config is None:
            config = {}
        parent.loadConfig(name, config)

        parent.saveConfig()
        return parent

    def saveConfig(self, copyMissingItems=False):
        """
        Saves the config file, optionally copying missing items to the project location
        as well. "Missing items" are images and annotations in base folders / existing
        in the project config but not in the actual project directory
        """
        if copyMissingItems:
            for image in self._findMissingImages():
                self.addImageByPath(image)
            for annotation in self._findMissingAnnotations():
                self.addAnnotationByPath(annotation)

        tblName = Path(self.tableData.configPath).absolute()
        if tblName != self.configPath:
            if tblName.parent == self.location:
                tblName = tblName.name
            self.config["table-config"] = str(tblName)
        fns.saveToFile(self.config, self.configPath)

    def _findMissingAnnotations(self):
        annDir = self.annotationsPath
        missingAnns = []
        for img, ann in self.imageAnnotationMap.items():
            if ann.parent != annDir:
                missingAnns.append(str(ann))
        return missingAnns

    def _findMissingImages(self):
        location = self.location
        imgDir = self.imagesPath
        strImgNames = []
        for folder in self.imageFolders:
            if location in folder.parents:
                folder = folder.relative_to(location)
            strImgNames.append(str(folder))
        for img in self.images:
            if img.parent == imgDir or img.parent in self.imageFolders:
                # This image is already accounted for in the base directories
                continue
            strImgNames.append(str(absolutePath(img)))
        return strImgNames

    def _addConfigImages(self):
        addedImages = []
        for image in self.config.get("images", []):
            if isinstance(image, dict):
                image.setdefault("copyToProject", True)
                renamed = self.addImage(**image)
                if renamed is not None:
                    addedImages.append(renamed)
            else:
                addedImages.extend(self.addImageByPath(image))
        return addedImages

    def _addConfigAnnotations(self):
        for annotation in set(self.config.get("annotations", [])) - {
            self.annotationsPath
        }:
            if isinstance(annotation, dict):
                self.addAnnotation(**annotation)
            else:
                self.addAnnotationByPath(annotation)

    def addImageByPath(self, name: FilePath, copyToProject=True):
        """
        Determines whether to add as a folder or file based on filepath type. Since
        adding a folder returns a list of images and adding a single image returns a
        name or None, this function unifies the return signature by always returning a
        list. If the path is a single image and not a folder, and the return value is
        *None*, this function will return an empty list instead.
        """
        image = Path(name)
        if not image.is_absolute():
            image = self.location / image
        if not image.exists():
            getAppLogger(__name__).attention(
                f"Provided image path does not exist: {image}\nNo action performed."
            )
            return []
        if image.is_dir():
            ret = self.addImageFolder(image, copyToProject)
        else:
            ret = self.addImage(name, copyToProject=copyToProject)
            ret = [] if ret is None else [ret]
        return ret

    def addImage(
        self,
        name: FilePath,
        data: NChanImg = None,
        copyToProject=True,
        allowOverwrite=False,
    ) -> Optional[FilePath]:
        """
        Returns None if an image with the same name already exists in the project,
        else the new full filepath
        """
        fullName = Path(name)
        if not fullName.is_absolute():
            fullName = self.imagesPath / fullName
        if copyToProject or data is not None:
            fullName = self._copyImageToProject(fullName, data, allowOverwrite)
        if fullName.name in [i.name for i in self.images]:
            # Indicate the image was already present to calling scope
            return None
        self.images.append(fullName)
        self._maybeEmit(self.sigImagesAdded, [fullName])
        return fullName
        # TODO: Create less hazardous undo operation
        # yield name
        # self.removeImage(name)

    def changeImagePath(self, oldName: Path, newName: Path = None):
        """
        Changes the filepath associated with a project image. Note that this doesn't do
        anything to the path of either oldName or newName (i.e. the actual image isn't
        moved/copied/etc.), it just changes the association within the project from the
        old image to the new image.
          * If newName is None, the association is deleted. Don't do this for images
            inside the project directory.
          * if newName already exists in the current project (i.e. matches an image
            already added to the project), oldName is deleted from the project
            associations
          * Otherwise, oldName is re-associated to newName.
        """
        oldName = absolutePath(oldName)
        oldIdx = self.images.index(oldName)
        if newName is not None:
            newName = absolutePath(newName)
        if newName is None or newName in self.images:
            del self.images[oldIdx]
        else:
            self.images[oldIdx] = newName
        self._maybeEmit(self.sigImagesMoved, [(oldName, newName)])

    def addImageFolder(self, folder: FilePath, copyToProject=True):
        folder = absolutePath(folder)
        if folder in self.imageFolders:
            return []
        # Need to keep track of actually added images instead of using all globbed
        # images. If an added image already existed in the project, it won't be added.
        # Also, if the images are copied into the project, the paths will change.
        addedImgs = []
        with self.suppressSignals():
            for img in folder.glob("*.*"):
                finalName = self.addImage(img, copyToProject=copyToProject)
                if finalName is not None:
                    addedImgs.append(finalName)
        self._maybeEmit(self.sigImagesAdded, addedImgs)

        return addedImgs

    def addAnnotationByPath(self, name: FilePath):
        """Determines whether to add as a folder or file based on filepath type"""
        name = Path(name)
        if not name.is_absolute():
            name = self.location / name
        if not name.exists():
            getAppLogger(__name__).attention(
                f"Provided annotation path does not exist: {name}\nNo action performed."
            )
            return
        if name.is_dir():
            self.addAnnotationFolder(name)
        else:
            self.addAnnotation(name)

    def addAnnotationFolder(self, folder: FilePath):
        folder = absolutePath(folder)
        for file in folder.glob("*.*"):
            self.addAnnotation(file)

    def addImageGui(self, copyToProject=True):
        fileFilter = (
            "Image Files (*.png *.tif *.jpg *.jpeg *.bmp *.jfif);;All files(*.*)"
        )
        fname = fns.popupFilePicker(None, "Add Image to Project", fileFilter)
        if fname is not None:
            self.addImage(fname, copyToProject=copyToProject)

    def removeImage(self, imageName: FilePath):
        imageName = absolutePath(imageName)
        if imageName not in self.images:
            return
        self.images.remove(imageName)
        # Remove copied annotations for this image
        for ann in self.annotationsPath.glob(f"{imageName.stem}.*"):
            ann.unlink()
        self.imageAnnotationMap.pop(imageName, None)
        self._maybeEmit(self.sigImagesRemoved, [imageName])
        if imageName.parent == self.imagesPath:
            imageName.unlink()
        # yield
        # if imageName.parent == self.imagesPath:
        #     # TODO: Cache removed images in a temp dir, then move them to that temp dir
        #     #  instead of unlinking on delete. This will make 'remove' undoable
        #     raise IOError(
        #         "Can only undo undo image removal when the image was outside the "
        #         f"project. Image `{imageName.name}` was either annotated or directly "
        #         "placed in the project images directory, and was deleted during "
        #         "removal. To re-add, do so from the original image location outside "
        #         " the project directory."
        #     )
        # self.addImage(imageName)

    def removeAnnotation(self, annotationName: FilePath):
        annotationName = absolutePath(annotationName)
        # Since no mapping exists of all annotations, loop the long way until the file
        # is found
        for key, ann in list(self.imageAnnotationMap.items()):
            if annotationName == ann:
                del self.imageAnnotationMap[key]
                ann.unlink(missing_ok=True)
                self._maybeEmit(self.sigAnnotationsRemoved, [ann])
                break

    def addAnnotation(
        self,
        name: FilePath = None,
        data: pd.DataFrame = None,
        image: FilePath = None,
        overwriteOld=False,
    ):
        # Housekeeping for default arguments
        if name is None and data is None:
            raise IOError("`name` and `data` cannot both be `None`")
        elif name in self.imageAnnotationMap.values() and not overwriteOld:
            # Already present, shouldn't be added
            return
        if data is None:
            data = self.componentIo.importByFileType(name)
        if image is None:
            # If no explicit matching to an image is provided, try to determine based
            # on annotation name
            xpondingImgs = np.unique(data[REQD_TBL_FIELDS.IMAGE_FILE].to_numpy())
            # Break into annotaitons by iamge
            for img in xpondingImgs:
                # Copy to avoid pandas warning
                self.addAnnotation(
                    name, data[data[REQD_TBL_FIELDS.IMAGE_FILE] == img].copy(), img
                )
            return
        image = self.getFullImgName(Path(image))
        # Force provided annotations to now belong to this image
        data.loc[:, REQD_TBL_FIELDS.IMAGE_FILE] = image.name
        # Since only one annotation file can exist per image, concatenate this with any
        # existing files for the same image if needed
        if image.parent != self.imagesPath:
            # None result means already exists in project prior to copying, which is OK
            # in this context
            image = self.addImage(image) or self.imagesPath / image.name
        annForImg = self.imageAnnotationMap.get(image, None)
        oldAnns = []
        if annForImg is not None and not overwriteOld:
            oldAnns.append(self.componentIo.importByFileType(annForImg))
        combinedAnns = oldAnns + [data]
        outAnn = pd.concat(combinedAnns, ignore_index=True)
        outAnn[REQD_TBL_FIELDS.ID] = outAnn.index
        outFmt = f".{self.config['annotation-format']}"
        outName = self.annotationsPath / f"{image.name}{outFmt}"
        # If no annotations exist, this is the same as deleting the old annotations
        # since there's nothing to save
        if len(outAnn):
            self.componentIo.exportByFileType(
                outAnn, outName, verifyIntegrity=False, readonly=False
            )
            self.imageAnnotationMap[image] = outName
            self._maybeEmit(self.sigAnnotationsAdded, [outName])
        elif outName.exists():
            self.removeAnnotation(outName)

    def addFormattedAnnotation(self, file: FilePath, overwriteOld=False):
        """
        Adds an annotation file that is already formatted in the following ways:
          * The right source image file column (i.e. REQD_TBL_FIELDS.IMAGE_FILE set
            to the image name
          * The file stem already matches a project image (not remote, i.e. an image
            in the `images` directory)
          * The annotations correspond to exactly one image
        """
        image = self.getFullImgName(file.stem, thorough=False)
        if file.parent != self.annotationsPath:
            if not overwriteOld and (self.annotationsPath / file.name).exists():
                raise IOError(
                    f"Cannot add annotation {file} since a corresponding annotation "
                    "with that name already exists and `overwriteOld` was False"
                )
            newName = self.annotationsPath / file.name
            shutil.copy2(file, newName)
            file = newName
        self.imageAnnotationMap[image] = file

    def _copyImageToProject(self, name: Path, data: NChanImg = None, overwrite=False):
        newName = self.imagesPath / name.name
        if newName.exists() and (not overwrite or newName == name):
            # Already in the project, no need to copy
            return newName
        if name.exists() and data is None:
            shutil.copy(name, newName)
        elif data is not None:
            # Programmatically created or not from a local file
            # noinspection PyTypeChecker
            cvImsaveRgb(newName, data)
        else:
            raise IOError(
                f"No image data associated with {name.name}. Either the file does "
                f"not exist or no image information was provided."
            )
        if name in self.images:
            self.changeImagePath(name, newName)
            self._maybeEmit(self.sigImagesMoved, [(name, newName)])
        return newName

    def getFullImgName(self, name: FilePath, thorough=True):
        """
        From an absolute or relative image name, attempts to find the absolute path it
        corresponds to based on current project images. A match is located in the
        following order:
          - If the image path is already absolute, it is resolved, checked for existence,
            and returned
          - Solitary project images are searched to see if they end with the specified
            relative path
          - All base image directories are checked to see if they contain this subpath

        Parameters
        ----------
        name
            Image name
        thorough
            If `False`, as soon as a match is found the function returns. Otherwise,
            all solitary paths and images will be checked to ensure there is exactly
            one matching image for the name provided.
        """
        name = Path(name)
        if name.is_absolute():
            # Ok to call 'resolve', since relative paths are the ones with issues.
            return name.resolve()

        candidates = set()
        strName = str(name)
        # Compare using "endswith" when folder names are included, otherwise a direct
        # analysis of the name is preferred
        if strName == name.name:
            compare = lambda strName, selfImageName: strName == selfImageName.name
        else:
            compare = lambda strName, selfImageName: str(selfImageName).endswith(
                strName
            )
        for img in self.images:
            if compare(strName, img):
                if not thorough:
                    return img
                candidates.add(img)

        for parent in self.imageFolders:
            curName = parent / name
            if curName.exists():
                if not thorough:
                    return curName
                candidates.add(curName)

        numCandidates = len(candidates)
        if numCandidates != 1:
            msg = (
                f"Exactly one corresponding image file must exist for a given "
                f"annotation. However, {numCandidates} candidate images were found "
                f"for image {name.name}"
            )
            if numCandidates == 0:
                msg += "."
            else:
                msg += f':\n{", ".join([c.name for c in candidates])}'
            raise IOError(msg)
        return candidates.pop()

    def exportProject(self, outputFolder: FilePath = "s3a-export"):
        """
        Exports the entire project, making a copy of it at the destination directory

        Parameters
        ----------
        outputFolder
            Where to place the exported project
            pType: filepicker
            asFolder: True
        """
        shutil.copytree(self.location, outputFolder)
        getAppLogger(__name__).info("Exported project")

    @fns.dynamicDocstring(fileTypes=list(defaultIo.exportTypes))
    def exportAnnotations(
        self,
        outputFolder: FilePath = "s3a-export",
        annotationFormat="csv",
        combine=False,
        includeImages=True,
        **exportOpts,
    ):
        """
        Exports project annotations, optionally including their source images

        Parameters
        ----------
        outputFolder
            Folder for exported annotations
            pType: filepicker
            asFolder: True
        annotationFormat
            Annotation file type. E.g. if 'csv', annotations will be saved as csv files.
            Available file types are: {fileTypes}
            pType: list
            limits: {fileTypes}
        combine
            If `True`, all annotation files will be combined into one exported file
            with name `annotations.<format>`
        includeImages
            If `True`, the corresponding image for each annotation will also be exported
            into an `images` folder
        exportOpts
            Additional options passed to the exporting function
        """
        self.saveConfig()
        outputFolder = Path(outputFolder)

        if outputFolder.resolve() == self.location and not combine:
            return

        outputFolder.mkdir(exist_ok=True)
        if includeImages:
            outImgDir = outputFolder / "images"
            outImgDir.mkdir(exist_ok=True)
            for img in self.imagesPath.glob("*.*"):
                if self.imageAnnotationMap.get(img, None) is not None:
                    shutil.copy(img, outImgDir)

        existingAnnFiles = [
            f for f in self.imageAnnotationMap.values() if f is not None
        ]
        if combine:
            outAnnsPath = outputFolder / f"annotations.{annotationFormat}"
            outAnn = pd.concat(
                map(self.componentIo.importByFileType, existingAnnFiles),
                ignore_index=True,
            )
            outAnn[REQD_TBL_FIELDS.ID] = outAnn.index
            self.componentIo.exportByFileType(outAnn, outAnnsPath, **exportOpts)
        else:
            outAnnsPath = outputFolder / "annotations"
            outAnnsPath.mkdir(exist_ok=True)
            if self.config["annotation-format"] == annotationFormat:
                shutil.copytree(self.annotationsPath, outAnnsPath, dirs_exist_ok=True)
            else:
                for annFile in existingAnnFiles:
                    self.componentIo.convert(
                        annFile,
                        outAnnsPath / f"{annFile.stem}.{annotationFormat}",
                        importArgs=exportOpts,
                        exportArgs=exportOpts,
                    )
        getAppLogger(__name__).attention(
            f"Exported project annotations to "
            f"{os.path.join(outAnnsPath.parent.name, outAnnsPath.name)}"
        )

    def _maybeEmit(
        self, signal: QtCore.Signal, emitList: Sequence[Union[Path, Tuple[Path, Path]]]
    ):
        if not self._suppressSignals:
            signal.emit(emitList)

    @contextmanager
    def suppressSignals(self):
        oldSuppress = self._suppressSignals
        self._suppressSignals = True
        yield
        self._suppressSignals = oldSuppress

    def __reduce__(self):
        return ProjectData, (self.configPath, self.config)
