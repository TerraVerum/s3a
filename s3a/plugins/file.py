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

from utilitys import CompositionMixin, AtomicProcess, NestedProcess
from utilitys.params import *
from .. import models
from ..compio import ComponentIO, defaultIo
from ..compio.base import AnnotationExporter
from ..constants import (
  APP_STATE_DIR,
  PROJ_FILE_TYPE,
  PROJ_BASE_TEMPLATE,
  PRJ_ENUMS,
  PRJ_CONSTS as CNST,
  REQD_TBL_FIELDS)
from ..generalutils import hierarchicalUpdate, cvImsave_rgb
from ..graphicsutils import DropList
from ..logger import getAppLogger
from ..structures import FilePath, NChanImg


def absolutePath(p: Optional[Path]):
  """
  Bug in Path.resolve means it doesn't actually return an absolute path on Windows for a non-existent path
  While we're here, return None nicely without raising error
  """
  if p is None:
    return None
  return Path(os.path.abspath(p))

class FilePlugin(CompositionMixin, ParamEditorPlugin):
  name = 'File'
  win: models.s3abase.S3A

  def __init__(self, startupName: FilePath=None, startupCfg: dict=None):
    super().__init__()
    self.projData = self.exposes(ProjectData(startupName, startupCfg))
    self.autosaveTimer = QtCore.QTimer()

    self.registerFunc(self.save, btnOpts=CNST.TOOL_PROJ_SAVE)
    self.registerFunc(self.showProjImgs_gui, btnOpts=CNST.TOOL_PROJ_OPEN_IMG)
    self.menu.addSeparator()

    self.registerFunc(self.create_gui, btnOpts=CNST.TOOL_PROJ_CREATE)
    self.registerFunc(self.open_gui, btnOpts=CNST.TOOL_PROJ_OPEN)

    self.registerPopoutFuncs([self.updateProjectProperties, self.addImages_gui, self.addAnnotations_gui],
                             ['Update Project Properties', 'Add Images', 'Add Annotations'],
                             btnOpts=CNST.TOOL_PROJ_SETTINGS)

    self.registerFunc(lambda: self.win.setMainImg_gui(), btnOpts=CNST.TOOL_PROJ_ADD_IMG)
    self.registerFunc(lambda: self.win.openAnnotation_gui(), btnOpts=CNST.TOOL_PROJ_ADD_ANN)

    self.registerPopoutFuncs([self.startAutosave, self.stopAutosave], btnOpts=CNST.TOOL_AUTOSAVE)

    self._projImgMgr = ProjectImageManager()
    self._projImgMgr.sigImageSelected.connect(lambda imgFname: self.win.setMainImg(imgFname))
    self.projData.sigCfgLoaded.connect(lambda: self._projImgMgr.setRootDir(str(self.projData.imagesDir)))

    def onCfgLoad():
      self._updateProjLbl()
      if self.win:
        # Other arguments are consumed by app state editor
        state = self.win.appStateEditor
        if state.loading:
          hierarchicalUpdate(state.startupSettings, self.projData.startup)
        else:
          # Opening a project after s3a is already loaded
          state.loadParamValues(stateDict={}, **self.projData.startup)

    self.projData.sigCfgLoaded.connect(onCfgLoad)

    self.projNameLbl = QtWidgets.QLabel()

    useDefault = startupName is None and startupCfg is None
    self._createDefaultProj(useDefault)

  def _updateProjLbl(self):
    self.projNameLbl.setText(f'Project: {self.projData.cfgFname.name}')
    self.projNameLbl.setToolTip(str(self.projData.cfgFname))

  def _buildIoOpts(self):
    """
    Builds export option parameters for user interaction. Assumes export popout funcs have already been created
    """
    compIo = self.projData.compIo
    exportOptsParam = fns.getParamChild(self.toolsEditor.params, CNST.TOOL_PROJ_EXPORT.name, 'Export Options')
    # Use a wrapper to easily get hyperparams created
    for name, fn in inspect.getmembers(compIo, lambda el: isinstance(el, AnnotationExporter)):
      metadata = fn.optsMetadata()
      for child in metadata.values():
        if child['value'] is None or child['type'] not in PARAM_TYPES:
          # Not representable
          continue
        fns.getParamChild(exportOptsParam, chOpts=child)
    # Add catch-all that will be literally evaluated later
    exportOptsParam.addChild(dict(name='extra', type='text', value='', expanded=False))
    return exportOptsParam

  def attachWinRef(self, win: models.s3abase.S3ABase):
    super().attachWinRef(win)
    self.projData.compIo.tableData = win.sharedAttrs.tableData
    win.statBar.addPermanentWidget(self.projNameLbl)
    def handleExport(_dir):
      saveImg = win.srcImgFname
      ret = str(self.projData.cfgFname)
      if not saveImg:
        self.projData.startup.pop('image', None)
        return ret
      if saveImg and saveImg.parent == self.projData.imagesDir:
        saveImg = saveImg.name
      self.projData.startup['image'] = str(saveImg)
      self.save()
      return ret
    win.appStateEditor.addImportExportOpts('project', self.open, handleExport, 0)

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
      """Stores current autosave configuration at the specified location, if autosave is running"""
      if not self.autosaveTimer.isActive():
        return None

      cfg = {}
      for proc, params in self.toolsEditor.procToParamsMapping.items():
        if proc.name == 'Start Autosave':
          cfg = fns.paramValues(params).pop('Start Autosave', {})
          break

      saveName = str(savePath/'autosave.params')
      fns.saveToFile(cfg, saveName)
      return saveName

    win.appStateEditor.addImportExportOpts('autosave', receiveAutosave, exportAutosave)

    def startImg(imgName: str):
      imgName = Path(imgName)
      if not imgName.is_absolute():
        imgName = self.projData.imagesDir/imgName
      if not imgName.exists():
        return
      addedName = self.projData.addImage(imgName)
      self.win.setMainImg(addedName or imgName)
    win.appStateEditor.addImportExportOpts('image', startImg, lambda *args: None, 1)

    def exportWrapper(func):
      def wrapper(**kwargs):
        initial = {**self.exportOptsParam}
        # Fixup the special "extra" parameter
        extra = initial.pop('extra')
        if extra:
          try:
            newOpts = eval(f'dict({extra})', {}, {})
            initial.update(newOpts)
          except Exception as ex:
            warn(f'Could not parse extra arguments:\n{ex}')
        initial.update(kwargs)
        return func(**initial)
      return wrapper

    doctoredCur = AtomicProcess(exportWrapper(win.exportCurAnnotation),
                                'Current Annotation', outFname='', docFunc=win.exportCurAnnotation)
    doctoredAll = AtomicProcess(exportWrapper(self.projData.exportAnnotations),
                                docFunc=self.projData.exportAnnotations)
    self.registerPopoutFuncs([self.projData.exportProj, doctoredAll, doctoredCur],
                         ['Project', 'All Annotations', 'Current Annotation'], btnOpts=CNST.TOOL_PROJ_EXPORT)
    self._projImgMgr.hide()
    self._updateProjLbl()
    win.addTabbedDock(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._projImgMgr)
    self.exportOptsParam = self._buildIoOpts()

  def _createDefaultProj(self, setAsCur=True):
    defaultName = APP_STATE_DIR / PROJ_FILE_TYPE
    # Delete default prj on startup, if not current
    if not setAsCur:
      for prj in defaultName.glob(f'*.{PROJ_FILE_TYPE}'):
        prj.unlink()
    parent = self.projData if setAsCur else None
    if setAsCur or not defaultName.exists():
      # Otherwise, no action needed
      self.projData.create(name=defaultName, parent=parent)

  def open(self, cfgFname: FilePath=None, cfgDict: dict=None):
    if cfgFname is None and cfgDict is None:
      # Do nothing, happens when e.g. intentionally not loading a project
      return
    _, cfgDict = fns.resolveYamlDict(cfgFname, cfgDict)
    if (cfgFname is not None
        and (absolutePath(cfgFname) != self.projData.cfgFname
          or not pg.eq(cfgDict, self.projData.cfg)
        )
    ):
      self.win.setMainImg(None)
      self.projData.loadCfg(cfgFname, cfgDict, force=True)

  def open_gui(self):
    fname = fns.popupFilePicker(None, 'Select Project File', f'S3A Project (*.{PROJ_FILE_TYPE})')
    with pg.BusyCursor():
      self.open(fname)

  def save(self):
    self.win.saveCurAnnotation()
    self.projData.saveCfg()
    getAppLogger(__name__).attention('Saved project')

  @fns.dynamicDocstring(ioTypes=['<Unchanged>'] + list(defaultIo.roundTripTypes))
  def updateProjectProperties(self, tableConfig:FilePath=None, annotationFormat:str=None):
    """
    Updates the specified project properties, for each one that is provided

    :param tableConfig:
      pType: filepicker
    :param annotationFormat:
      helpText: "How to save annotations internally. Note that altering
      this value may alter the speed of saving and loading annotations."
      pType: list
      limits: {ioTypes}
    """
    if tableConfig is not None:
      tableConfig = Path(tableConfig)
      self.projData.tableData.loadCfg(tableConfig)
    if (annotationFormat is not None
        and annotationFormat in self.projData.compIo.importTypes
        and annotationFormat in self.projData.compIo.exportTypes):
      self.projData.cfg['annotation-format'] = annotationFormat


  @fns.dynamicDocstring(ioTypes=list(defaultIo.exportTypes))
  def startAutosave(self, interval=5, backupFolder='', baseName='autosave', annotationFormat='csv'):
    """
    Saves the current annotation set evert *interval* minutes

    :param interval:
      helpText: Interval in minutes between saves
      limits: [1, 1e9]
    :param backupFolder:
      helpText: "If provided, annotations are saved here sequentially afte reach *interval* minutes.
      Each output is named `[Parent Folder]/[base name]_[counter].[export type]`, where `counter`
      is the current save file number.'"
      pType: filepicker
      asFolder: True
    :param baseName: What to name the saved annotation file
    :param annotationFormat:
      helpText: File format for backups
      pType: list
      limits: {ioTypes}
    """
    self.autosaveTimer.stop()
    self.autosaveTimer = QtCore.QTimer()
    saveToBackup = len(str(backupFolder)) > 0
    if not saveToBackup:
      getAppLogger(__name__).attention(f'No backup folder selected. Will save to project without saving to backup directory.')
    backupFolder = Path(backupFolder)
    backupFolder.mkdir(exist_ok=True, parents=True)
    lastSavedDf = self.win.exportableDf.copy()
    # Qtimer expects ms, turn mins->s->ms
    # Figure out where to start the counter
    globExpr = lambda: backupFolder.glob(f'{baseName}*.{annotationFormat}')
    existingFiles = list(globExpr())
    if len(existingFiles) == 0:
      counter = 0
    else:
      counter = max(map(lambda fname: int(fname.stem.rsplit('_')[1]), existingFiles)) + 1

    def save_incrementCounter():
      nonlocal counter, lastSavedDf
      baseSaveNamePlusFolder = backupFolder / f'{baseName}_{counter}.{annotationFormat}'
      counter += 1
      curDf = self.win.exportableDf
      # A few things can go wrong during the comparison
      # noinspection PyBroadException
      try:
        isEqual = curDf.equals(lastSavedDf)
      except:
        isEqual = False
      if not isEqual:
        self.win.exportCurAnnotation(baseSaveNamePlusFolder)
        lastSavedDf = curDf.copy()

    if saveToBackup:
      self.autosaveTimer.timeout.connect(save_incrementCounter)
    self.autosaveTimer.timeout.connect(self.win.saveCurAnnotation)
    self.autosaveTimer.start(int(interval * 60 * 1000))
    getAppLogger(__name__).attention(f'Started autosaving')

  def stopAutosave(self):
    self.autosaveTimer.stop()
    getAppLogger(__name__).attention(f'Stopped autosaving')

  def showProjImgs_gui(self):
    if len(self.projData.images) == 0:
      getAppLogger(__name__).warning(
        'This project does not have any images yet. You can add them either in\n'
        '`Project Settings... > Add Image Files` or\n'
        '`File > Add New Image`.')
      return
    self._projImgMgr.show()
    self._projImgMgr.raise_()

  def addImages_gui(self):
    wiz = QtWidgets.QWizard()
    page = NewProjectWizard.createFilePage('Images', wiz)
    wiz.addPage(page)
    if wiz.exec_():
      for file in page.fileList.files:
        self.projData.addImageByPath(file)

  def addAnnotations_gui(self):
    wiz = QtWidgets.QWizard()
    page = NewProjectWizard.createFilePage('Annotations', wiz)
    wiz.addPage(page)
    if wiz.exec_():
      # Make sure if a file is added on top of the current image, recent changes don't get lost
      self.win.saveCurAnnotation()
      for file in page.fileList.files:
        self.projData.addAnnotationByPath(file)

  def create_gui(self):
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
    projName = settings['Name']
    prevTemplate = settings['Template Project']
    if prevTemplate is not None and len(prevTemplate) > 0:
      baseCfg = fns.attemptFileLoad(prevTemplate)
    else:
      baseCfg = {}
    projPath = Path(wiz.projSettings['Location'])/projName
    outPrj = self.projData.create(name=projPath, cfg=baseCfg)
    if prevTemplate:
      prevTemplateLoc = Path(prevTemplate).parent
      if settings['Keep Existing Images']:
        outPrj.addImageFolder(prevTemplateLoc/'images')
      if settings['Keep Existing Annotations']:
        outPrj.addImageFolder(prevTemplateLoc/'annotations')
    for image in images:
      outPrj.addImageByPath(image)
    for ann in annotations:
      outPrj.addAnnotation(ann)


    self.open(outPrj.cfgFname)


class NewProjectWizard(QtWidgets.QWizard):

  def __init__(self, project: FilePlugin, parent=None) -> None:
    super().__init__(parent)
    self.project = project
    self.fileLists : Dict[str, DropList] = {}

    # -----
    # PROJECT SETTINGS
    # -----
    page = QtWidgets.QWizardPage(self)
    page.setTitle('Project Settings')
    settings = [
      dict(name='Name', type='str', value='new-project'),
      dict(name='Location', type='filepicker', value='.', asFolder=True),
      dict(name='Template Project', type='filepicker', value=None,
           tip="Path to existing project config file. This will serve as a template for"
               " the newly created project, except for overridden settings"),
      dict(name='Keep Existing Images', type='bool', value=True,
           tip="Whether to keep images specified in the existing config"),
      dict(name='Keep Existing Annotations', type='bool', value=True,
           tip="Whether to keep annotations specified in the existing config"),
      # dict(name='Annotation Storage Format', value='pkl',
      #      tip="Project's internal representation of annotation files. Pickle (pkl) is the"
      #          " fastest for interaction, but is not as human readable. Alternatives (e.g. csv)"
      #          " are more human readable, but much slower when switching from image to image")
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
    for fType in ['Images', 'Annotations']:
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
    curLayout.addWidget(QtWidgets.QLabel(f'New project {name.lower()} are shown below. Use the buttons'
                                         ' or drag and drop to add files.'))
    flist = DropList(wizard)
    page.fileList = flist
    fileBtnLayout = QtWidgets.QHBoxLayout()
    curLayout.addWidget(flist)
    for title in f'Add Files', f'Add Folder':
      selectFolder = 'Folder' in title
      btn = QtWidgets.QPushButton(title, wizard)
      btn.clicked.connect(partial(getFileList_gui, wizard, flist, title, selectFolder))
      fileBtnLayout.addWidget(btn)
    curLayout.addLayout(fileBtnLayout)
    return page

def getFileList_gui(wizard, _flist: DropList, _title: str, _selectFolder=False):
  dlg = QtWidgets.QFileDialog()
  dlg.setModal(True)
  getFn = lambda *args, **kwargs: dlg.getOpenFileNames(*args, **kwargs, options=dlg.DontUseNativeDialog)[0]
  if _selectFolder:
    getFn = lambda *args, **kwargs: [dlg.getExistingDirectory(*args, **kwargs, options=dlg.DontUseNativeDialog)]
  files = getFn(wizard, _title)
  _flist.addItems(files)

class ProjectImageManager(QtWidgets.QDockWidget):
  sigImageSelected = QtCore.Signal(str) # Selected image name
  sigDeleteRequested = QtCore.Signal(str) # Image name to delete

  def __init__(self, parent=None, rootDir: FilePath=None):
    super().__init__(parent)
    self.setWindowTitle('Project Images')
    self.setObjectName('Project Images')
    self.setFeatures(self.DockWidgetMovable|self.DockWidgetFloatable|self.DockWidgetClosable)
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
    self.completer.setPlaceholderText('Type to filter')
    self.completer.textChanged.connect(self._filterFiles)

    self.fileViewer.activated.connect(self._emitFileName)

    layout.addWidget(self.completer)
    layout.addWidget(self.fileViewer)

  def _emitFileName(self, idx: QtCore.QModelIndex):
    self.sigImageSelected.emit(str(self.rootDir/idx.siblingAtColumn(0).data()))

  def _filterFiles(self, text):
    self.fileModel.setNameFilters([f'*{text}*'])

  def setRootDir(self, rootDir: FilePath):
    self.rootDir = Path(rootDir).absolute()
    rootDir = str(self.rootDir)
    rootIdx = self.fileModel.index(rootDir)
    if not rootIdx.isValid():
      raise ValueError(f'Invalid root directory specified: {rootDir}')
    self.fileModel.setRootPath(rootDir)
    self.fileViewer.setRootIndex(rootIdx)

class ProjectData(QtCore.QObject):
  sigCfgLoaded = QtCore.Signal()

  sigImagesAdded = QtCore.Signal(object)
  """List[Path] of added images"""
  sigImagesRemoved = QtCore.Signal(object)
  """List[Path] of removed images"""
  sigImagesMoved = QtCore.Signal(object)
  """
  List[(oldPath, NewPath)]
  Used mainly when images from outside the project are annotated. In that case, images are copied to inside the project,
  and this signal will be emitted.  
  """

  sigAnnotationsAdded = QtCore.Signal(object)
  """List[Path]"""
  sigAnnotationsRemoved = QtCore.Signal(object)
  """List[Path]"""


  def __init__(self, cfgFname: FilePath=None, cfgDict: dict=None):
    super().__init__()
    self.compIo = ComponentIO()
    self.templateName = PROJ_BASE_TEMPLATE
    self.cfg = fns.attemptFileLoad(self.templateName)
    self.cfgFname: Optional[Path] = None
    self.images: List[Path] = []
    self.baseImgDirs: Set[Path] = set()
    self.imgToAnnMapping: Dict[Path, Path] = {}
    """Records annotations belonging to each image"""
    self.spawnedPlugins: List[ParamEditorPlugin] = []
    """Plugin instances stored separately from plugin-cfg to maintain serializability of self.cfg"""

    self._suppressSignals = False
    """If this is *True*, no signals will be emitted """
    self.watcher = QtCore.QFileSystemWatcher()
    self.watcher.directoryChanged.connect(self._handleDirChange)

    if cfgFname is not None or cfgDict is not None:
      self.loadCfg(cfgFname, cfgDict)

  def _handleDirChange(self):
    imgs = list(self.imagesDir.glob('*.*'))
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
      if img.parent == self.imagesDir and img not in imgs:
        delIdxs.append(ii)
        delImgs.append(img)
    for idx in delIdxs:
      del self.images[idx]
    self.sigImagesRemoved.emit(delImgs)

    anns = list(self.annotationsDir.glob(f'*.{self.cfg["annotation-format"]}'))
    # Images already in the project will be ignored on add
    # Assume new annotations here are already formatted properly
    for ann in anns:
      if ann not in self.imgToAnnMapping.values():
        self.addFormattedAnnotation(ann)
    # Convert to list to avoid "dictionary changed size on iteration" error
    for ann in list(self.imgToAnnMapping.values()):
      if ann not in anns:
        self.removeAnnotation(ann)

  @property
  def location(self):
      return self.cfgFname.parent
  @property
  def imagesDir(self):
      return self.location/'images'
  @property
  def annotationsDir(self):
      return self.location/'annotations'
  @property
  def startup(self):
    return self.cfg['startup']
  @property
  def pluginCfg(self) -> Dict[str, str]:
      return self.cfg['plugin-cfg']

  @property
  def tableData(self):
      return self.compIo.tableData

  def clearImgs_anns(self):
    oldImgs = self.images.copy()
    for lst in self.images, self.imgToAnnMapping, self.baseImgDirs:
      lst.clear()
    self._maybeEmit(self.sigImagesRemoved, oldImgs)

  def loadCfg(self, cfgFname: FilePath, cfgDict: dict = None, force=False):
    """
    Loads the specified project configuration by name (if a file) or dict (if programmatically created)
    :param cfgFname: Name of file to open. If `cfgDict` is provided instead, it will be saved here.
    :param cfgDict: If provided, this config is used and saved to `cfgFname` instead of using the file.
    :param force: If *True*, the new config will be loaded even if it is the same name as the
      current config
    """
    _, baseCfgDict = fns.resolveYamlDict(self.templateName)
    cfgFname, cfgDict = fns.resolveYamlDict(cfgFname, cfgDict)
    cfgFname = absolutePath(cfgFname)
    if not force and self.cfgFname == cfgFname:
      return None

    hierarchicalUpdate(baseCfgDict, cfgDict, uniqueListElements=True)

    loadPrjPlugins = baseCfgDict.get('plugin-cfg', {})
    newPlugins = {k: v for (k, v) in loadPrjPlugins.items() if k not in self.pluginCfg}
    removedPlugins = set(self.pluginCfg).difference(loadPrjPlugins)
    if removedPlugins:
      raise ValueError(f'The previous project loaded custom plugins, which cannot easily'
                       f' be removed. To load a new project without plugin(s) {", ".join(removedPlugins)}, close and'
                       f' re-open S3A with the new project instance instead. Alternatively, add these missing plugins'
                       f' to the project you wish to add.')
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
      warn(f'Some project plugins were specified, but could not be found:\n'
                    f'{warnPlgs}', UserWarning)

    self.cfgFname = cfgFname
    cfg = self.cfg = baseCfgDict

    self.annotationsDir.mkdir(exist_ok=True)
    self.imagesDir.mkdir(exist_ok=True)
    self.clearImgs_anns()

    tableInfo = cfg.get('table-cfg', None)
    if isinstance(tableInfo, str):
      tableDict = None
      tableName = tableInfo
    else:
      if tableInfo is None:
        tableInfo = {}
      tableDict = tableInfo
      tableName = cfgFname
    tableName = Path(tableName)
    if not tableName.is_absolute():
      tableName = self.location/tableName
    self.tableData.loadCfg(tableName, tableDict, force=True)

    allAddedImages = []
    with self.suppressSignals():
      allAddedImages.extend(self.addImageFolder(self.imagesDir, copyToProject=False))
      # Leave out for now due to the large number of problems with lacking idempotence
      # allAddedImages.extend(self._addConfigImages())
    self._maybeEmit(self.sigImagesAdded, allAddedImages)

    with self.suppressSignals():
      for file in self.annotationsDir.glob(f'*.{self.cfg["annotation-format"]}'):
        self.addFormattedAnnotation(file)
      # Leave out for now due to the large number of problems with lacking idempotence
      # self._addConfigAnnotations()
      
    self._maybeEmit(self.sigAnnotationsAdded, list(self.imgToAnnMapping.values()))

    self.sigCfgLoaded.emit()
    dirs = self.watcher.directories()
    if dirs:
      self.watcher.removePaths(dirs)
    self.watcher.addPaths([str(self.imagesDir), str(self.annotationsDir)])
    for ioType in PRJ_ENUMS.IO_EXPORT, PRJ_ENUMS.IO_IMPORT:
      self.compIo.updateOpts(ioType, srcDir=self.imagesDir)
    return self.cfgFname

  @classmethod
  def create(cls, *, name: FilePath= f'./{PROJ_FILE_TYPE}', cfg: dict=None, parent: ProjectData=None):
    """
    Creates a new project with the specified settings in the specified directory.
    :param name:
      helpText: Project Name. The parent directory of this name indicates the directory in which to create the project
      pType: filepicker
    :param cfg: see `ProjectData.loadCfg` for information
    :param parent: Associated ProjectData instance for a non-classmethod version of this function
    """
    name = Path(name)
    name = name/f'{name.name}.{PROJ_FILE_TYPE}'
    location = name.parent
    location.mkdir(exist_ok=True, parents=True)
    if parent is None:
      parent = cls()

    if not name.exists() and cfg is None:
      cfg = {}
    parent.loadCfg(name, cfg)

    parent.saveCfg()
    return parent

  def saveCfg(self, copyMissingItems=False):
    """
    Saves the config file, optionally copying missing items to the project location as well. "Missing items" are
    images and annotations in base folders / existing in the project config but not in the actual project directory
    """
    if copyMissingItems:
      for image in  self._findMissingImages():
        self.addImageByPath(image)
      for annotation in self._findMissingAnnotations():
        self.addAnnotationByPath(annotation)

    tblName = Path(self.tableData.cfgFname).absolute()
    if tblName != self.cfgFname:
      if tblName.parent == self.location:
        tblName = tblName.name
      self.cfg['table-cfg'] = str(tblName)
    fns.saveToFile(self.cfg, self.cfgFname)

  def _findMissingAnnotations(self):
    annDir = self.annotationsDir
    missingAnns = []
    for img, ann in self.imgToAnnMapping.items():
      if ann.parent != annDir:
        missingAnns.append(str(ann))
    return missingAnns

  def _findMissingImages(self):
    location = self.location
    imgDir = self.imagesDir
    strImgNames = []
    for folder in self.baseImgDirs:
      if location in folder.parents:
        folder = folder.relative_to(location)
      strImgNames.append(str(folder))
    for img in self.images:
      if img.parent == imgDir or img.parent in self.baseImgDirs:
        # This image is already accounted for in the base directories
        continue
      strImgNames.append(str(absolutePath(img)))
    return strImgNames

  def _addConfigImages(self):
    addedImages = []
    for image in self.cfg.get('images', []):
      if isinstance(image, dict):
        image.setdefault('copyToProject', True)
        renamed = self.addImage(**image)
        if renamed is not None:
          addedImages.append(renamed)
      else:
        addedImages.extend(self.addImageByPath(image))
    return addedImages

  def _addConfigAnnotations(self):
    for annotation in set(self.cfg.get('annotations', [])) - {self.annotationsDir}:
      if isinstance(annotation, dict):
        self.addAnnotation(**annotation)
      else:
        self.addAnnotationByPath(annotation)


  def addImageByPath(self, name: FilePath, copyToProject=True):
    """
    Determines whether to add as a folder or file based on filepath type. Since adding a folder returns a list of
    images and adding a single image returns a name or None, this function unifies the return signature by always
    returning a list. If the path is a single image and not a folder, and the return value is *None*, this function
    will return an empty list instead.
    """
    image = Path(name)
    if not image.is_absolute():
      image = self.location/image
    if not image.exists():
      getAppLogger(__name__).attention(f'Provided image path does not exist: {image}\nNo action performed.')
      return []
    if image.is_dir():
      ret = self.addImageFolder(image, copyToProject)
    else:
      ret = self.addImage(name, copyToProject=copyToProject)
      ret = [] if ret is None else [ret]
    return ret

  def addImage(self, name: FilePath, data: NChanImg=None, copyToProject=True, allowOverwrite=False) -> Optional[FilePath]:
    """Returns None if an image with the same name already exists in the project, else the new full filepath"""
    fullName = Path(name)
    if not fullName.is_absolute():
      fullName = self.imagesDir / fullName
    if copyToProject or data is not None:
      fullName = self._copyImgToProj(fullName, data, allowOverwrite)
    if fullName.name in [i.name for i in self.images]:
      # Indicate the image was already present to calling scope
      return None
    self.images.append(fullName)
    self._maybeEmit(self.sigImagesAdded, [fullName])
    return fullName
    # TODO: Create less hazardous undo operation
    # yield name
    # self.removeImage(name)

  def changeImgPath(self, oldName: Path, newName: Path=None):
    """
    Changes the filepath associated with a project image. Note that this doesn't do anything to the path
    of either oldName or newName (i.e. the actual image isn't moved/copied/etc.), it just changes the association
    within the project from the old image to the new image.
      * If newName is None, the association is deleted. Don't do this for images inside the project directory.
      * if newName already exists in the current project (i.e. matches an image already added to the project),
        oldName is deleted from the project associations
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
    if folder in self.baseImgDirs:
      return []
    # Need to keep track of actually added images instead of using all globbed images. If an added image already
    # existed in the project, it won't be added. Also, if the images are copied into the project, the paths
    # will change.
    addedImgs = []
    with self.suppressSignals():
      for img in folder.glob('*.*'):
        finalName = self.addImage(img, copyToProject=copyToProject)
        if finalName is not None:
          addedImgs.append(finalName)
    self._maybeEmit(self.sigImagesAdded, addedImgs)

    return addedImgs

  def addAnnotationByPath(self, name: FilePath):
    """Determines whether to add as a folder or file based on filepath type"""
    name = Path(name)
    if not name.is_absolute():
      name = self.location/name
    if not name.exists():
      getAppLogger(__name__).attention(f'Provided annotation path does not exist: {name}\nNo action performed.')
      return
    if name.is_dir():
      self.addAnnotationFolder(name)
    else:
      self.addAnnotation(name)

  def addAnnotationFolder(self, folder: FilePath):
    folder = absolutePath(folder)
    for file in folder.glob('*.*'):
      self.addAnnotation(file)

  def addImage_gui(self, copyToProject=True):
    fileFilter = "Image Files (*.png *.tif *.jpg *.jpeg *.bmp *.jfif);;All files(*.*)"
    fname = fns.popupFilePicker(None, 'Add Image to Project', fileFilter)
    if fname is not None:
      self.addImage(fname, copyToProject=copyToProject)

  def removeImage(self, imgName: FilePath):
    imgName = absolutePath(imgName)
    if imgName not in self.images:
      return
    self.images.remove(imgName)
    # Remove copied annotations for this image
    for ann in self.annotationsDir.glob(f'{imgName.stem}.*'):
      ann.unlink()
    self.imgToAnnMapping.pop(imgName, None)
    self._maybeEmit(self.sigImagesRemoved, [imgName])
    if imgName.parent == self.imagesDir:
      imgName.unlink()
    # yield
    # if imgName.parent == self.imagesDir:
    #   # TODO: Cache removed images in a temp dir, then move them to that temp dir instead of unlinking
    #   #  on delete. This will make 'remove' undoable
    #   raise IOError('Can only undo undo image removal when the image was outside the project.'
    #                    f' Image {imgName.name} was either annotated or directly placed in the project images'
    #                    f' directory, and was deleted during removal. To re-add, do so from the original image'
    #                    f' location outside the project directory.')
    # self.addImage(imgName)

  def removeAnnotation(self, annName: FilePath):
    annName = absolutePath(annName)
    # Since no mapping exists of all annotations, loop the long way until the file is found
    for key, ann in list(self.imgToAnnMapping.items()):
      if annName == ann:
        del self.imgToAnnMapping[key]
        ann.unlink(missing_ok=True)
        self._maybeEmit(self.sigAnnotationsRemoved, [ann])
        break

  def addAnnotation(self, name: FilePath=None, data: pd.DataFrame=None, image: FilePath=None, overwriteOld=False):
    # Housekeeping for default arguments
    if name is None and data is None:
      raise IOError('`name` and `data` cannot both be `None`')
    elif name in self.imgToAnnMapping.values() and not overwriteOld:
      # Already present, shouldn't be added
      return
    if data is None:
      data = self.compIo.importByFileType(name)
    if image is None:
      # If no explicit matching to an image is provided, try to determine based on annotation name
      xpondingImgs = np.unique(data[REQD_TBL_FIELDS.IMG_FILE].to_numpy())
      # Break into annotaitons by iamge
      for img in xpondingImgs:
        # Copy to avoid pandas warning
        self.addAnnotation(name, data[data[REQD_TBL_FIELDS.IMG_FILE] == img].copy(), img)
      return
    image = self.getFullImgName(Path(image))
    # Force provided annotations to now belong to this image
    data.loc[:, REQD_TBL_FIELDS.IMG_FILE] = image.name
    # Since only one annotation file can exist per image, concatenate this with any existing files for the same image
    # if needed
    if image.parent != self.imagesDir:
      # None result means already exists in project prior to copying, which is OK in this context
      image = self.addImage(image) or self.imagesDir/image.name
    annForImg = self.imgToAnnMapping.get(image, None)
    oldAnns = []
    if annForImg is not None and not overwriteOld:
      oldAnns.append(self.compIo.importByFileType(annForImg))
    combinedAnns = oldAnns + [data]
    outAnn = pd.concat(combinedAnns, ignore_index=True)
    outAnn[REQD_TBL_FIELDS.INST_ID] = outAnn.index
    outFmt = f".{self.cfg['annotation-format']}"
    outName = self.annotationsDir / f'{image.name}{outFmt}'
    # If no annotations exist, this is the same as deleting the old annotations since there's nothing to save
    if len(outAnn):
      self.compIo.exportByFileType(outAnn, outName, verifyIntegrity=False, readonly=False)
      self.imgToAnnMapping[image] = outName
      self._maybeEmit(self.sigAnnotationsAdded, [outName])
    elif outName.exists():
      self.removeAnnotation(outName)

  def addFormattedAnnotation(self, file: FilePath, overwriteOld=False):
    """
    Adds an annotation file that is already formatted in the following ways:
      * The right source image file column (i.e. REQD_TBL_FIELDS.IMG_FILE set to the image name
      * The file stem already matches a project image (not remote, i.e. an image in the `images` directory)
      * The annotations correspond to exactly one image
    """
    image = self.getFullImgName(file.stem, thorough=False)
    if file.parent != self.annotationsDir:
      if not overwriteOld and (self.annotationsDir/file.name).exists():
        raise IOError(f'Cannot add annotation {file} since a corresponding annotation with that name already'
                      f' exists and `overwriteOld` was False')
      newName = self.annotationsDir/file.name
      shutil.copy2(file, newName)
      file = newName
    self.imgToAnnMapping[image] = file

  def _copyImgToProj(self, name: Path, data: NChanImg=None, overwrite=False):
    newName = self.imagesDir/name.name
    if newName.exists() and (not overwrite or newName == name):
      # Already in the project, no need to copy
      return newName
    if name.exists() and data is None:
      shutil.copy(name, newName)
    elif data is not None:
      # Programmatically created or not from a local file
      # noinspection PyTypeChecker
      cvImsave_rgb(newName, data)
    else:
      raise IOError(f'No image data associated with {name.name}. Either the file does not exist or no'
                       f' image information was provided.')
    if name in self.images:
      self.changeImgPath(name, newName)
      self._maybeEmit(self.sigImagesMoved, [(name, newName)])
    return newName

  def getFullImgName(self, name: FilePath, thorough=True):
    """
    From an absolute or relative image name, attempts to find the absolute path it corresponds
    to based on current project images. A match is located in the following order:
      - If the image path is already absolute, it is resolved, checked for existence, and returned
      - Solitary project images are searched to see if they end with the specified relative path
      - All base image directories are checked to see if they contain this subpath

    :param name: Image name
    :param thorough: If `False`, as soon as a match is found the function returns. Otherwise,
      all solitary paths and images will be checked to ensure there is exactly one matching
      image for the name provided.
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
      compare = lambda strName, selfImageName: str(selfImageName).endswith(strName)
    for img in self.images:
      if compare(strName, img):
        if not thorough:
          return img
        candidates.add(img)

    for parent in self.baseImgDirs:
      curName = parent/name
      if curName.exists():
        if not thorough:
          return curName
        candidates.add(curName)

    numCandidates = len(candidates)
    if numCandidates != 1:
      msg = f'Exactly one corresponding image file must exist for a given annotation. However,' \
            f' {numCandidates} candidate images were found for image {name.name}'
      if numCandidates == 0:
        msg += '.'
      else:
        msg += f':\n{", ".join([c.name for c in candidates])}'
      raise IOError(msg)
    return candidates.pop()

  def exportProj(self, outputFolder: FilePath= 's3a-export'):
    """
    Exports the entire project, making a copy of it at the destination directory
    :param outputFolder:
      helpText: Where to place the exported project
      pType: filepicker
      asFolder: True
    """
    shutil.copytree(self.location, outputFolder)
    getAppLogger(__name__).info('Exported project')

  @fns.dynamicDocstring(fileTypes=list(defaultIo.exportTypes))
  def exportAnnotations(self, outputFolder:FilePath= 's3a-export',
                        annotationFormat='csv',
                        combine=False,
                        includeImages=True,
                        **exportOpts):
    """
    Exports project annotations, optionally including their source images
    :param outputFolder:
      helpText: Folder for exported annotations
      pType: filepicker
      asFolder: True
    :param annotationFormat:
      helpText: "Annotation file type. E.g. if 'csv', annotations will be saved as csv files. Available
      file types are:
      {fileTypes}"
      pType: list
      limits: {fileTypes}
    :param combine: If `True`, all annotation files will be combined into one exported file with name `annotations.<format>`
    :param includeImages: If `True`, the corresponding image for each annotation will also be exported into an `images`
      folder
    :param exportOpts: Additional options passed to the exporting function
    """
    self.saveCfg()
    outputFolder = Path(outputFolder)

    if outputFolder.resolve() == self.location and not combine:
      return

    outputFolder.mkdir(exist_ok=True)
    if includeImages:
      outImgDir = outputFolder / 'images'
      outImgDir.mkdir(exist_ok=True)
      for img in self.imagesDir.glob('*.*'):
        if self.imgToAnnMapping.get(img, None) is not None:
          shutil.copy(img, outImgDir)

    existingAnnFiles = [f for f in self.imgToAnnMapping.values() if f is not None]
    if combine:
      outAnnsPath = outputFolder / f'annotations.{annotationFormat}'
      outAnn = pd.concat(map(self.compIo.importByFileType, existingAnnFiles), ignore_index=True)
      outAnn[REQD_TBL_FIELDS.INST_ID] = outAnn.index
      self.compIo.exportByFileType(outAnn, outAnnsPath, **exportOpts)
    else:
      outAnnsPath = outputFolder / 'annotations'
      outAnnsPath.mkdir(exist_ok=True)
      if self.cfg['annotation-format'] == annotationFormat:
        shutil.copytree(self.annotationsDir, outAnnsPath, dirs_exist_ok=True)
      else:
        for annFile in existingAnnFiles:
          self.compIo.convert(annFile, outAnnsPath/f'{annFile.stem}.{annotationFormat}',
                              importArgs=exportOpts, exportArgs=exportOpts)
    getAppLogger(__name__).attention(f'Exported project annotations to {os.path.join(outAnnsPath.parent.name, outAnnsPath.name)}')

  def _maybeEmit(self, signal: QtCore.Signal, emitList: Sequence[Union[Path, Tuple[Path, Path]]]):
    if not self._suppressSignals:
      signal.emit(emitList)

  @contextmanager
  def suppressSignals(self):
    oldSuppress = self._suppressSignals
    self._suppressSignals = True
    yield
    self._suppressSignals = oldSuppress

  def __reduce__(self):
    return ProjectData, (self.cfgFname, self.cfg)
