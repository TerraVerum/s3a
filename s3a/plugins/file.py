from __future__ import annotations

import inspect
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

from s3a import PRJ_CONSTS as CNST, models, REQD_TBL_FIELDS
from s3a._io import ComponentIO
from s3a.generalutils import hierarchicalUpdate, cvImsave_rgb
from s3a.graphicsutils import DropList, ThumbnailViewer
from s3a.structures import FilePath, NChanImg
from utilitys import CompositionMixin, AtomicProcess, NestedProcess
from utilitys import fns
from utilitys.fns import warnLater
from utilitys.params import *
from ..constants import APP_STATE_DIR, PROJ_FILE_TYPE, PROJ_BASE_TEMPLATE


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
    self._imgThumbnails = self._projImgMgr.thumbnails
    self.nameToFullPathMapping = self._imgThumbnails.nameToFullPathMapping

    def onAdd(imList):
      for im in imList:
        self._imgThumbnails.addThumbnail(im, force=True)
    def onMove(imList):
      for oldName, newName in imList:
        self.nameToFullPathMapping[oldName.name] = newName
    def onDel(imList):
      if len(imList) == 0:
        # or QtWidgets.QMessageBox.question(
        # self.win, 'Confirm Delete', f'Are you sure you want to delete the following images?\n'
        #                             f'{[i.name for i in imList]})') != QtWidgets.QMessageBox.Yes:
        return
      delCurrent = False

      for name in imList:
        if name == self.win.srcImgFname and self.win.srcImgFname.exists():
          delCurrent = True
          continue
        self._imgThumbnails.removeThumbnail(name.name)
      if delCurrent:
        raise IOError(f'Cannot delete {self.win.srcImgFname.name} since it is currently'
             f' being annotated. Change the image and try again.')
    self.projData.sigImagesAdded.connect(onAdd)
    self.projData.sigImagesMoved.connect(onMove)
    self.projData.sigImagesRemoved.connect(onDel)

    self._projImgMgr.sigImageSelected.connect(lambda imgFname: self.win.setMainImg(imgFname))
    def handleDelete(delImgs):
      for img in delImgs:
        self.projData.removeImage(img)
    self._projImgMgr.sigDeleteRequested.connect(handleDelete)

    def onCfgLoad():
      for im in self._imgThumbnails.nameToFullPathMapping.values():
        if im not in self.projData.images:
          self._imgThumbnails.removeThumbnail(im.name)
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

  def _buildIoOpts(self):
    """
    Builds export option parameters for user interaction. Assumes export popout funcs have already been created
    """
    compIo = self.projData.compIo
    exportOptsParam = fns.getParamChild(self.toolsEditor.params, CNST.TOOL_PROJ_EXPORT.name, 'Export Options')
    # Use a wrapper to easily get hyperparams created
    wrapper = NestedProcWrapper(NestedProcess('Export Options'), exportOptsParam, nestHyperparams=False)
    for name, fn in inspect.getmembers(type(compIo), inspect.isfunction):
      if not name.startswith('export'):
        continue
      # Make params to expose options per export type
      atomic = AtomicProcess(fn)
      for key in reversed(atomic.input.hyperParamKeys):
        # Reverse to delete without issue
        # Remove none values that can't be edited by the user
        # TODO: This is awful! But it works... Get a list of all generally editable parameters, except for known internal
        #   ones
        if atomic.input[key] is None or key in ('outDir', 'returnLblMapping', 'offset'):
          atomic.input.hyperParamKeys.remove(key)
      if atomic.input.hyperParamKeys:
        wrapper.addStage(atomic)
    return wrapper.parentParam

  def attachWinRef(self, win: models.s3abase.S3ABase):
    super().attachWinRef(win)
    self.projData.compIo.tableData = win.sharedAttrs.tableData
    win.statBar.addWidget(self.projNameLbl)
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

    def startImg(imgName: str):
      imgName = Path(imgName)
      if not imgName.is_absolute():
        imgName = self.projData.imagesDir/imgName
      if not imgName.exists():
        return
      name = imgName.name
      self.projData.addImage(imgName)
      fullName = self.nameToFullPathMapping[name]
      self.win.setMainImg(fullName)
    win.appStateEditor.addImportExportOpts('image', startImg, lambda *args: None, 1)

    def exportWrapper(func):
      def wrapper(**kwargs):
        initial = {**self.exportOptsParam}
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
    win.addTabbedDock(QtCore.Qt.RightDockWidgetArea, self._projImgMgr)
    self.exportOptsParam = self._buildIoOpts()

  def _createDefaultProj(self, setAsCur=True):
    defaultName = APP_STATE_DIR / PROJ_FILE_TYPE
    # Delete default prj on startup
    for prj in defaultName.glob(f'*.{PROJ_FILE_TYPE}'):
      prj.unlink()
    parent = self.projData if setAsCur else None
    self.projData.create(name=defaultName, parent=parent)

  def open(self, cfgFname: FilePath=None, cfgDict: dict=None):
    _, cfgDict = fns.resolveYamlDict(cfgFname, cfgDict)
    if (cfgFname is not None
        and (Path(cfgFname).resolve() != self.projData.cfgFname
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

  @fns.dynamicDocstring(ioTypes=['<Unchanged>'] + list(ComponentIO.roundTripTypes))
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


  @fns.dynamicDocstring(ioTypes=list(ComponentIO.exportTypes))
  def startAutosave(self, interval=5, backupFolder='', baseName='autosave', exportType='pkl'):
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
    :param exportType:
      helpText: File format for backups
      pType: list
      limits: {ioTypes}
    """
    self.autosaveTimer = QtCore.QTimer()
    self.autosaveTimer.start(int(interval * 60 * 1000))
    self.autosaveTimer.timeout.connect(self.win.saveCurAnnotation)
    if len(str(backupFolder)) == 0:
      return
    backupFolder = Path(backupFolder)
    backupFolder.mkdir(exist_ok=True, parents=True)
    lastSavedDf = self.win.exportableDf.copy()
    # Qtimer expects ms, turn mins->s->ms
    # Figure out where to start the counter
    globExpr = lambda: backupFolder.glob(f'{baseName}*.{exportType}')
    existingFiles = list(globExpr())
    if len(existingFiles) == 0:
      counter = 0
    else:
      counter = max(map(lambda fname: int(fname.stem.rsplit('_')[1]), existingFiles)) + 1

    def save_incrementCounter():
      nonlocal counter, lastSavedDf
      baseSaveNamePlusFolder = backupFolder / f'{baseName}_{counter}.{exportType}'
      counter += 1
      curDf = self.win.exportableDf
      if not curDf.equals(lastSavedDf):
        self.win.exportCurAnnotation(baseSaveNamePlusFolder)
        lastSavedDf = curDf.copy()

    self.autosaveTimer.timeout.connect(save_incrementCounter)

  def stopAutosave(self):
    self.autosaveTimer.stop()

  def showProjImgs_gui(self):
    if len(self.projData.images) == 0:
      warn('This project does not have any images yet. You can add them either in\n'
           '<code>File > Project Settings... > Add Image Files</code> or\n'
           '<code>File > Add New Image</code>.', UserWarning)
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
      if not settings['Keep Existing Images'] or 'images' not in baseCfg:
        baseCfg['images'] = []
      if not settings['Keep Existing Annotations'] or 'annotations' not in baseCfg:
        baseCfg['annotations'] = []
    else:
      baseCfg = {'images': [], 'annotations': []}
    baseCfg['images'].extend(images)
    baseCfg['annotations'].extend(annotations)
    projPath = Path(wiz.projSettings['Location'])/projName
    outPrj = self.projData.create(name=projPath, cfg=baseCfg)
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
      btn.clicked.connect(partial(getFileList, wizard, flist, title, selectFolder))
      fileBtnLayout.addWidget(btn)
    curLayout.addLayout(fileBtnLayout)
    return page

def getFileList(wizard, _flist: DropList, _title: str, _selectFolder=False):
  dlg = QtWidgets.QFileDialog()
  dlg.setModal(True)
  getFn = lambda *args, **kwargs: dlg.getOpenFileNames(*args, **kwargs, options=dlg.DontUseNativeDialog)[0]
  if _selectFolder:
    getFn = lambda *args, **kwargs: [dlg.getExistingDirectory(*args, **kwargs, options=dlg.DontUseNativeDialog)]
  files = getFn(wizard, _title)
  _flist.addItems(files)

class ProjectImageManager(QtWidgets.QDockWidget):
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle('Project Images')
    self.setObjectName('Project Images')
    self.setFeatures(self.DockWidgetMovable|self.DockWidgetFloatable|self.DockWidgetClosable)
    wid = QtWidgets.QWidget()
    self.setWidget(wid)

    layout = QtWidgets.QVBoxLayout()
    wid.setLayout(layout)

    self.thumbnails = ThumbnailViewer()
    self.completer = QtWidgets.QLineEdit()
    self.completer.setPlaceholderText('Type to filter')
    self.completer.textChanged.connect(self._filterThumbnails)

    self.sigImageSelected = self.thumbnails.sigImageSelected
    self.sigDeleteRequested = self.thumbnails.sigDeleteRequested

    layout.addWidget(self.completer)
    layout.addWidget(self.thumbnails)

  def _filterThumbnails(self, text):
    for ii in range(self.thumbnails.model().rowCount()):
      item = self.thumbnails.item(ii)
      if text in item.text():
        item.setHidden(False)
      else:
        item.setHidden(True)

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


  def __init__(self, cfgFname: FilePath=None, cfgDict: dict=None, io: ComponentIO=None):
    super().__init__()
    if io is None:
      io = ComponentIO()
    self.compIo = io
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
    for ann in anns:
      if ann not in self.imgToAnnMapping.values():
        self.addAnnotation(ann)
    for ann in self.imgToAnnMapping.values():
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
    cfgFname = cfgFname.resolve()
    if not force and self.cfgFname == cfgFname:
      return None

    hierarchicalUpdate(baseCfgDict, cfgDict)

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
      fns.warnLater(f'Some project plugins were specified, but could not be found:\n'
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
      allAddedImages.extend(self.addImageFolder(self.imagesDir))
      for image in self.cfg['images']:
        if isinstance(image, dict):
          image.setdefault('copyToProj', False)
          renamed = self.addImage(**image)
          if renamed is not None:
            allAddedImages.append(renamed)
        else:
          allAddedImages.extend(self.addImageByPath(image, False))
    self._maybeEmit(self.sigImagesAdded, allAddedImages)

    with self.suppressSignals():
      self.addAnnotationFolder(self.annotationsDir)
      for annotation in set(self.cfg['annotations']) - {self.annotationsDir}:
        if isinstance(annotation, dict):
          self.addAnnotation(**annotation)
        else:
          self.addAnnotationByPath(annotation)
    self._maybeEmit(self.sigAnnotationsAdded, list(self.imgToAnnMapping.values()))

    self.sigCfgLoaded.emit()
    dirs = self.watcher.directories()
    if dirs:
      self.watcher.removePaths(dirs)
    self.watcher.addPaths([str(self.imagesDir), str(self.annotationsDir)])
    self.compIo.importOpts['imgDir'] = self.imagesDir
    self.compIo.exportOpts['imgDir'] = self.imagesDir
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

    tdName = Path(parent.tableData.cfgFname)
    if tdName.resolve() != parent.cfgFname:
      tdName = tdName.name
      fns.saveToFile(parent.tableData.cfg, location / tdName, True)
      parent.tableData.cfgFname = tdName
      parent.cfg['table-cfg'] = tdName

    parent.saveCfg()
    return parent

  def saveCfg(self):
    location = self.location
    annDir = self.annotationsDir
    strImgNames = []
    for folder in self.baseImgDirs:
      if location in folder.parents:
        folder = folder.relative_to(location)
      strImgNames.append(str(folder))
    for img in self.images:
      if img.parent in self.baseImgDirs:
        # This image is already accounted for in the base directories
        continue
      strImgNames.append(str(img.resolve()))

    offendingAnns = []
    for img, ann in self.imgToAnnMapping.items():
      if ann.parent != annDir:
        offendingAnns.append(str(ann))
        self.addAnnotation(ann)
    if len(offendingAnns) > 0:
      warnLater('Encountered annotation(s) in project config, but not officially added. '
            'Adding them now.'
           '  Offending files:\n'
           + ',\n'.join(offendingAnns), UserWarning)
    self.cfg['images'] = strImgNames
    # 'Ann' folder is always added on startup so no need to record it here. However,
    # if it is shown explicitly the user is aware.
    self.cfg['annotations'] = [self.annotationsDir.name]
    tblName = Path(self.tableData.cfgFname).absolute()
    if tblName !=  self.cfgFname:
      if tblName.parent == self.location:
        tblName = tblName.name
      self.cfg['table-cfg'] = str(tblName)
    fns.saveToFile(self.cfg, self.cfgFname)

  def addImageByPath(self, name: FilePath, copyToProj=False):
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
      warnLater(f'Provided image path does not exist: {image}\nNo action performed.',
            UserWarning)
      return []
    if image.is_dir():
      ret = self.addImageFolder(image, copyToProj)
    else:
      ret = self.addImage(name, copyToProj=copyToProj)
      ret = [] if ret is None else [ret]
    return ret

  def addImage(self, name: FilePath, data: NChanImg=None, copyToProj=False, allowOverwrite=False) -> Optional[FilePath]:
    fullName = Path(name)
    if not fullName.is_absolute():
      fullName = self.imagesDir / fullName
    if copyToProj or data is not None:
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
    oldIdx = self.images.index(oldName)
    if newName is None or newName in self.images:
      del self.images[oldIdx]
    else:
      self.images[oldIdx] = newName
    self._maybeEmit(self.sigImagesMoved, [(oldName, newName)])

  def addImageFolder(self, folder: FilePath, copyToProj=False):
    folder = Path(folder).resolve()
    if folder in self.baseImgDirs:
      return []
    if copyToProj:
      newFolder = self.imagesDir/folder.name
      shutil.copytree(folder, newFolder)
      folder = newFolder
      copyToProj = False
    self.baseImgDirs.add(folder)
    # Need to keep track of actually added images instead of using all globbed images. If an added image already
    # existed in the project, it won't be added. Also, if the images are copied into the project, the paths
    # will change.
    addedImgs = []
    with self.suppressSignals():
      for img in folder.glob('*.*'):
        finalName = self.addImage(img, copyToProj=copyToProj)
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
      warnLater(f'Provided annotation path does not exist: {name}\nNo action '
                f'performed.', UserWarning)
      return
    if name.is_dir():
      self.addAnnotationFolder(name)
    else:
      self.addAnnotation(name)

  def addAnnotationFolder(self, folder: FilePath):
    folder = Path(folder).resolve()
    for file in folder.glob('*.*'):
      self.addAnnotation(file)

  def addImage_gui(self, copyToProject=True):
    fileFilter = "Image Files (*.png *.tif *.jpg *.jpeg *.bmp *.jfif);;All files(*.*)"
    fname = fns.popupFilePicker(None, 'Add Image to Project', fileFilter)
    if fname is not None:
      self.addImage(fname, copyToProj=copyToProject)

  def removeImage(self, imgName: FilePath):
    imgName = Path(imgName).resolve()
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
    annName = Path(annName).resolve()
    # Since no mapping exists of all annotations, loop the long way until the file is found
    for key, ann in self.imgToAnnMapping.items():
      if annName == ann:
        del self.imgToAnnMapping[key]
        ann.unlink()
        break

  def addAnnotation(self, name: FilePath=None, data: pd.DataFrame=None, image: FilePath=None,
                    overwriteOld=False):
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
      xpondingImgs = np.unique(data[REQD_TBL_FIELDS.SRC_IMG_FILENAME].to_numpy())
      # Break into annotaitons by iamge
      for img in xpondingImgs:
        self.addAnnotation(name, data[data[REQD_TBL_FIELDS.SRC_IMG_FILENAME] == img], img)
      return
    image = self._getFullImgName(Path(image))
    # Force provided annotations to now belong to this image
    data[REQD_TBL_FIELDS.SRC_IMG_FILENAME] = image.name
    # Since only one annotation file can exist per image, concatenate this with any existing files for the same image
    # if needed
    if image.parent != self.imagesDir:
      image = self._copyImgToProj(image)
    annForImg = self.imgToAnnMapping.get(image, None)
    oldAnns = []
    if annForImg is not None and not overwriteOld:
      oldAnns.append(self.compIo.importByFileType(annForImg))
    combinedAnns = oldAnns + [data]
    outAnn = pd.concat(combinedAnns, ignore_index=True)
    outAnn[REQD_TBL_FIELDS.INST_ID] = outAnn.index
    outFmt = f".{self.cfg['annotation-format']}"
    outName = self.annotationsDir / f'{image.name}{outFmt}'
    self.compIo.exportByFileType(outAnn, outName, verifyIntegrity=False, readOnly=False,
                       imgDir=self.imagesDir)
    self.imgToAnnMapping[image] = outName

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

  def _getFullImgName(self, name: Path, thorough=True):
    """
    From an absolute or relative image name, attempts to find the absolute path it corresponds
    to based on current project images. A match is located in the following order:
      - If the image path is already absolute, it is resolved, checked for existence, and returned
      - Solitary project images are searched to see if they end with the specified relative path
      - All base image directories are checked to see if they contain this subpath

    :param thorough: If `False`, as soon as a match is found the function returns. Otherwise,
      all solitary paths and images will be checked to ensure there is exactly one matching
      image for the name provided.
    """
    if name.is_absolute():
      return name.resolve()

    candidates = set()
    strName = str(name)
    for img in self.images:
      if str(img).endswith(strName):
        if not thorough:
          return img
        candidates.add(img)

    for parent in self.baseImgDirs:
      curName = (parent/name).resolve()
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

  @fns.dynamicDocstring(fileTypes=list(ComponentIO.exportTypes))
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
      limits:
        {fileTypes}
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
      outAnn = pd.concat(map(self.compIo.importByFileType, existingAnnFiles), ignore_index=True)
      outAnn[REQD_TBL_FIELDS.INST_ID] = outAnn.index
      self.compIo.exportByFileType(outAnn, outputFolder / f'annotations.'
                                                          f'{annotationFormat}', **exportOpts)
    else:
      outAnnsDir = outputFolder / 'annotations'
      outAnnsDir.mkdir(exist_ok=True)
      if self.cfg['annotation-format'] == annotationFormat:
        shutil.copytree(self.annotationsDir, outAnnsDir, dirs_exist_ok=True)
      else:
        for annFile in existingAnnFiles:
          self.compIo.convert(annFile, outAnnsDir/f'{annFile.stem}.{annotationFormat}',
                              importArgs=exportOpts, exportArgs=exportOpts)

  def _maybeEmit(self, signal: QtCore.Signal, emitList: Sequence[Union[Path, Tuple[Path, Path]]]):
    if not self._suppressSignals:
      signal.emit(emitList)

  @contextmanager
  def suppressSignals(self):
    oldSuppress = self._suppressSignals
    self._suppressSignals = True
    yield
    self._suppressSignals = oldSuppress