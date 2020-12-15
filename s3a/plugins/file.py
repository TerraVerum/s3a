from __future__ import annotations

import shutil
from _warnings import warn
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Dict, Optional, List, Set, Union, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from skimage import io

from s3a import ParamEditor, FR_SINGLETON, FR_CONSTS, ComponentIO, models, REQD_TBL_FIELDS
from s3a.generalutils import attemptFileLoad, dynamicDocstring, resolveYamlDict
from s3a.graphicsutils import popupFilePicker, DropList, \
  ThumbnailViewer, saveToFile
from s3a.parameditors.table import TableData
from s3a.structures import FilePath, S3AWarning, NChanImg, S3AIOError
from .base import ParamEditorPlugin
from ..constants import APP_STATE_DIR, PROJ_FILE_TYPE, BASE_DIR


class FilePlugin(ParamEditorPlugin):
  name = '&File'
  win: models.s3abase.S3A
  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()

  def __init__(self):
    super().__init__()
    self.projData = ProjectData()
    ioCls = FR_SINGLETON.registerGroup(FR_CONSTS.CLS_COMP_EXPORTER)(ComponentIO)
    ioCls.exportOnlyVis, ioCls.includeFullSourceImgName = \
      FR_SINGLETON.generalProps.registerProps(ioCls,
                                              [FR_CONSTS.EXP_ONLY_VISIBLE, FR_CONSTS.INCLUDE_FNAME_PATH]
                                              )
    self.compIo: ComponentIO = ioCls()
    self.autosaveTimer = QtCore.QTimer()

    self.registerFunc(self.save)
    self.registerFunc(self.showProjImgs_gui, name='Open Project Image')
    self.menu.addSeparator()

    self.registerFunc(self.create_gui, name='Create Project')
    self.registerFunc(self.open_gui, name='Open Project')

    self.registerFunc(lambda: self.win.setMainImg_gui(), name='Add New Image')
    self.registerFunc(lambda: self.win.openAnnotation_gui(), name='Add New Annotation')
    self.menu.addSeparator()

    self.registerFunc(self.addImages_gui, name='Add Image Batch')
    self.registerFunc(self.addAnnotations_gui, name='Add Annotation Batch')
    self.menu.addSeparator()

    self.registerPopoutFuncs('Export...', [self.projData.exportProj, self.projData.exportAnnotations], ['Project', 'Annotations'])

    self.registerPopoutFuncs('Autosave...', [self.startAutosave, self.stopAutosave])

    self._projImgMgr = ProjectImageManager()
    self._imgThumbnails = self._projImgMgr.thumbnails

    def onAdd(imList):
      for im in imList:
        self._imgThumbnails.addThumbnail(im, force=True)
    def onMove(imList):
      for oldName, newName in imList:
        self._imgThumbnails.nameToFullPathMapping[oldName.name] = newName
    def onDel(imList):
      if len(imList) == 0:
        # or QtWidgets.QMessageBox.question(
        # self.win, 'Confirm Delete', f'Are you sure you want to delete the following images?\n'
        #                             f'{[i.name for i in imList]})') != QtWidgets.QMessageBox.Yes:
        return
      delCurrent = False
      for name in imList:
        if name == self.win.srcImgFname:
          delCurrent = True
          continue
        self._imgThumbnails.removeThumbnail(name.name)
      if delCurrent:
        raise S3AIOError(f'Cannot delete {self.win.srcImgFname.name} since it is currently'
             f' being annotated. Change the image and try again.')
    self.projData.sigImagesAdded.connect(onAdd)
    self.projData.sigImagesMoved.connect(onMove)
    self.projData.sigImagesRemoved.connect(onDel)

    self._projImgMgr.sigImageSelected.connect(lambda imgFname: self.win.setMainImg(imgFname))
    def handleDelete(delImgs):
      for img in delImgs:
        self.projData.removeImage(img)
    self._projImgMgr.sigDeleteRequested.connect(handleDelete)
    self.projNameLbl = QtWidgets.QLabel()

    self._createDefaultProj()

  def _updateProjLbl(self):
    self.projNameLbl.setText(f'Project: {self.projData.cfgFname.name}')

  def attachWinRef(self, win: models.s3abase.S3ABase):
    super().attachWinRef(win)
    win.statBar.addWidget(self.projNameLbl)
    def handleChange():
      img = self.win.srcImgFname
      if img is not None:
        self.projData.addImage(img)
        self.loadNewAnns()
    win.sigImageChanged.connect(handleChange)
    win.sigImageAboutToChange.connect(lambda oldImg, newImg: self.saveCurAnnotation())
    def handleExport(_dir):
      self.projData.startup['image'] = str(win.srcImgFname)
      self.save()
      return str(self.projData.cfgFname)
    win.appStateEditor.addImportExportOpts('Project', self.open, handleExport, 0)

    self._projImgMgr.hide()
    win.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._projImgMgr)

  def _createDefaultProj(self):
    self.projData.create(name=APP_STATE_DIR / PROJ_FILE_TYPE, parent=self.projData)

  def loadNewAnns(self, imgFname: FilePath=None):
    if imgFname is None:
      imgFname = self.win.srcImgFname
    if imgFname is None:
      return
    imgAnns = self.projData.imgToAnnMapping.get(imgFname, None)
    if imgAnns is not None:
      self.win.compMgr.addComps(self.compIo.buildByFileType(imgAnns, imgDir=self.projData.imagesDir,
                                                            imShape=self.win.mainImg.image.shape))

  def open(self, name: str):
    if Path(name).resolve() == self.projData.cfgFname:
      return
    self._imgThumbnails.clear()
    self.projData.loadCfg(name)
    self._updateProjLbl()
    startupImg = self.projData.startup['image']
    if startupImg is not None:
      startupImg = Path(startupImg)
      if not startupImg.is_absolute():
        startupImg = self.projData.imagesDir/startupImg
      self.win.setMainImg(startupImg)

  def open_gui(self):
    fname = popupFilePicker(None, 'Select Project File', f'S3A Project (*.{PROJ_FILE_TYPE})')
    if fname is not None:
      self.win.setMainImg(None)
      with pg.BusyCursor():
        self.open(fname)

  def save(self):
    self.saveCurAnnotation()
    self.projData.saveCfg()

  @dynamicDocstring(ioTypes=list(ComponentIO.handledIoTypes))
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
    self.autosaveTimer.start(interval * 60 * 1000)
    self.autosaveTimer.timeout.connect(self.saveCurAnnotation)
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
        self.win.exportAnnotations(baseSaveNamePlusFolder)
        lastSavedDf = curDf.copy()

    self.autosaveTimer.timeout.connect(save_incrementCounter)

  def stopAutosave(self):
    self.autosaveTimer.stop()

  def showProjImgs_gui(self):
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

  def saveCurAnnotation(self):
    srcImg = self.win.srcImgFname
    if srcImg is None:
      return
    elif not srcImg.exists():
      # Data may have been set programmatically and given a name, so make sure this exists before saving
      srcImg = self.projData.addImage(name=srcImg, data=self.win.mainImg.image, copyToProj=True, allowOverwrite=True)
      if srcImg is None:
        # Orig file didn't exist, but it was in the image folder
        srcImg = self.win.srcImgFname
    self.projData.addAnnotation(data=self.win.exportableDf, image=srcImg, overwriteOld=True)
    self.win.srcImgFname = self.projData.imagesDir / srcImg.name
    self.win.hasUnsavedChanges = False

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
      baseCfg = attemptFileLoad(prevTemplate)
      if not settings['Keep Existing Images'] or 'images' not in baseCfg:
        baseCfg['images'] = []
      if not settings['Keep Existing Annotations' or 'annotations' not in baseCfg]:
        baseCfg['annotations'] = []
    else:
      baseCfg = {'images': [], 'annotations': []}
    baseCfg['images'].extend(images)
    baseCfg['annotations'].extend(annotations)
    projPath = Path(wiz.projSettings['Location'])/projName
    self.projData.create(name=projPath, cfg=baseCfg, parent=self.projData)


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
    editor = ParamEditor(saveDir=None, paramList=settings)
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


def hierarchicalUpdate(curDict: dict, other: dict):
  """Dictionary update that allows nested keys to be updated without deleting the non-updated keys"""
  if other is None:
    return
  for k, v in other.items():
    curVal = curDict.get(k, None)
    if isinstance(curVal, dict) and isinstance(v, dict):
      hierarchicalUpdate(curVal, v)
    else:
      curDict[k] = v


class ProjectData(QtCore.QObject):
  sigImagesAdded = QtCore.Signal(object)
  """List[Path] of added image"""
  sigImagesRemoved = QtCore.Signal(object)
  """List[Path] of removed image"""
  sigImagesMoved = QtCore.Signal(object)
  """
  List[(oldPath, NewPath)]
  
  Used mainly when images from outside the project are annotated. In that case, images are copied to inside the project,
  and this signal will be emitted.  
  """

  def __init__(self, cfgFname: FilePath=None, cfgDict: dict=None):
    super().__init__()
    self.tableData = TableData()
    self.cfg = {}
    self.cfgFname: Optional[Path] = None
    self.images: List[Path] = []
    self.baseImgDirs: Set[Path] = set()
    self.imgToAnnMapping: Dict[Path, Path] = {}
    """Records annotations belonging to each image"""

    self._suppressSignals = False
    """If this is *True*, no signals will be emitted """
    self.compIo = ComponentIO()
    self.compIo.tableData = self.tableData

    if cfgFname is not None or cfgDict is not None:
      self.loadCfg(cfgFname, cfgDict)

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
    _, defaultCfg = resolveYamlDict(BASE_DIR/'projectcfg.yml')
    cfgFname, cfgDict = resolveYamlDict(cfgFname, cfgDict)
    cfgFname = cfgFname.resolve()
    if not force and self.cfgFname == cfgFname:
      return None
    hierarchicalUpdate(defaultCfg, cfgDict)
    self.cfgFname = cfgFname
    cfg = self.cfg = defaultCfg
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
    self.tableData.loadCfg(tableName, tableDict)

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

    for annotation in self.cfg['annotations']:
      if isinstance(annotation, dict):
        self.addAnnotation(**annotation)
      else:
        self.addAnnotationByPath(annotation)
    return self.cfgFname

  @classmethod
  def create(cls, *, name: FilePath= f'./{PROJ_FILE_TYPE}', cfg: dict=None, parent: ProjectData=None):
    """
    Creates a new project with the specified settings in the specified directory.
    :param name:
      helpText: Project Name. The parent directory of this name indicates the directory in which to create the project
      pType: filepicker
    :param cfg: see `ProjectData.loadCfg` for information
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

    tdName = parent.tableData.cfgFname
    if tdName.resolve() != parent.cfgFname:
      tdName = tdName.name
      saveToFile(parent.tableData.cfg, location / tdName, True)
      parent.tableData.cfgFname = tdName
    else:
      tdName = tdName.name
    parent.cfg['table-cfg'] = tdName

    parent.saveCfg()
    return parent

  @classmethod
  def open(cls, name: FilePath):
    proj = cls()
    proj.loadCfg(name)

  def saveCfg(self):
    location = self.location
    annDir = self.annotationsDir
    strAnnNames = [str(annDir.relative_to(location))]
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
    if len(offendingAnns) > 0:
      warn('Encountered annotation(s) in project config, but not officially added. Offending files:\n'
           + ',\n'.join(offendingAnns), S3AWarning)
    self.cfg['images'] = strImgNames
    self.cfg['annotations'] = strAnnNames
    saveToFile(self.cfg, self.cfgFname)

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
      warn(f'Provided image path does not exist: {image}\nNo action performed.', S3AWarning)
      return []
    if image.is_dir():
      ret = self.addImageFolder(image, copyToProj)
    else:
      ret = self.addImage(name, copyToProj=copyToProj)
      ret = [] if ret is None else [ret]
    return ret

  @FR_SINGLETON.actionStack.undoable('Add Project Image')
  def addImage(self, name: FilePath, data: NChanImg=None, copyToProj=False, allowOverwrite=False):
    name = Path(name).resolve()
    if copyToProj or data is not None:
      name = self._copyImgToProj(name, data, allowOverwrite)
    if name in self.images:
      # Indicate the image was already present to calling scope
      return None
    self.images.append(name)
    self._maybeEmit(self.sigImagesAdded, [name])
    yield name
    self.removeImage(name)

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
      warn(f'Provided annotation path does not exist: {name}\nNo action performed.', S3AWarning)
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
    fname = popupFilePicker(None, 'Add Image to Project', fileFilter)
    if fname is not None:
      self.addImage(fname, copyToProj=copyToProject)

  @FR_SINGLETON.actionStack.undoable('Remove Project Image')
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
    yield
    if imgName.parent == self.imagesDir:
      # TODO: Cache removed images in a temp dir, then move them to that temp dir instead of unlinking
      #  on delete. This will make 'remove' undoable
      raise S3AIOError('Can only undo undo image removal when the image was outside the project.'
                       f' Image {imgName.name} was either annotated or directly placed in the project images'
                       f' directory, and was deleted during removal. To re-add, do so from the original image'
                       f' location outside the project directory.')
    self.addImage(imgName)

  def removeAnnotation(self, annName: FilePath):
    annName = Path(annName).resolve()
    # Since no mapping exists of all annotations, loop the long way until the file is found
    for key, ann in self.imgToAnnMapping.items():
      if annName == ann:
        del self.imgToAnnMapping[key]
        break

  def addAnnotation(self, name: FilePath=None, data: pd.DataFrame=None, image: FilePath=None,
                    overwriteOld=False):
    # Housekeeping for default arguments
    if name is None and data is None:
      raise S3AIOError('`name` and `data` cannot both be `None`')
    elif name in self.imgToAnnMapping.values():
      # Already present, shouldn't be added
      return
    if data is None:
      data = ComponentIO.buildByFileType(name)
    if image is None:
      # If no explicit matching to an image is provided, try to determine based on annotation name
      xpondingImgs = np.unique(data[REQD_TBL_FIELDS.SRC_IMG_FILENAME].to_numpy())
      # Break into annotaitons by iamge
      for img in xpondingImgs:
        self.addAnnotation(name, data[data[REQD_TBL_FIELDS.SRC_IMG_FILENAME] == img], img)
      return
    image = self._getFullImgName(Path(image))
    # Since only one annotation file can exist per image, concatenate this with any existing files for the same image
    # if needed
    if image.parent != self.imagesDir:
      image = self._copyImgToProj(image)
    annForImg = self.imgToAnnMapping.get(image, None)
    oldAnns = []
    if annForImg is not None and not overwriteOld:
      oldAnns.append(ComponentIO.buildByFileType(annForImg))
    combinedAnns = oldAnns + [data]
    outAnn = pd.concat(combinedAnns, ignore_index=True)
    outAnn[REQD_TBL_FIELDS.INST_ID] = outAnn.index
    outFmt = f".{self.cfg['annotation-format']}"
    outName = self.annotationsDir / f'{image.name}{outFmt}'
    ComponentIO.exportByFileType(outAnn, outName, verifyIntegrity=False, readOnly=False, imgDir=self.imagesDir)
    self.imgToAnnMapping[image] = outName

  def _copyImgToProj(self, name: Path, data: NChanImg=None, allowOverwrite=False):
    newName = (self.imagesDir/name.name).resolve()
    if (newName.exists() and not allowOverwrite) or newName == name:
      # Already in the project, no need to copy
      return newName
    if name.exists() and data is None:
      shutil.copy(name, newName)
    elif data is not None:
      # Programmatically created or not from a local file
      # noinspection PyTypeChecker
      io.imsave(newName, data)
    else:
      raise S3AIOError(f'No image data associated with {name.name}. Either the file does not exist or no'
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
            f' {numCandidates} candidate images were found'
      if numCandidates == 0:
        msg += '.'
      else:
        msg += f':\n{", ".join([c.name for c in candidates])}'
      raise S3AIOError(msg)
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

  @dynamicDocstring(fileTypes=ComponentIO.handledIoTypes_fileFilter().split(';;'))
  def exportAnnotations(self, outputFolder:FilePath= 's3a-export', annotationFormat='csv', combine=False, includeImages=True):
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

    :param combine: If `True`, all annotation files will be combined into one exported file with name `annotations.<format>`
    :param includeImages: If `True`, the corresponding image for each annotation will also be exported into an `images`
      folder
    """
    self.saveCfg()
    outputFolder = Path(outputFolder)

    if outputFolder.resolve() == self.cfgFname and not combine:
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
      outAnn = pd.concat(map(ComponentIO.buildByFileType, existingAnnFiles))
      ComponentIO.exportByFileType(outAnn, outputFolder / f'annotations.{annotationFormat}')
    else:
      outAnnsDir = outputFolder / 'annotations'
      if self.cfg['annotation-format'] == annotationFormat:
        shutil.copytree(self.annotationsDir, outAnnsDir)
      else:
        for annFile in existingAnnFiles:
          ioArgs = {'imgDir': self.imagesDir}
          ComponentIO.convert(annFile, outAnnsDir/f'{annFile.stem}.{annotationFormat}', ioArgs, ioArgs)

  def _maybeEmit(self, signal: QtCore.Signal, imgList: List[Union[Path, Tuple[Path, Path]]]):
    if not self._suppressSignals:
      signal.emit(imgList)

  @contextmanager
  def suppressSignals(self):
    oldSuppress = self._suppressSignals
    self._suppressSignals = True
    yield
    self._suppressSignals = oldSuppress