from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Dict

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import Parameter

from s3a import ParamEditor, FR_SINGLETON, FR_CONSTS, ComponentIO, models
from s3a.generalutils import attemptFileLoad, dynamicDocstring
from s3a.graphicsutils import popupFilePicker, DropList, \
  ThumbnailViewer, flexibleParamTree, paramWindow
from s3a.parameditors import ProjectData
from s3a.structures import FilePath
from .base import ParamEditorPlugin
from ..constants import APP_STATE_DIR, PROJ_FILE_TYPE


class FilePlugin(ParamEditorPlugin):
  name = '&File'

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
    self.registerFunc(self.showProjImgs_gui, name='Open Image')
    self.menu.addSeparator()

    self.registerFunc(self.create_gui, name='Create Project')
    self.registerFunc(self.open_gui, name='Open Project')

    self.registerFunc(lambda: self.win.setMainImg_gui, name='Add New Image')
    self.registerFunc(lambda: self.win.openAnnotation_gui, name='Add New Annotation')
    self.menu.addSeparator()

    self.registerFunc(self.addImages_gui, name='Add Image Batch')
    self.registerFunc(self.addAnnotations_gui, name='Add Annotation Batch')
    self.menu.addSeparator()

    self.registerPopoutFuncs('Export...', [self.projData.exportProj, self.projData.exportAnnotations], ['Project', 'Annotations'])

    self.registerPopoutFuncs('Autosave...', [self.startAutosave, self.stopAutosave])

    self._projImgMgr = ProjectImageManager()
    self._projImgThumbnails = self._projImgMgr.thumbnails
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
      self.projData.settings['startup-image'] = str(win.srcImgFname)
      self.save()
      return str(self.projData.cfgFname)
    win.appStateEditor.addImportExportOpts('Project', self.open, handleExport, 0)

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
    self.projData.loadCfg(name)
    self._projImgThumbnails.clear()
    for img in self.projData.images:
      self._projImgThumbnails.addThumbnail(img)
    self._updateProjLbl()
    startupImg = self.projData.settings['startup-image']
    if startupImg is not None:
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
    self._projImgMgr.exec_()

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
    self.projData.addAnnotation(data=self.win.exportableDf, image=srcImg, overwriteOld=True)
    self.win.srcImgFname = self.projData.imagesDir / srcImg.name

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
    self.projData = ProjectData.create(name=projPath, cfg=baseCfg)


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

class ProjectImageManager(QtWidgets.QDialog):
  def __init__(self, parent=None):
    super().__init__(parent)
    layout = QtWidgets.QVBoxLayout()
    self.setLayout(layout)
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