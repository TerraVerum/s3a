from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Dict

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from s3a import ParamEditor, FR_SINGLETON, FR_CONSTS, ComponentIO, models
from s3a.generalutils import attemptFileLoad
from s3a.graphicsutils import popupFilePicker, DropList, \
  ThumbnailViewer
from s3a.parameditors import ProjectData
from s3a.structures import FilePath
from .misc import MiscFunctionsPluginBase
from ..constants import APP_STATE_DIR, PROJ_FILE_TYPE


class ProjectsPlugin(MiscFunctionsPluginBase):
  name = 'Project'

  def __init__(self):
    super().__init__()
    self.data = ProjectData()
    ioCls = FR_SINGLETON.registerGroup(FR_CONSTS.CLS_COMP_EXPORTER)(ComponentIO)
    ioCls.exportOnlyVis, ioCls.includeFullSourceImgName = \
      FR_SINGLETON.generalProps.registerProps(ioCls,
                                              [FR_CONSTS.EXP_ONLY_VISIBLE, FR_CONSTS.INCLUDE_FNAME_PATH]
                                              )
    self.compIo: ComponentIO = ioCls()

    self.registerFunc(self.create_gui, name='Create Project')

    self.registerFunc(self.addImages_gui, name='Images', submenuName='Add')
    self.registerFunc(self.addAnnotations_gui, name='Annotations', submenuName='Add')

    self.registerFunc(self.open_gui, name='Project', submenuName='Open')
    self.registerFunc(self.showProjImgs_gui, name='Image', submenuName='Open')

    self.registerFunc(self.save, name='Project', submenuName='Save')
    self.registerFunc(self.saveCurAnnotation, name='Annotation', submenuName='Save')

    self.toolsEditor.params.addChild(dict(name='Export', type='group'))
    for title, func in zip(['Project', 'Annotations'], [self.data.exportProj, self.data.exportAnnotations]):
      self.toolsEditor.registerFunc(func, title, paramPath=('Export',))
    act = self.menu.addAction('Export...')
    act.triggered.connect(self.showExportOpts)

    self._projImgMgr = ProjectImageManager()
    self._projImgThumbnails = self._projImgMgr.thumbnails
    self._projImgMgr.sigImageSelected.connect(lambda imgFname: self.s3a.setMainImg(imgFname))
    def handleDelete(delImgs):
      for img in delImgs:
        self.data.removeImage(img)
    self._projImgMgr.sigDeleteRequested.connect(handleDelete)
    self.projNameLbl = QtWidgets.QLabel()

    self._createDefaultProj()

  def _updateProjLbl(self):
    self.projNameLbl.setText(f'Project: {self.data.cfgFname.name}')

  def attachS3aRef(self, s3a: models.s3abase.S3ABase):
    super().attachS3aRef(s3a)
    s3a.statBar.addWidget(self.projNameLbl)
    def handleChange():
      img = self.s3a.srcImgFname
      if img is not None:
        self.data.addImage(img)
        self.loadNewAnns()
    s3a.sigImageChanged.connect(handleChange)
    s3a.sigImageAboutToChange.connect(lambda oldImg, newImg: self.saveCurAnnotation())
    def handleExport(_dir):
      self.save()
      return str(self.data.cfgFname)
    s3a.appStateEditor.addImportExportOpts('Project', self.open, handleExport, 0)

  def _createDefaultProj(self):
    self.data.create(name=APP_STATE_DIR/PROJ_FILE_TYPE)

  def loadNewAnns(self, imgFname: FilePath=None):
    if imgFname is None:
      imgFname = self.s3a.srcImgFname
    if imgFname is None:
      return
    imgAnns = self.data.imgToAnnMapping.get(imgFname, None)
    if imgAnns is not None:
      self.s3a.compMgr.addComps(self.compIo.buildByFileType(imgAnns, imgDir=self.data.imagesDir,
                                                            imShape=self.s3a.mainImg.image.shape))

  def open(self, name: str):
    self.data.loadCfg(name)
    self._projImgThumbnails.clear()
    for img in self.data.images:
      self._projImgThumbnails.addThumbnail(img)
    self._updateProjLbl()
    startupImg = self.data.settings['startup-image']
    if startupImg is not None:
      self.s3a.setMainImg(startupImg)

  def open_gui(self):
    fname = popupFilePicker(None, 'Select Project File', f'S3A Project (*.{PROJ_FILE_TYPE})')
    if fname is not None:
      self.s3a.setMainImg(None)
      with pg.BusyCursor():
        self.open(fname)

  def save(self):
    self.saveCurAnnotation()
    self.data.saveCfg()

  def showProjImgs_gui(self):
    self._projImgMgr.show()
    self._projImgMgr.exec_()

  def addImages_gui(self):
    wiz = QtWidgets.QWizard()
    page = NewProjectWizard.createFilePage('Images', wiz)
    wiz.addPage(page)
    if wiz.exec_():
      for file in page.fileList.files:
        self.data.addImageByPath(file)

  def addAnnotations_gui(self):
    wiz = QtWidgets.QWizard()
    page = NewProjectWizard.createFilePage('Annotations', wiz)
    wiz.addPage(page)
    if wiz.exec_():
      for file in page.fileList.files:
        self.data.addAnnotationByPath(file)

  def showExportOpts(self):
    self.s3a.showEditorDock(self.toolsEditor)
    next(iter(self.toolsEditor.params.child('Export').items)).setExpanded(True)
    self.s3a.fixDockWidth(self.toolsEditor.dock)


  def saveCurAnnotation(self):
    srcImg = self.s3a.srcImgFname
    if srcImg is None:
      return
    self.data.addAnnotation(data=self.s3a.exportableDf, image=srcImg, overwriteOld=True)
    self.s3a.srcImgFname = self.data.imagesDir/srcImg.name

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
    self.data = ProjectData.create(name=projPath, cfg=baseCfg)


class NewProjectWizard(QtWidgets.QWizard):

  def __init__(self, project: ProjectsPlugin, parent=None) -> None:
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
    curLayout.addWidget(QtWidgets.QLabel(f'Project {name.lower()} are shown below. Use the buttons'
                                         ' or drag and drop to add files.'))
    flist = DropList(wizard)
    page.fileList = flist
    fileBtnLayout = QtWidgets.QHBoxLayout()
    curLayout.addWidget(flist)
    for title in f'Add Files', f'Add Folder':
      selectFolder = 'Folder' in title
      btn = QtWidgets.QPushButton(title, wizard)
      btn.clicked.connect(partial(getFileList, flist, title, selectFolder))
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