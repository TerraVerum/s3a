from __future__ import annotations
from functools import partial
from pathlib import Path
from typing import Dict

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from s3a import ParamEditor, FR_SINGLETON, FR_CONSTS, ComponentIO, models
from s3a.generalutils import attemptFileLoad
from s3a.graphicsutils import menuFromEditorActions, popupFilePicker, DropList, \
  ThumbnailViewer
from s3a.parameditors import ParamEditorPlugin, ProjectData
from s3a.structures import FilePath


class ProjectsPlugin(ParamEditorPlugin):
  name = 'Project'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls, 'Tools')

  def __init__(self):
    super().__init__()
    self.data = ProjectData()
    ioCls = FR_SINGLETON.registerGroup(FR_CONSTS.CLS_COMP_EXPORTER)(ComponentIO)
    ioCls.exportOnlyVis, ioCls.includeFullSourceImgName = \
      FR_SINGLETON.generalProps.registerProps(ioCls,
                                              [FR_CONSTS.EXP_ONLY_VISIBLE, FR_CONSTS.INCLUDE_FNAME_PATH]
                                              )
    self.compIo: ComponentIO = ioCls()

    self.toolsEditor.registerFunc(self.create_gui, name='Create')
    self.toolsEditor.registerFunc(self.open_gui, name='Open')
    self.toolsEditor.registerFunc(self.imageMgr_gui, name='Open Project Image')
    self._projImgMgr = ProjectImageManager()
    self._projImgThumbnails = self._projImgMgr.thumbnails
    self._projImgMgr.sigImageSelected.connect(lambda imgFname: self.s3a.setMainImg(self._projImgThumbnails.nameToFullPathMapping[imgFname]))

  def attachS3aRef(self, s3a: models.s3abase.S3ABase):
    self.menu = menuFromEditorActions(self.toolsEditor, menuParent=s3a)

    super().attachS3aRef(s3a)
    s3a.sigImageChanged.connect(lambda: self.loadNewAnns())

  def loadNewAnns(self, imgFname: FilePath=None):
    if imgFname is None:
      imgFname = self.s3a.srcImgFname
    if imgFname is None:
      return
    imgAnns = self.data.imgToAnnMapping.get(imgFname, None)
    if imgAnns is not None:
      self.s3a.compMgr.addComps(self.compIo.buildByFileType(imgAnns))

  def open(self, name: str):
    self.data.loadCfg(name)
    self._projImgThumbnails.clear()
    for img in self.data.images:
      self._projImgThumbnails.addThumbnail(img)

  def open_gui(self):
    fname = popupFilePicker(self.s3a, 'Select Project File', 'S3A Project (*.s3aprj)')
    if fname is not None:
      with pg.BusyCursor():
        self.open(fname)

  def imageMgr_gui(self):
    dlg = self._projImgMgr
    dlg.show()
    ok = dlg.exec_()
    if not ok:
      return

  def saveCurAnnotation(self):
    self.data.addAnnotation(data=self.s3a.exportableDf, image=self.s3a.srcImgFname, overwriteOld=True)

  def saveAll(self):
    self.saveCurAnnotation()
    self.data.saveCfg()

  def create_gui(self):
    wiz = NewProjectWizard(self)
    ok = wiz.exec_()
    if not ok:
      return
    parsedLists = {}
    for k, val in wiz.fileLists.items():
      model = val.model()
      lst = []
      for ii in range(model.rowCount()):
        lst.append(model.index(ii, 0).data())
      parsedLists[k] = lst
    # Since insertion order is preserved the extraction can be done without keys
    images, annotations = parsedLists.values()
    settings = wiz.projSettings
    projName = settings['Name']
    prevTemplate = settings['Template Project']
    if prevTemplate is not None:
      baseCfg = attemptFileLoad(prevTemplate)
      if not settings['Keep Existing Images'] or 'images' not in baseCfg:
        baseCfg['images'] = []
      if not settings['Keep Existing Annotations' or 'annotations' not in baseCfg]:
        baseCfg['annotations'] = []
    else:
      baseCfg = {'images': [], 'annotations': []}
    baseCfg['images'].extend(images)
    baseCfg['annotations'].extend(annotations)
    projPath = Path(wiz.projSettings['Location'])/projName/f'{projName}.yml'
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
    layout = QtWidgets.QVBoxLayout()
    page.setLayout(layout)
    layout.addWidget(tree)
    self.addPage(page)


    def getFileList(_flist: DropList, _title: str, _selectFolder=False):
      dlg = QtWidgets.QFileDialog()
      dlg.setModal(True)
      getFn = lambda *args, **kwargs: dlg.getOpenFileNames(*args, **kwargs)[0]
      if _selectFolder:
        getFn = lambda *args, **kwargs: [dlg.getExistingDirectory(*args, **kwargs)]
      files = getFn(self, _title, str(self.project.data.location))
      _flist.addItems(files)


    # -----
    # SELECTING PROJECT FILES
    # -----
    for fType in ['Images', 'Annotations']:
      page = QtWidgets.QWizardPage(self)
      page.setTitle(fType)
      curLayout = QtWidgets.QVBoxLayout()
      page.setLayout(curLayout)
      curLayout.addWidget(QtWidgets.QLabel(f'Project {fType.lower()} are shown below. Use the buttons'
                                      ' or drag and drop to add files.'))
      flist = DropList(self)
      self.fileLists[fType] = flist
      fileBtnLayout = QtWidgets.QHBoxLayout()
      curLayout.addWidget(flist)
      for title in f'Add Files', f'Add Folder':
        selectFolder = 'Folder' in title
        btn = QtWidgets.QPushButton(title, self)
        btn.clicked.connect(partial(getFileList, flist, title, selectFolder))
        fileBtnLayout.addWidget(btn)
      curLayout.addLayout(fileBtnLayout)
      self.addPage(page)


class ProjectImageManager(QtWidgets.QDialog):
  sigImageSelected = QtCore.Signal(str)
  def __init__(self, parent=None):
    super().__init__(parent)
    layout = QtWidgets.QVBoxLayout()
    self.setLayout(layout)
    self.thumbnails = ThumbnailViewer()
    self.completer = QtWidgets.QLineEdit()
    self.completer.setPlaceholderText('Type to filter')
    self.completer.textChanged.connect(self._filterThumbnails)
    self.thumbnails.itemDoubleClicked.connect(lambda item: self.sigImageSelected.emit(item.text()))

    layout.addWidget(self.completer)
    layout.addWidget(self.thumbnails)

  def _filterThumbnails(self, text):
    for ii in range(self.thumbnails.model().rowCount()):
      item = self.thumbnails.item(ii)
      if text in item.text():
        item.setHidden(False)
      else:
        item.setHidden(True)