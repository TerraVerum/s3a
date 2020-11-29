import typing
from functools import partial
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import ParameterTree, Parameter

from s3a import FR_SINGLETON, FR_CONSTS as FRC, REQD_TBL_FIELDS as RTF, ComplexXYVertices, \
  XYVertices, FRParam, ComponentIO as frio, FR_CONSTS, ComponentIO, ParamEditor, models
from s3a.generalutils import pascalCaseToTitle, attemptFileLoad
from s3a.graphicsutils import ThumbnailViewer, DropList
from s3a.models.s3abase import S3ABase
from s3a.parameditors import ParamEditorDockGrouping, ParamEditorPlugin, ProjectData
from s3a.parameditors.genericeditor import TableFieldPlugin
from s3a.processing.algorithms import _historyMaskHolder
from s3a.structures import NChanImg, GrayImg, FilePath
from s3a.views.regions import MultiRegionPlot, makeMultiRegionDf


class VerticesPlugin(TableFieldPlugin):
  name = 'Vertices'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.procCollection = FR_SINGLETON.imgProcClctn.createProcessorForClass(cls)

    cls.docks = ParamEditorDockGrouping([cls.toolsEditor, cls.procCollection], cls.name)

  def __init__(self):
    self.region = MultiRegionPlot()
    self.region.hide()
    self.firstRun = True

    # Disable local cropping on primitive grab cut by default
    self.procCollection.nameToProcMapping['Primitive Grab Cut'].setStageEnabled(['Crop To Local Area'], False)

  def attachS3aRef(self, s3a: S3ABase):
    super().attachS3aRef(s3a)
    s3a.focusedImg.addItem(self.region)

    def fill():
      """Completely fill the focused region mask"""
      if self.focusedImg.image is None: return
      clsIdx = self.focusedImg.classIdx
      filledImg = np.ones(self.focusedImg.image.shape[:2], dtype='uint16')*(clsIdx+1)
      self.updateRegionFromClsImg(filledImg)
    def clear():
      """
      Clear the vertices in the focused image
      """
      self.updateRegionFromDf(None)
    def clearProcessorHistory():
      """
      Each time an update is made in the processor, it is saved so algorithmscan take
      past edits into account when performing their operations. Clearing that history
      will erase algorithm knowledge of past edits.
      """
      _historyMaskHolder[0].fill(0)
    funcLst = [self.resetFocusedRegion, fill, clear, clearProcessorHistory]
    paramLst = [FRC.TOOL_RESET_FOC_REGION, FRC.TOOL_FILL_FOC_REGION,
                FRC.TOOL_CLEAR_FOC_REGION, FRC.TOOL_CLEAR_HISTORY]
    for func, param in zip(funcLst, paramLst):
      self.toolsEditor.registerFunc(func, btnOpts=param)


  def updateAll(self, mainImg: Optional[NChanImg], newComp: Optional[pd.Series] = None):
    if self.focusedImg.image is None:
      self.updateRegionFromDf(None)
      return
    self.updateRegionFromDf(self.focusedImg.compSer.to_frame().T, self.focusedImg.bbox[0, :])
    self.firstRun = True

  def handleShapeFinished(self, roiVerts: XYVertices):
    roiVerts = roiVerts.astype(int)
    vertsDict = {}
    act = self.focusedImg.drawAction
    img = self.focusedImg.image

    if act == FRC.DRAW_ACT_ADD:
      vertsDict['fgVerts'] = roiVerts
    elif act == FRC.DRAW_ACT_REM:
      vertsDict['bgVerts'] = roiVerts

    if img is None:
      compGrayscale = None
      compMask = None
    else:
      compGrayscale = self.region.toGrayImg(img.shape[:2])
      compMask = compGrayscale > 0
    # TODO: When multiple classes can be represented within focused image, this is where
    #  change will have to occur
    newGrayscale = self.curProcessor.run(image=img, prevCompMask=compMask, **vertsDict,
                                    firstRun=self.firstRun).astype('uint16')
    newGrayscale *= (self.focusedImg.classIdx+1)
    self.firstRun = False
    if not np.array_equal(newGrayscale, compGrayscale):
      self.updateRegionFromClsImg(newGrayscale)


  @FR_SINGLETON.actionStack.undoable('Modify Focused Component')
  def updateRegionFromDf(self, newData: pd.DataFrame=None, offset: XYVertices=None):
    """
    Updates the current focused region using the new provided vertices
    :param newData: Dataframe to use.If *None*, the image will be totally reset and the component
      will be removed. Otherwise, the provided value will be used. For column information,
      see `makeMultiRegionDf`
    :param offset: Offset of newVerts relative to main image coordinates
    """
    fImg = self.focusedImg
    if fImg.image is None:
      self.region.clear()
      return
    oldData = self.region.regionData

    oldSelfImg = fImg.image
    oldSer = fImg.compSer

    if offset is None:
      offset = fImg.bbox[0,:]
    if newData is None:
      newData = makeMultiRegionDf(0)
    # 0-center new vertices relative to FocusedImage image
    # Make a copy of each list first so we aren't modifying the
    # original data
    centeredData = newData.copy()
    centeredVerts = []
    for complexVerts in centeredData[RTF.VERTICES]:
      newVertList = ComplexXYVertices()
      for vertList in complexVerts:
        newVertList.append(vertList-offset)
      centeredVerts.append(newVertList)
    centeredData[RTF.VERTICES] = centeredVerts
    if np.any(fImg.bbox[0,:] != offset) or not oldData.equals(centeredData):
      self.region.resetRegionList(newRegionDf=centeredData)
      yield
    else:
      return
    if (fImg.compSer.loc[RTF.INST_ID] != oldSer.loc[RTF.INST_ID]
        or fImg.image is None):
      fImg.updateAll(oldSelfImg, oldSer)
    self.region.resetRegionList(oldData, convertClasses=False)

  def updateRegionFromClsImg(self, clsImg: GrayImg):
    df = frio.buildFromClassPng(clsImg)
    self.updateRegionFromDf(df, offset=XYVertices([0, 0]))
    pass

  def acceptChanges(self, overrideVerts: ComplexXYVertices=None):
    # Add in offset from main image to VertexRegion vertices
    ser = self.focusedImg.compSer
    if overrideVerts is not None:
      ser.loc[RTF.VERTICES] = overrideVerts
      return
    newVerts_lst = self.region.regionData[RTF.VERTICES].copy()
    newVerts = ComplexXYVertices()
    for verts in newVerts_lst:
      newVerts.extend(verts.copy())
    for vertList in newVerts:
      vertList += self.focusedImg.bbox[0,:]
    ser.at[RTF.VERTICES] = newVerts

  def clearFocusedRegion(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.focusedImg.image is None:
      return
    self.updateRegionFromDf(None)

  def resetFocusedRegion(self):
    """Reset the focused image by restoring the region mask to the last saved state"""
    if self.focusedImg.image is None:
      return
    self.updateRegionFromDf(self.focusedImg.compSer.to_frame().T)

  def _onActivate(self):
    self.region.show()

  def _onDeactivate(self):
    self.region.hide()

class ProjectsPlugin(ParamEditorPlugin):
  name = 'Project'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls, 'Tools')
    cls.docks = ParamEditorDockGrouping([cls.toolsEditor], cls.name)

  def __init__(self):
    self.data = ProjectData()
    ioCls = FR_SINGLETON.registerGroup(FR_CONSTS.CLS_COMP_EXPORTER)(ComponentIO)
    ioCls.exportOnlyVis, ioCls.includeFullSourceImgName = \
      FR_SINGLETON.generalProps.registerProps(ioCls,
                                              [FR_CONSTS.EXP_ONLY_VISIBLE, FR_CONSTS.INCLUDE_FNAME_PATH]
                                              )
    self.compIo: ComponentIO = ioCls()

    self.toolsEditor.registerFunc(self.create_gui, name='Create')
    self._projImgThumbnails = ThumbnailViewer()

  def attachS3aRef(self, s3a: models.s3abase.S3ABase):
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

  def openProject(self, name: str):
    self.data.loadCfg(name)
    self._projImgThumbnails.clear()
    for img in self.data.images:
      self._projImgThumbnails.addThumbnail(img)

  def save(self):
    self.data.addAnnotation(data=self.s3a.compMgr.compDf, image=self.s3a.srcImgFname, overwriteOld=True)

  def create_gui(self):
    wiz = NewProjectWizard(self)
    wiz.exec_()
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