import sys
from copy import copy
from pathlib import Path
from typing import Optional, Union, Type
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets

from s3a import ComponentIO, RunOpts
from s3a.constants import PRJ_CONSTS, REQD_TBL_FIELDS
from s3a.constants import PRJ_ENUMS
from s3a.controls.tableviewproxy import CompDisplayFilter, CompSortFilter
from s3a.generalutils import resolveAuthorName, dynamicDocstring, \
  serAsFrame
from s3a.models.tablemodel import ComponentMgr
from s3a.parameditors import FR_SINGLETON, EditorPropsMixin
from s3a.parameditors.appstate import AppStateEditor
from s3a.plugins.base import ParamEditorPlugin
from s3a.plugins.file import FilePlugin
from s3a.structures import FilePath, NChanImg, S3AIOError, \
  AlgProcessorError, S3AWarning
from s3a.views.imageareas import MainImage
from s3a.views.tableview import CompTableView

__all__ = ['S3ABase']

class S3ABase(EditorPropsMixin, QtWidgets.QMainWindow):
  """
  Top-level widget for producing component bounding regions from an input image.
  """

  sigRegionAccepted = QtCore.Signal()

  __groupingName__ = 'S3A Window'

  def __init__(self, parent=None, **startupSettings):
    super().__init__(parent)
    self.tblFieldToolbar = QtWidgets.QToolBar('Table Field Plugins')
    self.generalToolbar = QtWidgets.QToolBar('General')

    self.mainImg = MainImage(toolbar=self.generalToolbar)
    self.focusedImg = self.mainImg
    PRJ_CONSTS.TOOL_ACCEPT_FOC_REGION.opts['ownerObj'] = self.focusedImg
    self.focusedImg.toolsEditor.registerFunc(self.acceptFocusedRegion,
                                             btnOpts=PRJ_CONSTS.TOOL_ACCEPT_FOC_REGION)
    FR_SINGLETON.generalProps.registerFunc(FR_SINGLETON.actionStack.resizeStack,
                                           name=PRJ_CONSTS.PROP_UNDO_BUF_SZ.name,
                                           runOpts=RunOpts.ON_CHANGED,
                                           newMaxLen=300)
    self.statBar = QtWidgets.QStatusBar(self)
    self.menuBar_ = self.menuBar()

    FR_SINGLETON.shortcuts.registerShortcut(PRJ_CONSTS.TOOL_CLEAR_ROI,
                                            self.focusedImg.clearCurRoi,
                                            overrideOwnerObj=self.mainImg
                                            )

    self.compMgr = ComponentMgr()
    # Register exporter to allow user parameters
    ioCls = ComponentIO
    with FR_SINGLETON.generalProps.setBaseRegisterPath(PRJ_CONSTS.CLS_COMP_EXPORTER.name):
      ioCls.exportOnlyVis, ioCls.includeFullSourceImgName = \
        FR_SINGLETON.generalProps.registerProps(
          [PRJ_CONSTS.EXP_ONLY_VISIBLE, PRJ_CONSTS.INCLUDE_FNAME_PATH])
    self.compIo: ComponentIO = ioCls()
    ComponentIO.tableData = FR_SINGLETON.tableData

    self.compTbl = CompTableView()
    self.compDisplay = CompDisplayFilter(self.compMgr, self.mainImg, self.compTbl)

    self.compTbl.setSortingEnabled(True)
    self.compTbl.setAlternatingRowColors(True)
    # Allow filtering/sorting
    self.sortFilterProxy = CompSortFilter(self.compMgr)
    self.compTbl.setModel(self.sortFilterProxy)

    self.hasUnsavedChanges = False
    self.srcImgFname: Optional[Path] = None

    # -----
    # INTERFACE WITH QUICK LOADER
    # -----
    self.appStateEditor = AppStateEditor(self, name='App State Editor')

    for plugin in FR_SINGLETON.clsToPluginMapping.values(): # type: ParamEditorPlugin
      # Plugins created before window was initialized may need their plugins forcefully
      # attached here
      if plugin.win is not self:
        self._handleNewPlugin(plugin)
    FR_SINGLETON.sigPluginAdded.connect(self._handleNewPlugin)

    self.filePlg: FilePlugin = self.addPlugin(FilePlugin)

    # Connect signals
    # -----
    # COMPONENT MANAGER
    # -----
    def handleUpdate(*_args):
      self.hasUnsavedChanges = True
    self.compMgr.sigCompsChanged.connect(handleUpdate)

    # -----
    # MAIN IMAGE
    # -----
    def handleCompsChanged(changedDict: dict):
      focusedId = self.focusedImg.compSer[REQD_TBL_FIELDS.INST_ID]
      if focusedId in changedDict['deleted']:
        self.changeFocusedComp()
    self.compMgr.sigCompsChanged.connect(handleCompsChanged)

    # -----
    # COMPONENT TABLE
    # -----
    self.compDisplay.sigCompsSelected.connect(lambda newComps: self.changeFocusedComp(newComps))

    # -----
    # MISC
    # -----
    authorName = resolveAuthorName(startupSettings.pop('author', None))
    if authorName is None:
      sys.exit('No author name provided and no default author exists. Exiting.\n'
               'To start without error, provide an author name explicitly, e.g.\n'
               '"python -m s3a --author=<Author Name>"')
    FR_SINGLETON.tableData.annAuthor = authorName
    self.saveAllEditorDefaults()
    FR_SINGLETON.tableData.sigCfgUpdated.connect(lambda: self.resetTblFields())

  def resetTblFields(self):
    """
    When table fields change, the displayed columns must change and the view
    must be made aware. Ensure this occurs here
    """
    # Even if the field names are the same, e.g. classes may added or default values could
    # be changed. So, reset the cell editor delegates no matter what
    self.compTbl.setColDelegates()
    self.compTbl.popup.reflectDelegateChange()
    # Make sure this is necessary, first
    for mgr in self.compMgr, self.compTbl.popup.tbl.mgr:
      if mgr.colTitles == list([f.name for f in FR_SINGLETON.tableData.allFields]):
        # Fields haven't changed since last reset. Types could be different, but nothing
        # will break. So, the table doesn't have to be completely reset
        return

      mgr.beginResetModel()
      mgr.rmComps()
      mgr.resetFields()
      mgr.endResetModel()

  def _handleNewPlugin(self, plugin: ParamEditorPlugin):
    plugin.attachWinRef(self)
    if plugin.dock:
      plugin.dock.setParent(self)

  @staticmethod
  def saveAllEditorDefaults():
    for editor in FR_SINGLETON.registerableEditors:
      editor.saveCurStateAsDefault()

  @FR_SINGLETON.actionStack.undoable('Accept Focused Region')
  def acceptFocusedRegion(self):
    """Applies the focused image vertices to the corresponding component in the table"""
    # If the component was deleted
    mgr = self.compMgr
    focusedId = self.focusedImg.compSer[REQD_TBL_FIELDS.INST_ID]
    exists = focusedId in mgr.compDf.index
    if not exists and focusedId != REQD_TBL_FIELDS.INST_ID.value:
      # Could be a brand new component, allow in that case
      warn('Cannot accept region as this component was deleted.', S3AWarning)
      return

    self.sigRegionAccepted.emit()

    ser = self.focusedImg.compSer
    if ser[REQD_TBL_FIELDS.VERTICES].isEmpty():
      # Component should be erased. Since new components will not match existing
      # IDs the same function will work regardless of whether this was new or existing
      self.compMgr.rmComps([ser[REQD_TBL_FIELDS.INST_ID]])
      return

    if exists:
      undo = self._acceptFocused_existing(ser)
    else:
      undo = self._acceptFocused_new(ser)
    self.changeFocusedComp()
    yield
    undo()


  def _acceptFocused_new(self, compSer: pd.Series):
    # New, make a brand new table entry
    newIds = self.compMgr.addComps(serAsFrame(compSer))['added']
    def undo():
      self.compMgr.rmComps(newIds)
    return undo

  def _acceptFocused_existing(self, compSer: pd.Series):
    oldComp = self.compMgr.compDf.loc[[compSer[REQD_TBL_FIELDS.INST_ID]]].copy()
    modifiedDf = serAsFrame(compSer[[REQD_TBL_FIELDS.INST_ID, REQD_TBL_FIELDS.VERTICES]])
    self.compMgr.addComps(modifiedDf, addtype=PRJ_ENUMS.COMP_ADD_AS_MERGE)
    def undo():
      self.add_focusComps(oldComp, addType=PRJ_ENUMS.COMP_ADD_AS_MERGE)
      self.changeFocusedComp(oldComp.iloc[0])
    return undo


  def clearBoundaries(self):
    """Removes all components from the component table"""
    self.compMgr.rmComps()

  def addPlugin(self, pluginCls: Type[ParamEditorPlugin], *args, **kwargs):
    """See FR_SINGLETON.addPlugin"""
    plugin = FR_SINGLETON.addPlugin(pluginCls, *args, **kwargs)
    return plugin

  @FR_SINGLETON.actionStack.undoable('Change Main Image')
  def setMainImg(self, fileName: FilePath=None, imgData: NChanImg=None,
                 clearExistingComps=True):
    """
    * If fileName is None, the main and focused images are blacked out.
    * If only fileName is provided, it is assumed to be an image. The image data
    will be populated by reading in that file.
    * If both fileName and imgData are provided, then imgData is used to populate the
    image, and fileName is assumed to be the file associated with that data.

    :param fileName: Filename either to load or that corresponds to imgData
    :param imgData: N-Channel numpy image
    :param clearExistingComps: If True, erases all existing components on image load.
      Else, they are retained.
    """
    oldFile = self.srcImgFname
    oldData = self.mainImg.image
    if fileName is not None:
      fileName = Path(fileName).resolve()
    if fileName == self.srcImgFname:
      return

    self.saveCurAnnotation()

    if imgData is not None:
      self.mainImg.setImage(imgData)
    else:
      self.mainImg.setImage(fileName)
    self.srcImgFname = fileName

    self.changeFocusedComp()
    self.mainImg.plotItem.vb.autoRange()
    self.filePlg.addImage(fileName)
    self.loadNewAnnotations()
    yield
    self.setMainImg(oldFile, oldData, clearExistingComps)

  def saveCurAnnotation(self, allowOverwrite=False):
    srcImg = self.srcImgFname
    if srcImg is None:
      return
    srcImg_proj = self.filePlg.imagesDir/srcImg.name
    if srcImg_proj.exists() and not allowOverwrite and srcImg_proj != srcImg:
      warn(f'Image {srcImg.name} already present in project images. Using'
           f' existing image instead of current data.', S3AWarning)
    else:
      self.filePlg.addImage(name=srcImg, data=self.mainImg.image, copyToProj=True,
                            allowOverwrite=True)
    # srcImg_proj is guaranteed to exist at this point
    self.filePlg.addAnnotation(data=self.exportableDf, image=srcImg_proj, overwriteOld=True)
    self.srcImgFname = srcImg_proj
    self.hasUnsavedChanges = False

  def loadNewAnnotations(self, imgFname: FilePath=None):
    if imgFname is None:
      imgFname = self.srcImgFname
    if imgFname is None:
      return
    imgAnns = self.filePlg.imgToAnnMapping.get(imgFname, None)
    if imgAnns is not None:
      self.compMgr.addComps(self.compIo.buildByFileType(imgAnns, imgDir=self.filePlg.imagesDir,
                                                        imShape=self.mainImg.image.shape))
      # 'hasUnsavedChanges' will be true after this, even though the changes are saved.
      self.hasUnsavedChanges = False

  @dynamicDocstring(filters=ComponentIO.handledIoTypes_fileFilter())
  def exportCurAnnotation(self, outFname: Union[str, Path], readOnly=True, verifyIntegrity=True):
    """
    Exports current image annotations to a file. This may be more convenient than exporting
    an entire project if just the current current annotations are needed

    :param outFname:
      title: Output File
      helpText: Where to export. The file extension determines the save type
      pType: filepicker
      existing: False
      fileFilter: {filters}
    :param readOnly: Whether the export should be read-only
    :param verifyIntegrity: Whether the export should be re-imported to confirm
      its contents were successfully exported. This is only applicable for file types
      in `ComponentIO.roundTripIoTypes`
    """
    self.compIo.exportByFileType(self.exportableDf, outFname, imShape=self.mainImg.image.shape,
                                 readOnly=readOnly, verifyIntegrity=verifyIntegrity)
  @property
  def exportableDf(self):
    """
    Dataframe from manager with populated information for main image name and
    potentially filtered to only visible components (if requested by the user)
    """
    displayIds = self.compDisplay.displayedIds
    srcImgFname = self.srcImgFname
    if self.compIo.exportOnlyVis and displayIds is not None:
      exportIds = displayIds
    else:
      exportIds = self.compMgr.compDf.index
    exportDf: df = self.compMgr.compDf.loc[exportIds].copy()
    if not self.compIo.includeFullSourceImgName and srcImgFname is not None:
      # Only use the file name, not the whole path
      srcImgFname = srcImgFname.name
    elif srcImgFname is not None:
      srcImgFname = str(srcImgFname)
    # Assign correct export name for only new components
    overwriteIdxs = exportDf[REQD_TBL_FIELDS.SRC_IMG_FILENAME] == PRJ_CONSTS.ANN_CUR_FILE_INDICATOR.value
    # TODO: Maybe the current filename will match the current file indicator. What happens then?
    exportDf.loc[overwriteIdxs, REQD_TBL_FIELDS.SRC_IMG_FILENAME] = srcImgFname
    return exportDf

  def openAnnotations(self, inFname: str, loadType=PRJ_ENUMS.COMP_ADD_AS_NEW):
    pathFname = Path(inFname)
    if self.mainImg.image is None:
      raise S3AIOError('Cannot load components when no main image is set.')
    fType = pathFname.suffix[1:]
    if not any(fType in typ for typ in self.compIo.handledIoTypes):
      raise S3AIOError(f'Extension {fType} is not recognized. Must be one of:\n'
                       + self.compIo.handledIoTypes_fileFilter())
    newComps = self.compIo.buildByFileType(inFname, self.mainImg.image.shape)
    self.compMgr.addComps(newComps, loadType)

  def showModCompAnalytics(self):
    """
    Shows the result of each process stage for most recent result of the currently
    selected plugin
    """
    try:
      proc = self.focusedImg.currentPlugin.curProcessor
      proc.processor.stageSummary_gui()
    except AttributeError:
      # Processor or proc collection not set
      raise AlgProcessorError('Either no plugin is activated or the activated plugin'
                                ' has no processors')

  @FR_SINGLETON.actionStack.undoable('Create New Comp', asGroup=True)
  def add_focusComps(self, newComps: df, addType=PRJ_ENUMS.COMP_ADD_AS_NEW):
    changeDict = self.compMgr.addComps(newComps, addType)
    # Focus is performed by comp table
    # Arbitrarily choose the last possible component
    changeList = np.concatenate([changeDict['added'], changeDict['changed']])
    if len(changeList) == 0:
      return
    self.changeFocusedComp(self.compMgr.compDf.loc[[changeList[-1]]])

  @FR_SINGLETON.actionStack.undoable('Change Focused Component')
  def changeFocusedComp(self, newComps: df=None, forceKeepLastChange=False):
    oldSer = self.focusedImg.compSer.copy()
    oldImg = self.focusedImg.image
    if newComps is None or len(newComps) == 0:
      self.focusedImg.updateFocusedComp()
      self.compDisplay.regionPlot.focusById([])
      self.compDisplay.selectRowsById([])
    else:
      # TODO: More robust scenario if multiple comps are in the dataframe
      #   For now, just use the last in the selection. This is so that if multiple
      #   components are selected in a row, the most recently selected is always
      #   the current displayed.
      newComp: pd.Series = newComps.iloc[-1,:]
      newCompId = newComp[REQD_TBL_FIELDS.INST_ID]
      self.compDisplay.regionPlot.focusById([newCompId])
      self.focusedImg.updateFocusedComp(newComp)
    # Nothing happened since the last component change, so just replace it instead of
    # adding a distinct action to the buffer queue
    stack = FR_SINGLETON.actionStack
    if not forceKeepLastChange and stack.undoDescr == 'Change Focused Component':
      stack.actions.pop()
    yield
    if oldImg is not None and len(oldSer.loc[REQD_TBL_FIELDS.VERTICES]) > 0:
      self.changeFocusedComp(oldSer.to_frame().T, forceKeepLastChange=True)
    else:
      self.changeFocusedComp()