import sys
from pathlib import Path
from typing import Optional, Union, Type
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets

from s3a.generalutils import hierarchicalUpdate
from utilitys import EditorPropsMixin, RunOpts, ParamEditorPlugin, fns, ParamEditor

from s3a import ComponentIO
from s3a.constants import PRJ_CONSTS, REQD_TBL_FIELDS, LAYOUTS_DIR
from s3a.constants import PRJ_ENUMS
from s3a.controls.tableviewproxy import CompDisplayFilter, CompSortFilter
from s3a.models.tablemodel import ComponentMgr
from s3a.parameditors import PRJ_SINGLETON
from s3a.parameditors.appstate import AppStateEditor
from s3a.plugins.file import FilePlugin
from s3a.structures import FilePath, NChanImg
from s3a.views.imageareas import MainImage
from s3a.views.tableview import CompTableView

__all__ = ['S3ABase']

class S3ABase(EditorPropsMixin, QtWidgets.QMainWindow):
  """
  Top-level widget for producing component bounding regions from an input image.
  """

  sigRegionAccepted = QtCore.Signal()

  __groupingName__ = 'S3A Window'

  @classmethod
  def __initEditorParams__(cls):
    with PRJ_SINGLETON.generalProps.setBaseRegisterPath(PRJ_CONSTS.CLS_COMP_EXPORTER.name):
      cls.exportOnlyVis, cls.includeFullSourceImgName = \
        PRJ_SINGLETON.generalProps.registerProps(
          [PRJ_CONSTS.EXP_ONLY_VISIBLE, PRJ_CONSTS.INCLUDE_FNAME_PATH])

  def __init__(self, parent=None, **startupSettings):
    super().__init__(parent)
    self.tblFieldToolbar = QtWidgets.QToolBar('Table Field Plugins')
    self.generalToolbar = QtWidgets.QToolBar('General')

    self.mainImg = MainImage(toolbar=self.generalToolbar)
    PRJ_CONSTS.TOOL_ACCEPT_FOC_REGION.opts['ownerObj'] = self.mainImg
    self.mainImg.toolsEditor.registerFunc(self.acceptFocusedRegion,
                                          btnOpts=PRJ_CONSTS.TOOL_ACCEPT_FOC_REGION)
    _, param = PRJ_SINGLETON.generalProps.registerFunc(PRJ_SINGLETON.actionStack.resizeStack,
                                                       name=self.__groupingName__,
                                                       runOpts=RunOpts.ON_CHANGED,
                                                       maxLength=PRJ_CONSTS.PROP_UNDO_BUF_SZ.value,
                                                       returnParam=True)
    param.child('maxLength').setOpts(title=PRJ_CONSTS.PROP_UNDO_BUF_SZ.name)
    self.statBar = QtWidgets.QStatusBar(self)
    self.menuBar_ = self.menuBar()

    PRJ_SINGLETON.shortcuts.registerShortcut(PRJ_CONSTS.TOOL_CLEAR_ROI,
                                             self.mainImg.clearCurRoi,
                                             overrideOwnerObj=self.mainImg
                                             )

    self.compMgr = ComponentMgr()
    # Register exporter to allow user parameters
    ComponentIO.tableData = PRJ_SINGLETON.tableData

    self.compTbl = CompTableView()
    self.compDisplay = CompDisplayFilter(self.compMgr, self.mainImg, self.compTbl)

    self.compTbl.setSortingEnabled(True)
    self.compTbl.setAlternatingRowColors(True)
    # Allow filtering/sorting
    self.sortFilterProxy = CompSortFilter(self.compMgr)
    self.compTbl.setModel(self.sortFilterProxy)

    self.hasUnsavedChanges = False
    self.srcImgFname: Optional[Path] = None

    self.appStateEditor = AppStateEditor(self, name='App State Editor')

    # -----
    # INTERFACE WITH QUICK LOADER / PLUGINS
    # -----
    PRJ_SINGLETON.tableData.sigCfgUpdated.connect(lambda: self.resetTblFields())
    self.filePlg: FilePlugin = self.addPlugin(FilePlugin)
    self.compIo = self.filePlg.projData.compIo

    for plugin in PRJ_SINGLETON.clsToPluginMapping.values(): # type: ParamEditorPlugin
      # Plugins created before window was initialized may need their plugins forcefully
      # attached here
      if plugin.win is not self:
        self._handleNewPlugin(plugin)
    PRJ_SINGLETON.sigPluginAdded.connect(self._handleNewPlugin)

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
      ser = self.mainImg.compSer
      focusedId = ser[REQD_TBL_FIELDS.INST_ID]
      if focusedId in changedDict['deleted']:
        self.changeFocusedComp()
      elif focusedId in changedDict['changed']:
        ser.update(self.compMgr.compDf.loc[focusedId])
    self.compMgr.sigCompsChanged.connect(handleCompsChanged)

    # -----
    # COMPONENT TABLE
    # -----
    self.compDisplay.sigCompsSelected.connect(lambda newComps: self.changeFocusedComp(newComps))

    # -----
    # MISC
    # -----
    self.saveAllEditorDefaults()
    PRJ_SINGLETON.tableData.sigCfgUpdated.connect(lambda: self.resetTblFields())

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
      if mgr.colTitles == list([f.name for f in PRJ_SINGLETON.tableData.allFields]):
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
    for editor in PRJ_SINGLETON.registerableEditors:
      editor.saveCurStateAsDefault()

  @PRJ_SINGLETON.actionStack.undoable('Accept Focused Region')
  def acceptFocusedRegion(self):
    """Applies the focused image vertices to the corresponding component in the table"""
    # If the component was deleted
    mgr = self.compMgr
    focusedId = self.mainImg.compSer[REQD_TBL_FIELDS.INST_ID]
    exists = focusedId in mgr.compDf.index
    if not exists and focusedId != REQD_TBL_FIELDS.INST_ID.value:
      # Could be a brand new component, allow in that case
      warn('Cannot accept region as this component was deleted.', UserWarning)
      return

    self.sigRegionAccepted.emit()

    ser = self.mainImg.compSer
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
    newIds = self.compMgr.addComps(fns.serAsFrame(compSer))['added']
    def undo():
      self.compMgr.rmComps(newIds)
    return undo

  def _acceptFocused_existing(self, compSer: pd.Series):
    oldComp = self.compMgr.compDf.loc[[compSer[REQD_TBL_FIELDS.INST_ID]]].copy()
    modifiedDf = fns.serAsFrame(compSer)
    self.compMgr.addComps(modifiedDf, addtype=PRJ_ENUMS.COMP_ADD_AS_MERGE)
    def undo():
      self.add_focusComps(oldComp, addType=PRJ_ENUMS.COMP_ADD_AS_MERGE)
    return undo


  def clearBoundaries(self):
    """Removes all components from the component table"""
    self.compMgr.rmComps()

  def addPlugin(self, pluginCls: Type[ParamEditorPlugin], *args, **kwargs):
    """See PRJ_SINGLETON.addPlugin"""
    plugin = PRJ_SINGLETON.addPlugin(pluginCls, *args, **kwargs)
    return plugin

  @PRJ_SINGLETON.actionStack.undoable('Change Main Image')
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

    self.clearBoundaries()
    self.mainImg.plotItem.vb.autoRange()
    if fileName is not None:
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
      # warn(f'Image {srcImg.name} already present in project images. Using'
      #      f' existing image instead of current data.', UserWarning)
      pass
    else:
      self.filePlg.addImage(srcImg, data=self.mainImg.image, copyToProj=True,
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

  @fns.dynamicDocstring(filters=ComponentIO.handledIoTypes_fileFilter())
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
    if self.exportOnlyVis and displayIds is not None:
      exportIds = displayIds
    else:
      exportIds = self.compMgr.compDf.index
    exportDf: df = self.compMgr.compDf.loc[exportIds].copy()
    if not self.includeFullSourceImgName and srcImgFname is not None:
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
      raise IOError('Cannot load components when no main image is set.')
    fType = pathFname.suffix[1:]
    if not any(fType in typ for typ in self.compIo.handledIoTypes):
      raise IOError(f'Extension {fType} is not recognized. Must be one of:\n'
                       + self.compIo.handledIoTypes_fileFilter())
    newComps = self.compIo.buildByFileType(inFname, self.mainImg.image.shape)
    self.compMgr.addComps(newComps, loadType)

  @PRJ_SINGLETON.actionStack.undoable('Create New Comp', asGroup=True)
  def add_focusComps(self, newComps: df, addType=PRJ_ENUMS.COMP_ADD_AS_NEW):
    changeDict = self.compMgr.addComps(newComps, addType)
    # Focus is performed by comp table
    # Arbitrarily choose the last possible component
    changeList = np.concatenate([changeDict['added'], changeDict['changed']])
    if len(changeList) == 0:
      return
    self.changeFocusedComp(self.compMgr.compDf.loc[[changeList[-1]]])

  @PRJ_SINGLETON.actionStack.undoable('Change Focused Component')
  def changeFocusedComp(self, newComps: df=None, forceKeepLastChange=False):
    oldSer = self.mainImg.compSer.copy()
    oldImg = self.mainImg.image
    if newComps is None or len(newComps) == 0:
      self.mainImg.updateFocusedComp()
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
      self.mainImg.updateFocusedComp(newComp)
    # Nothing happened since the last component change, so just replace it instead of
    # adding a distinct action to the buffer queue
    stack = PRJ_SINGLETON.actionStack
    if not forceKeepLastChange and stack.undoDescr == 'Change Focused Component':
      stack.actions.pop()
    yield
    if oldImg is not None and len(oldSer.loc[REQD_TBL_FIELDS.VERTICES]) > 0:
      self.changeFocusedComp(fns.serAsFrame(oldSer), forceKeepLastChange=True)
    else:
      self.changeFocusedComp()

  # Stolen and adapted for python from https://stackoverflow.com/a/42910109/9463643
  # noinspection PyTypeChecker
  def addTabbedDock(self, area: QtCore.Qt.DockWidgetArea, dockwidget: QtWidgets.QDockWidget):
    curAreaWidgets = [d for d in self.findChildren(QtWidgets.QDockWidget)
                      if self.dockWidgetArea(d) == area]
    try:
      self.tabifyDockWidget(curAreaWidgets[-1], dockwidget)
    except IndexError:
      # First dock in area
      self.addDockWidget(area, dockwidget)