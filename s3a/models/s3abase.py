import sys
from pathlib import Path
from typing import Optional, Union, Callable, Dict, Any
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets

from s3a import FRComplexVertices, FRComponentIO
from s3a.constants import FR_CONSTS, REQD_TBL_FIELDS
from s3a.constants import FR_ENUMS
from s3a.controls.tableviewproxy import FRCompDisplayFilter, FRCompSortFilter
from s3a.generalutils import resolveAuthorName, imgCornerVertices
from s3a.graphicsutils import addDirItemsToMenu, saveToFile
from s3a.models.tablemodel import FRComponentMgr
from s3a.parameditors import FRParamEditor
from s3a.parameditors import FRParamEditorPlugin
from s3a.parameditors import FR_SINGLETON
from s3a.parameditors.appstate import FRAppStateEditor
from s3a.structures import FilePath, NChanImg, FRIOError, \
  FRAlgProcessorError, FRS3AWarning
from s3a.views.imageareas import FRMainImage, FRFocusedImage
from s3a.views.tableview import FRCompTableView

__all__ = ['S3ABase']

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_S3A_MODEL)
class S3ABase(QtWidgets.QMainWindow):
  """
  Top-level widget for producing component bounding regions from an input image.
  """
  @classmethod
  def __initEditorParams__(cls):
    cls.estBoundsOnStart, cls.undoBuffSz = FR_SINGLETON.generalProps.registerProps(cls,
        [FR_CONSTS.PROP_EST_BOUNDS_ON_START, FR_CONSTS.PROP_UNDO_BUF_SZ])
    cls.useDarkTheme = FR_SINGLETON.colorScheme.registerProp(cls, FR_CONSTS.SCHEME_USE_DARK_THEME)

  def __init__(self, parent=None, **quickLoaderArgs):
    super().__init__(parent)
    self.mainImg = FRMainImage()
    self.focusedImg = FRFocusedImage()
    self.focusedImg.toolsEditor.registerFunc(self.acceptFocusedRegion,
                                             btnOpts=FR_CONSTS.TOOL_ACCEPT_FOC_REGION)

    self.compMgr = FRComponentMgr()
    # Register exporter to allow user parameters
    ioCls = FR_SINGLETON.registerGroup(FR_CONSTS.CLS_COMP_EXPORTER)(FRComponentIO)
    ioCls.exportOnlyVis, ioCls.includeFullSourceImgName = \
      FR_SINGLETON.generalProps.registerProps(ioCls,
                                              [FR_CONSTS.EXP_ONLY_VISIBLE, FR_CONSTS.INCLUDE_FNAME_PATH]
                                              )
    self.compIo: FRComponentIO = ioCls()

    self.compTbl = FRCompTableView()
    self.compDisplay = FRCompDisplayFilter(self.compMgr, self.mainImg, self.compTbl)

    self.compTbl.setSortingEnabled(True)
    self.compTbl.setAlternatingRowColors(True)
    # Allow filtering/sorting
    self.sortFilterProxy = FRCompSortFilter(self.compMgr)
    self.compTbl.setModel(self.sortFilterProxy)

    self.hasUnsavedChanges = False
    self.srcImgFname: Optional[Path] = None
    self.autosaveTimer: Optional[QtCore.QTimer] = None

    # -----
    # INTERFACE WITH QUICK LOADER
    # -----
    self.appStateEditor = FRAppStateEditor(self, name='App State Editor')

    for plugin in FR_SINGLETON.plugins: # type: FRParamEditorPlugin
      plugin.attachS3aRef(self)

    def loadCfg(_fname: str):
      FR_SINGLETON.tableData.loadCfg(_fname)
      self.resetTblFields()

    def saveCfg(_folderName: Path):
      td = FR_SINGLETON.tableData
      saveFpath = td.cfgFname
      if not saveFpath.exists():
        saveFpath = _folderName/td.cfgFname.name
        saveToFile(td.cfg, saveFpath, allowOverwriteDefault=True)
      return str(saveFpath)
    self.appStateEditor.addImportExportOpts('tablecfg', loadCfg, saveCfg)

    self.appStateEditor.addImportExportOpts(
      'image', lambda fname: self.setMainImg(fname, clearExistingComps=False),
      lambda _folderName: str(self.srcImgFname)
    )
    def saveExistingComps(_folderName: Path):
      if self.mainImg.image is None:
        return None
      saveName = _folderName / 'savedState.pkl'
      self.exportCompList(saveName, readOnly=False)
      return str(saveName)
    loadExistingComps = lambda infile: self.loadCompList(infile, FR_ENUMS.COMP_ADD_AS_MERGE)
    self.appStateEditor.addImportExportOpts(
      'annotations', loadExistingComps, saveExistingComps)

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
    self.mainImg.imgItem.sigImageChanged.connect(lambda: self.clearBoundaries())
    self.mainImg.sigCompsCreated.connect(self.add_focusComps)

    def handleCompsChanged(changedDict: dict):
      focusedId = self.focusedImg.compSer[REQD_TBL_FIELDS.INST_ID]
      if focusedId in changedDict['deleted']:
        self.focusedImg.updateAll()
      elif focusedId in changedDict['changed']:
        self.changeFocusedComp(self.compMgr.compDf.loc[[focusedId]])
    self.compMgr.sigCompsChanged.connect(handleCompsChanged)

    # -----
    # COMPONENT TABLE
    # -----
    self.compDisplay.sigCompsSelected.connect(lambda newComps: self.changeFocusedComp(newComps))

    # -----
    # MISC
    # -----
    FR_SINGLETON.generalProps.sigParamStateUpdated.connect(self.updateUndoBuffSz)

    authorName = resolveAuthorName(quickLoaderArgs.pop('author', None))
    if authorName is None:
      sys.exit('No author name provided and no default author exists. Exiting.\n'
               'To start without error, provide an author name explicitly, e.g.\n'
               '"python -m s3a --author=<Author Name>"')
    FR_SINGLETON.tableData.annAuthor = authorName
    self.saveAllEditorDefaults()

  def resetTblFields(self):
    """
    When table fields change, the displayed columns must change and the view
    must be made aware. Ensure this occurs here
    """
    # Make sure this is necessary, first
    if self.compMgr.colTitles == list([f.name for f in FR_SINGLETON.tableData.allFields]):
      # Fields haven't changed since last reset. Types could be different, but nothing
      # will break. So, the table doesn't have to be completely reset
      return
    self.compMgr.beginResetModel()
    self.compMgr.resetFields()
    self.compMgr.rmComps()
    self.compMgr.endResetModel()
    self.compTbl.setColDelegates()
    self.compTbl.popup.tbl.setColDelegates()

  @staticmethod
  def saveAllEditorDefaults():
    for editor in FR_SINGLETON.registerableEditors:
      editor.saveCurStateAsDefault()

  @staticmethod
  def populateParamEditorMenuOpts(objForMenu: FRParamEditor, winMenu: QtWidgets.QMenu,
                                  triggerFn: Callable):
    addDirItemsToMenu(winMenu,
                      objForMenu.saveDir.glob(f'*.{objForMenu.fileType}'),
                      triggerFn)

  def startAutosave(self, interval_mins: float, autosaveFolder: Path, baseName: str):
    autosaveFolder.mkdir(exist_ok=True, parents=True)
    lastSavedDf = self.compMgr.compDf.copy()
    # Qtimer expects ms, turn mins->s->ms
    self.autosaveTimer = QtCore.QTimer()
    # Figure out where to start the counter
    globExpr = lambda: autosaveFolder.glob(f'{baseName}*.csv')
    existingFiles = list(globExpr())
    if len(existingFiles) == 0:
      counter = 0
    else:
      counter = max(map(lambda fname: int(fname.stem.rsplit('_')[1]), existingFiles)) + 1

    def save_incrementCounter():
      nonlocal counter, lastSavedDf
      baseSaveNamePlusFolder = autosaveFolder/f'{baseName}_{counter}.csv'
      counter += 1
      if not np.array_equal(self.compMgr.compDf, lastSavedDf):
        self.exportCompList(baseSaveNamePlusFolder)
        lastSavedDf = self.compMgr.compDf.copy()

    self.autosaveTimer.timeout.connect(save_incrementCounter)
    self.autosaveTimer.start(int(interval_mins*60*1000))

  def stopAutosave(self):
    self.autosaveTimer.stop()

  def updateUndoBuffSz(self, _genProps: Dict[str, Any]):
    FR_SINGLETON.actionStack.resizeStack(self.undoBuffSz)

  def estimateBoundaries(self):
    oldAct = self.mainImg.drawAction
    try:
      self.mainImg.drawAction = FR_CONSTS.DRAW_ACT_ADD
      verts = imgCornerVertices(self.mainImg.image)
      newComp = FR_SINGLETON.tableData.makeCompDf(1)
      newComp.at[REQD_TBL_FIELDS.INST_ID.value, REQD_TBL_FIELDS.VERTICES] = FRComplexVertices([verts])
      self.compMgr.addComps(newComp, emitChange=False)
      self.focusedImg.updateAll(self.mainImg.image, newComp.squeeze())
      for plugin in FR_SINGLETON.tableFieldPlugins:
        plugin.handleShapeFinished(verts)

      self.acceptFocusedRegion()
    finally:
      self.mainImg.drawAction = oldAct

  @FR_SINGLETON.actionStack.undoable('Accept Focused Region')
  def acceptFocusedRegion(self):
    """Applies the focused image vertices to the corresponding component in the table"""
    # If the component was deleted
    mgr = self.compMgr
    focusedId = self.focusedImg.compSer[REQD_TBL_FIELDS.INST_ID]
    if focusedId not in mgr.compDf.index:
      warn('Cannot accept region as this component was deleted.', FRS3AWarning)
      return
    oldSer = mgr.compDf.loc[focusedId].copy()

    for plugin in FR_SINGLETON.tableFieldPlugins:
      plugin.acceptChanges()

    modifiedComp = self.focusedImg.compSer
    modified_df = modifiedComp.to_frame().T
    mgr.addComps(modified_df, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)
    self.compDisplay.regionPlot.focusById([modifiedComp[REQD_TBL_FIELDS.INST_ID]])
    yield
    self.focusedImg.updateAll(self.mainImg.image, oldSer)
    mgr.addComps(oldSer.to_frame().T, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)

  def clearBoundaries(self):
    """Removes all components from the component table"""
    self.compMgr.rmComps()

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
    oldComps = self.compMgr.compDf.copy()
    if fileName is not None:
      fileName = Path(fileName).resolve()
    if clearExistingComps:
      self.compMgr.rmComps()
    if imgData is not None:
      self.mainImg.setImage(imgData)
    else:
      self.mainImg.setImage(fileName)
    self.srcImgFname = fileName
    self.focusedImg.updateAll()
    self.mainImg.plotItem.vb.autoRange()
    if self.estBoundsOnStart:
      self.estimateBoundaries()
    yield
    self.setMainImg(oldFile, oldData, clearExistingComps)
    if clearExistingComps:
      # Old comps were cleared, so put them back
      self.compMgr.addComps(oldComps)

  def exportCompList(self, outFname: Union[str, Path], readOnly=True, verifyIntegrity=True):
    self.compIo.exportByFileType(self.exportableDf, outFname, imShape=self.mainImg.image.shape,
                                 readOnly=readOnly, verifyIntegrity=verifyIntegrity)
    self.hasUnsavedChanges = False

  def exportLabeledImg(self, outFname: str=None):
    return self.compIo.exportByFileType(self.exportableDf, outFname, imShape=self.mainImg.image.shape)

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
    overwriteIdxs = exportDf[REQD_TBL_FIELDS.SRC_IMG_FILENAME] == FR_CONSTS.ANN_CUR_FILE_INDICATOR.value
    # TODO: Maybe the current filename will match the current file indicator. What happens then?
    exportDf.loc[overwriteIdxs, REQD_TBL_FIELDS.SRC_IMG_FILENAME] = srcImgFname
    return exportDf

  def loadCompList(self, inFname: str, loadType=FR_ENUMS.COMP_ADD_AS_NEW):
    pathFname = Path(inFname)
    if self.mainImg.image is None:
      raise FRIOError('Cannot load components when no main image is set.')
    fType = pathFname.suffix[1:]
    if not any(fType in typ for typ in self.compIo.handledIoTypes):
      raise FRIOError(f'Extension {fType} is not recognized. Must be one of:\n'
                      + self.compIo.handledIoTypes_fileFilter())
    newComps = self.compIo.buildByFileType(inFname, self.mainImg.image.shape)
    self.compMgr.addComps(newComps, loadType)

  def showModCompAnalytics(self):
    try:
      proc = self.focusedImg.currentPlugin.curProcessor
      proc.processor.stageSummary_gui()
    except AttributeError:
      # Processor or proc collection not set
      raise FRAlgProcessorError('Either no plugin is activated or the activated plugin'
                                ' has no processors')

  @FR_SINGLETON.actionStack.undoable('Create New Comp', asGroup=True)
  def add_focusComps(self, newComps: df):
    changeDict = self.compMgr.addComps(newComps)
    # Focus is performed by comp table
    # Arbitrarily choose the last possible component
    changeList = np.concatenate([changeDict['added'], changeDict['changed']])
    if len(changeList) == 0:
      return
    self.changeFocusedComp(self.compMgr.compDf.loc[[changeList[-1]]])

  @FR_SINGLETON.actionStack.undoable('Change Focused Component')
  def changeFocusedComp(self, newComps: df, forceKeepLastChange=False):
    oldSer = self.focusedImg.compSer.copy()
    oldImg = self.focusedImg.image
    if len(newComps) == 0:
      return
    # TODO: More robust scenario if multiple comps are in the dataframe
    #   For now, just use the last in the selection. This is so that if multiple
    #   components are selected in a row, the most recently selected is always
    #   the current displayed.
    newComp: pd.Series = newComps.iloc[-1,:]
    newCompId = newComp[REQD_TBL_FIELDS.INST_ID]
    self.compDisplay.regionPlot.focusById([newCompId])
    mainImg = self.mainImg.image
    self.focusedImg.updateAll(mainImg, newComp)
    # Nothing happened since the last component change, so just replace it instead of
    # adding a distinct action to the buffer queue
    stack = FR_SINGLETON.actionStack
    if not forceKeepLastChange and stack.undoDescr == 'Change Focused Component':
      stack.actions.pop()
    yield
    if oldImg is not None and len(oldSer.loc[REQD_TBL_FIELDS.VERTICES]) > 0:
      self.changeFocusedComp(oldSer.to_frame().T, forceKeepLastChange=True)
    else:
      self.focusedImg.updateAll()