import sys
from pathlib import Path
from typing import Optional, Union, Callable, Dict, Any
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets

from s3a.controls.tableviewproxy import FRCompDisplayFilter, FRCompSortFilter
from s3a.generalutils import resolveAuthorName
from s3a.graphicsutils import addDirItemsToMenu
from s3a.models.tablemodel import FRComponentIO, FRComponentMgr
from s3a.projectvars import FR_CONSTS, FR_ENUMS, REQD_TBL_FIELDS
from s3a.structures import FRS3AWarning, FRVertices, FilePath, NChanImg, FRAppIOError, \
  FRAlgProcessorError
from s3a.views.imageareas import FRMainImage, FRFocusedImage, FREditableImgBase
from s3a.views.parameditors import FRParamEditor
from s3a.views.parameditors import FR_SINGLETON
from s3a.views.parameditors.appstate import FRAppStateEditor
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
    cls.useDarkTheme = FR_SINGLETON.generalProps.registerProp(cls, FR_CONSTS.SCHEME_USE_DARK_THEME)

  def __init__(self, parent=None, **quickLoaderArgs):
    super().__init__(parent)
    self.mainImg = FRMainImage()
    self.focusedImg = FRFocusedImage()
    self.compMgr = FRComponentMgr()
    self.compIo = FRComponentIO()
    self.compTbl = FRCompTableView()
    self.compDisplay = FRCompDisplayFilter(self.compMgr, self.mainImg, self.compTbl)

    self.compTbl.setSortingEnabled(True)
    self.compTbl.setAlternatingRowColors(True)
    # Allow filtering/sorting
    self.sortFilterProxy = FRCompSortFilter(self.compMgr)
    self.compTbl.setModel(self.sortFilterProxy)

    self.hasUnsavedChanges = False
    self.srcImgFname = None
    self.autosaveTimer: Optional[QtCore.QTimer] = None

    # -----
    # INTERFACE WITH QUICK LOADER
    # -----
    self.appStateEditor = FRAppStateEditor(self, name='App State Editor')
    self.appStateEditor.addImportExportOpts(
      'image', lambda fname: self.setMainImg(fname, clearExistingComps=False),
      lambda: str(self.srcImgFname)
    )
    def saveExistingComps():
      if self.mainImg.image is None:
        return None
      saveName = self.appStateEditor.saveDir / 'savedState.csv'
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
    self.mainImg.sigCompsCreated.connect(self.add_focusComp)
    self.mainImg.sigCompsRemoved.connect(
      lambda compIds: self.compMgr.rmComps(compIds)
    )

    # -----
    # COMPONENT TABLE
    # -----
    self.compDisplay.sigCompsSelected.connect(lambda newComps: self.changeFocusedComp(newComps))

    def handleUpdate(comps):
      self.compMgr.addComps(comps, FR_ENUMS.COMP_ADD_AS_MERGE)
      self.changeFocusedComp(comps)
    self.mainImg.sigCompsUpdated.connect(handleUpdate)

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

  def clearFocusedRegion(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.focusedImg.image is None:
      return
    self.focusedImg.updateRegionFromVerts(None)

  def resetFocusedRegion(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.focusedImg.image is None:
      return
    self.focusedImg.updateRegionFromVerts(self.focusedImg.compSer[REQD_TBL_FIELDS.VERTICES])

  @FR_SINGLETON.actionStack.undoable('Accept Focused Region')
  def acceptFocusedRegion(self):
    # If the component was deleted
    focusedId = self.focusedImg.compSer[REQD_TBL_FIELDS.INST_ID]
    if focusedId not in self.compMgr.compDf.index:
      warn('Cannot accept region as this component was deleted.', FRS3AWarning)
      return
    oldSer = self.compMgr.compDf.loc[focusedId].copy()

    self.focusedImg.saveNewVerts()
    modifiedComp = self.focusedImg.compSer
    modified_df = modifiedComp.to_frame().T
    self.compMgr.addComps(modified_df, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)
    self.compDisplay.regionPlot.focusById([modifiedComp[REQD_TBL_FIELDS.INST_ID]])
    yield
    self.focusedImg.saveNewVerts(oldSer[REQD_TBL_FIELDS.VERTICES])
    self.compMgr.addComps(oldSer.to_frame().T, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)

  def estimateBoundaries(self):
    oldAct = self.mainImg.drawAction
    try:
      self.mainImg.drawAction = FR_CONSTS.DRAW_ACT_ADD
      self.mainImg.handleShapeFinished(FRVertices())
    except Exception as ex:
      raise
    finally:
      self.mainImg.drawAction = oldAct

  def clearBoundaries(self):
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
    self.focusedImg.resetImage()
    self.mainImg.plotItem.vb.autoRange()
    if self.estBoundsOnStart:
      self.estimateBoundaries()
    yield
    self.setMainImg(oldFile, oldData, clearExistingComps)
    if clearExistingComps:
      # Old comps were cleared, so put them back
      self.compMgr.addComps(oldComps)

  def exportCompList(self, outFname: Union[str, Path], readOnly=True, verifyIntegrity=True):
    self.compIo.prepareDf(self.compMgr.compDf, self.compDisplay.displayedIds,
                          self.srcImgFname)
    self.compIo.exportByFileType(outFname, imShape=self.mainImg.image.shape,
                                 readOnly=readOnly, verifyIntegrity=verifyIntegrity)
    self.hasUnsavedChanges = False

  def exportLabeledImg(self, outFname: str=None):
    self.compIo.prepareDf(self.compMgr.compDf, self.compDisplay.displayedIds)
    return self.compIo.exportByFileType(outFname, imShape=self.mainImg.image.shape)

  def loadCompList(self, inFname: str, loadType=FR_ENUMS.COMP_ADD_AS_NEW):
    pathFname = Path(inFname)
    if self.mainImg.image is None:
      raise FRAppIOError('Cannot load components when no main image is set.')
    fType = pathFname.suffix[1:]
    if not any(fType in typ for typ in self.compIo.handledIoTypes):
      raise FRAppIOError(f'Extension {fType} is not recognized. Must be one of:\n'
                         + self.compIo.handledIoTypes_fileFilter())
    newComps = self.compIo.buildByFileType(inFname, self.mainImg.image.shape)
    self.compMgr.addComps(newComps, loadType)

  def showNewCompAnalytics(self):
    self._check_plotStages(self.mainImg)

  def showModCompAnalytics(self):
    self._check_plotStages(self.focusedImg)

  @FR_SINGLETON.actionStack.undoable('Create New Comp', asGroup=True)
  def add_focusComp(self, newComps: df):
    self.compMgr.addComps(newComps)
    # Make sure index matches ID before updating current component
    newComps = newComps.set_index(REQD_TBL_FIELDS.INST_ID, drop=False)
    # Focus is performed by comp table
    self.changeFocusedComp(newComps)

  @staticmethod
  def _check_plotStages(img: FREditableImgBase):
    proc = img.curProcessor.processor
    if proc.result is None:
      raise FRAlgProcessorError('Analytics can only be shown after the algorithm'
                                ' was run.')
    proc.plotStages(ignoreDuplicateResults=True)

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
    self.curCompIdLbl.setText(f'Component ID: {newCompId}')
    # Nothing happened since the last component change, so just replace it instead of
    # adding a distinct action to the buffer queue
    stack = FR_SINGLETON.actionStack
    if not forceKeepLastChange and stack.undoDescr == 'Change Focused Component':
      stack.actions.pop()
    yield
    if oldImg is not None and len(oldSer.loc[REQD_TBL_FIELDS.VERTICES]) > 0:
      self.changeFocusedComp(oldSer.to_frame().T, forceKeepLastChange=True)
    else:
      self.focusedImg.resetImage()