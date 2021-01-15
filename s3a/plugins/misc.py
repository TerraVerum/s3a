from __future__ import annotations

from typing import Callable, Sequence, List, Set

import cv2 as cv
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import console as pg_console
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from s3a import models, XYVertices, ComplexXYVertices
from s3a.constants import PRJ_CONSTS as CNST, REQD_TBL_FIELDS as RTF, PRJ_ENUMS
from s3a.controls.tableviewproxy import CompDisplayFilter
from s3a.graphicsutils import ConsoleWidget, menuFromEditorActions
from s3a.models import s3abase
from s3a.parameditors import FR_SINGLETON
from s3a.plugins.base import ParamEditorPlugin, ProcessorPlugin
from s3a.processing import ProcessIO


class MainImagePlugin(ParamEditorPlugin):
  name = __groupingName__ = 'Main Image'

  def __init__(self):
    super().__init__()

  def attachWinRef(self, win: s3abase.S3ABase):
    mainImg = win.mainImg

    copier = mainImg.regionCopier
    def startCopy():
      """
      Copies the selected components. They can be pasted by <b>double-clicking</b>
      on the destination location. When done copying, Click the *Clear ROI* tool change
      the current draw action.
      """
      copier.inCopyMode = True
      copier.sigCopyStarted.emit()
    def startMove():
      """
      Moves the selected components. They can be pasted by <b>double-clicking</b>
      on the destination location.
      """
      copier.inCopyMode = False
      copier.sigCopyStarted.emit()

    self.registerFunc(startMove, btnOpts=CNST.TOOL_MOVE_REGIONS)
    self.registerFunc(startCopy, btnOpts=CNST.TOOL_COPY_REGIONS)
    copier.sigCopyStopped.connect(win.mainImg.updateFocusedComp)


    disp = win.compDisplay
    # Wrap in process to ignore the default param

    def actHandler(verts, param):
      # When editing, only want to select if nothing is already selected
      if (param not in [CNST.DRAW_ACT_REM, CNST.DRAW_ACT_ADD]
          or  len(disp.selectedIds) == 0
      ):
          disp.reflectSelectionBoundsMade(verts)

    self.registerFunc(disp.mergeSelectedComps, btnOpts=CNST.TOOL_MERGE_COMPS, ignoreKeys=['keepId'])
    self.registerFunc(disp.splitSelectedComps, btnOpts=CNST.TOOL_SPLIT_COMPS)
    win.mainImg.registerDrawAction([CNST.DRAW_ACT_ADD, CNST.DRAW_ACT_REM, CNST.DRAW_ACT_SELECT, CNST.DRAW_ACT_PAN],
                                 actHandler)

    win.mainImg.registerDrawAction(CNST.DRAW_ACT_CREATE, self.createComponent)
    win.mainImg.addTools(self.toolsEditor)

    # No need for a dropdown menu
    self.dock = None
    super().attachWinRef(win)

  @property
  def image(self):
    return self.win.mainImg.image

  def createComponent(self, roiVerts: XYVertices):
    verts = np.clip(roiVerts.astype(int), 0, self.image.shape[:2][::-1])

    if cv.contourArea(verts) < self.win.mainImg.minCompSize:
      # Use as selection instead of creation
      self.win.compDisplay.reflectSelectionBoundsMade(roiVerts[[0]])
      return

    # noinspection PyTypeChecker
    verts = ComplexXYVertices([verts])
    newComps = FR_SINGLETON.tableData.makeCompDf()
    newComps[RTF.VERTICES] = [verts]
    self.win.add_focusComps(newComps)

class CompTablePlugin(ParamEditorPlugin):
  name = 'Component Table'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.dock.addEditors([FR_SINGLETON.filter])


  def attachWinRef(self, win: s3abase.S3ABase):

    tbl = win.compTbl
    for func, param in zip(
        [lambda: tbl.setSelectedCellsAs_gui(), tbl.removeSelectedRows_gui, tbl.setSelectedCellsAsFirst,
         lambda: win.compDisplay.scaleViewboxToSelectedIds()],
        [CNST.TOOL_TBL_SET_AS, CNST.TOOL_TBL_DEL_ROWS, CNST.TOOL_TBL_SET_SAME_AS_FIRST,
         CNST.TOOL_TBL_ZOOM_TO_COMPS]):
      param.opts['ownerObj'] = win.compTbl
      self.registerFunc(func, name=param.name, btnOpts=param)
    tbl.menu = menuFromEditorActions(self.toolsEditor, menuParent=tbl, nest=False)
    super().attachWinRef(win)

class EditPlugin(ParamEditorPlugin):

  name = '&Edit'

  def attachWinRef(self, win: s3abase.S3ABase):
    super().attachWinRef(win)
    stack = FR_SINGLETON.actionStack

    for param in CNST.TOOL_UNDO, CNST.TOOL_REDO: param.opts['ownerObj'] = win
    self.registerFunc(stack.undo, name='Undo', btnOpts=CNST.TOOL_UNDO)
    self.registerFunc(stack.redo, name='Redo', btnOpts=CNST.TOOL_REDO)

    def updateUndoRedoTxts():
      self.undoAct.setText(f'Undo: {stack.undoDescr}')
      self.redoAct.setText(f'Redo: {stack.redoDescr}')
    stack.stackChangedCallbacks.append(updateUndoRedoTxts)
    updateUndoRedoTxts()

  @property
  def undoAct(self):
      return [a for a in self.menu.actions() if a.text().startswith('Undo')][0]

  @property
  def redoAct(self):
    return [a for a in self.menu.actions() if a.text().startswith('Redo')][0]

class RandomToolsPlugin(ParamEditorPlugin):
  name = '&Tools'

  def attachWinRef(self, win: s3abase.S3ABase):
    super().attachWinRef(win)

    self.registerFunc(self.showDevConsole)
    self.registerFunc(win.clearBoundaries, btnOpts=CNST.TOOL_CLEAR_BOUNDARIES)

  def showDevConsole(self):
    """
    Opens a console that allows dynamic interaction with current variables. If IPython
    is on your system, a qt console will be loaded. Otherwise, a (less capable) standard
    pyqtgraph console will be used.
    """
    namespace = dict(app=self.win, rtf=RTF, singleton=FR_SINGLETON)
    # "dict" default is to use repr instead of string for internal elements, so expanding
    # into string here ensures repr is not used
    nsPrintout = [f"{k}: {v}" for k, v in namespace.items()]
    text = f'Starting console with variables:\n' \
           f'{nsPrintout}'
    try:
      console = ConsoleWidget(parent=self.win, namespace=namespace, text=text)
    except Exception as ex:
      # Ipy kernel can have issues for many different reasons. Always be ready to fall back to traditional console
      console = pg_console.ConsoleWidget(parent=self.win, namespace=namespace, text=text)
    console.setWindowFlags(QtCore.Qt.Window)
    console.show()

class HelpPlugin(ParamEditorPlugin):
  name = '&Help'

  def attachWinRef(self, win: QtWidgets.QMainWindow):
    super().attachWinRef(win)
    self.registerFunc(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl('https://gitlab.com/ficsresearch/s3a/-/wikis/home')),
                         name='Online User Guide')
    self.registerFunc(lambda: QtWidgets.QMessageBox.aboutQt(win, 'About Qt'), name='About Qt')

def miscFuncsPluginFactory(name_: str=None, regFuncs: Sequence[Callable]=None, titles: Sequence[str]=None, showFuncDetails=False):
  class DummyFuncsPlugin(ParamEditorPlugin):
    name = name_
    _showFuncDetails = showFuncDetails

    def attachWinRef(self, win: s3abase.S3ABase):
      super().attachWinRef(win)

      nonlocal regFuncs, titles
      if regFuncs is None:
        regFuncs = []
      if titles is None:
        titles = [None] * len(regFuncs)
      for func, title in zip(regFuncs, titles):
        self.registerFunc(func, title)

  return DummyFuncsPlugin


class GlobalPredictionsPlugin(ProcessorPlugin):
  name = 'Global Predictions'

  mgr: models.tablemodel.ComponentMgr

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.procCollection = FR_SINGLETON.globalPredClctn.createProcessorForClass(cls, cls.name + ' Processor')
    cls.dock.addEditors([cls.procCollection])

  def __init__(self):
    super().__init__()
    self.registerFunc(self.predictFromSelection, btnOpts=CNST.TOOL_PRED_SEL)
    self.registerFunc(self.lastRunAnalytics)

  def attachWinRef(self, win):
    super().attachWinRef(win)
    self.mgr = win.compMgr
    self.focusedImg = win.focusedImg
    win.focusedImg.toolsEditor.registerFunc(self.predictFromSelection)

  def makePrediction(self, comps: pd.DataFrame):
    if self.win.mainImg.image is None:
      return
    newComps = self.curProcessor.run(components=comps, image=self.win.mainImg.image)
    if isinstance(newComps, ProcessIO):
      newComps = newComps['components']
    self.mgr.addComps(newComps)

  def predictFromSelection(self):
    self.makePrediction(self.mgr.compDf.loc[self.win.compDisplay.selectedIds])

  def lastRunAnalytics(self):
    raise NotImplementedError

class FindDuplicatesPlugin(ProcessorPlugin):
  name = 'Find Duplicates'

  mgr: models.tablemodel.ComponentMgr

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.procCollection = FR_SINGLETON.globalPredClctn.createProcessorForClass(cls, cls.name + ' Processor')
    cls.dock.addEditors([cls.procCollection])

  def __init__(self):
    super().__init__()
    self.idGroupings: List[Set[int]] = []

    self.registerFunc(self.predictFromSelection, btnOpts=CNST.TOOL_PRED_SEL)
    self.registerFunc(lambda: self.deleteGroupsWithIds(self.win.compDisplay.selectedIds), btnOpts=CNST.TOOL_PRED_DEL_GRP)
    self.registerFunc(self.lastRunAnalytics)

  def attachWinRef(self, win: models.s3abase.S3ABase):
    super().attachWinRef(win)
    def onChange(changeDict):
      deleted = set(changeDict['deleted'])
      for grouping in self.idGroupings:
        grouping -= deleted
    win.compMgr.sigCompsChanged.connect(onChange)
    self.mgr = win.compMgr
    self.focusedImg = win.focusedImg
    win.sigRegionAccepted.connect(self.acceptChanges)

  def acceptChanges(self):
    origVerts = self.focusedImg.compSer[RTF.VERTICES].copy()
    origOffset = origVerts.stack().min(0)
    # Account for margin in focused image
    for verts in origVerts:
      verts -= origOffset
    origId = self.focusedImg.compSer[RTF.INST_ID]
    df = self.mgr.compDf
    modifiedCompIds = self._groupsWithIds([origId])
    if len(modifiedCompIds) == 0:
      return
    toModifyIdxs = df.index.isin(modifiedCompIds)
    toModifyVerts = df.loc[toModifyIdxs, RTF.VERTICES]
    allNewVerts = []
    for complexVerts in toModifyVerts: # type: ComplexXYVertices
      offset = complexVerts.stack().min(0)
      newVerts = []
      for verts in origVerts:
        newVerts.append(verts + offset)
      allNewVerts.append(ComplexXYVertices(newVerts))
    toAddDf = pd.DataFrame(np.c_[toModifyIdxs, allNewVerts], columns=[RTF.INST_ID, RTF.VERTICES])
    toAddDf = toAddDf.set_index(RTF.INST_ID, drop=False)
    self.mgr.addComps(toAddDf, PRJ_ENUMS.COMP_ADD_AS_MERGE)

  def _groupsWithIds(self, idList: Sequence[int]):
    allIds = []
    for grouping in self.idGroupings:
      if any(id_ in grouping for id_ in idList):
        allIds.extend(grouping)
    return allIds

  def makePrediction(self, comps: pd.DataFrame):
    if self.win.mainImg.image is None:
      return
    result = self.curProcessor.run(components=comps, image=self.win.mainImg.image)
    newComps = result['components']
    with FR_SINGLETON.actionStack.group('Make Prediction'):
      changeList = self.mgr.addComps(newComps)['added']
      deleteOrigs = result.get('deleteOrig', True)
      if deleteOrigs:
        self.mgr.rmComps(comps.index)
    origId = self.focusedImg.compSer[RTF.INST_ID]
    if result.get('asGroup', False):
      self.idGroupings.append(set(changeList + [origId]))

  def predictFromSelection(self):
    self.makePrediction(self.mgr.compDf.loc[self.win.compDisplay.selectedIds])

  def deleteGroupsWithIds(self, idList: Sequence[int]):
    toDelete = self._groupsWithIds(idList)
    self.mgr.rmComps(toDelete)

  def lastRunAnalytics(self):
    analyticsWin = self._buildAnalyticsLayout()
    scores = self.curProcessor.processor.result['scores']
    lastIds = np.asarray(list(self.idGroupings[-1]))
    keeps = np.isin(lastIds, self.mgr.compDf.index)
    lastIds = lastIds[keeps]
    scores = scores[keeps]
    lastComps = self.mgr.compDf.loc[lastIds].copy()
    lastComps[FR_SINGLETON.tableData.fieldFromName('Notes')] = scores
    analyticsWin.compMgr.addComps(lastComps)

    analyticsWin.mainImg.setImage(self.win.mainImg.image)
    tmImgOverlay = pg.ImageItem(self.curProcessor.processor.result['matchImg'])
    tmImgOverlay.setOpacity(0.75)
    verts =  lastComps.loc[lastComps.index[0], RTF.VERTICES].stack()
    templateSize = verts.max(0) - verts.min(0)
    tmImgOverlay.setPos(templateSize[0]//2, templateSize[1]//2)
    analyticsWin.mainImg.addItem(tmImgOverlay)

    analyticsWin.show()
    analyticsWin.exec_()

  @staticmethod
  def _buildAnalyticsLayout():
    from s3a.models.tablemodel import ComponentMgr
    from s3a.views.imageareas import MainImage
    from s3a.views.tableview import CompTableView
    from s3a.controls.tableviewproxy import CompSortFilter
    parent = QtWidgets.QDialog()
    parent.setModal(True)
    lay = QtWidgets.QVBoxLayout()
    parent.setLayout(lay)

    scoreHoverLbl = parent.scoreHoverLbl = QtWidgets.QLabel()
    parent.compMgr = newMgr = ComponentMgr()
    parent.mainImg = newImg = MainImage()
    newImg.drawActGrp.paramToBtnMapping[CNST.DRAW_ACT_ADD].hide()
    newTbl = CompTableView(minimal=True)
    newTbl.setModel(CompSortFilter(newMgr))
    cols = [RTF.INST_ID, FR_SINGLETON.tableData.fieldFromName('Notes')]
    for ii, col in enumerate(FR_SINGLETON.tableData.allFields):
      if col not in cols:
        newTbl.setColumnHidden(ii, True)
    newMgr.beginResetModel()
    newMgr.colTitles[newMgr.colTitles.index('Notes')] = 'Score'
    newMgr.endResetModel()

    lay.addWidget(scoreHoverLbl)
    lay.addWidget(newImg.widgetContainer())
    newImg.toolsGrp.hide()
    lay.addWidget(newTbl)
    parent.compDisplay = CompDisplayFilter(newMgr, newImg, newTbl)
    return parent