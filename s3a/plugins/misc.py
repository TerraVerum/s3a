from __future__ import annotations

from typing import Callable, Sequence

import cv2 as cv
import numpy as np
import pandas as pd
from pyqtgraph import console as pg_console
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from utilitys import ParamEditorPlugin, ProcessIO, widgets as uw

from s3a import models, XYVertices, ComplexXYVertices
from s3a.constants import PRJ_CONSTS as CNST, REQD_TBL_FIELDS as RTF, PRJ_ENUMS
from s3a.models import s3abase
from s3a.parameditors import PRJ_SINGLETON
from s3a.plugins.base import ProcessorPlugin


class MainImagePlugin(ParamEditorPlugin):
  name = __groupingName__ = 'Main Image'
  _makeMenuShortcuts = False

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
        # Special case: Selection with point shape should be a point
        if self.win.mainImg.shapeCollection.curShapeParam == CNST.DRAW_SHAPE_POINT:
          verts = verts.mean(0, keepdims=True)
        disp.reflectSelectionBoundsMade(verts)

    self.registerFunc(disp.mergeSelectedComps, btnOpts=CNST.TOOL_MERGE_COMPS, ignoreKeys=['keepId'])
    self.registerFunc(disp.splitSelectedComps, btnOpts=CNST.TOOL_SPLIT_COMPS)
    win.mainImg.registerDrawAction([CNST.DRAW_ACT_ADD, CNST.DRAW_ACT_REM, CNST.DRAW_ACT_SELECT, CNST.DRAW_ACT_PAN],
                                 actHandler)
    self.registerFunc(disp.removeSelectedCompOverlap, btnOpts=CNST.TOOL_REM_OVERLAP)

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
    newComps = PRJ_SINGLETON.tableData.makeCompDf()
    newComps[RTF.VERTICES] = [verts]
    self.win.add_focusComps(newComps)

class CompTablePlugin(ParamEditorPlugin):
  name = 'Component Table'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.dock.addEditors([PRJ_SINGLETON.filter])


  def attachWinRef(self, win: s3abase.S3ABase):

    tbl = win.compTbl
    for func, param in zip(
        [lambda: tbl.setSelectedCellsAs_gui(), tbl.removeSelectedRows_gui, tbl.setSelectedCellsAsFirst,
         lambda: win.compDisplay.scaleViewboxToSelectedIds()],
        [CNST.TOOL_TBL_SET_AS, CNST.TOOL_TBL_DEL_ROWS, CNST.TOOL_TBL_SET_SAME_AS_FIRST,
         CNST.TOOL_TBL_ZOOM_TO_COMPS]):
      param.opts['ownerObj'] = win.compTbl
      self.registerFunc(func, name=param.name, btnOpts=param)
    tbl.menu = self.toolsEditor.actionsMenuFromProcs(parent=tbl, nest=True)
    super().attachWinRef(win)

class EditPlugin(ParamEditorPlugin):

  name = '&Edit'

  def attachWinRef(self, win: s3abase.S3ABase):
    super().attachWinRef(win)
    stack = PRJ_SINGLETON.actionStack

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
  _showFuncDetails = True

  def attachWinRef(self, win: s3abase.S3ABase):
    super().attachWinRef(win)

    self.registerFunc(self.showDevConsole)
    self.registerFunc(win.clearBoundaries, btnOpts=CNST.TOOL_CLEAR_BOUNDARIES)
    self.registerFunc(win.compDisplay.exportCompOverlay, name='Export Component Overlay', toClipboard=True)
    self.registerFunc(lambda: win.setMainImg(None), name='Clear Current Image')

  def showDevConsole(self):
    """
    Opens a console that allows dynamic interaction with current variables. If IPython
    is on your system, a qt console will be loaded. Otherwise, a (less capable) standard
    pyqtgraph console will be used.
    """
    namespace = dict(app=self.win, rtf=RTF, singleton=PRJ_SINGLETON)
    # "dict" default is to use repr instead of string for internal elements, so expanding
    # into string here ensures repr is not used
    nsPrintout = [f"{k}: {v}" for k, v in namespace.items()]
    text = f'Starting console with variables:\n' \
           f'{nsPrintout}'
    # Broad exception is fine, fallback is good enough. Too many edge cases to properly diagnose when Pycharm's event
    # loop is sync-able with the Jupyter dev console
    # noinspection PyBroadException
    try:
      console = uw.ConsoleWidget(parent=self.win, namespace=namespace, text=text)
    except Exception:
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
    self.registerFunc(self.iconAttributions)

  def iconAttributions(self):
    htmlStr = """
    <div>Icons made by <a href="https://www.freepik.com" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>

    <div>Icons made by <a href="https://www.flaticon.com/authors/those-icons" title="Those Icons">Those Icons</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>

    <div>Icons made by <a href="https://www.flaticon.com/authors/pixel-perfect" title="Pixel perfect">Pixel perfect</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
    """

    QtWidgets.QMessageBox.information(self.win, 'Icon Attributions', htmlStr)


def miscFuncsPluginFactory(name_: str=None, regFuncs: Sequence[Callable]=None, titles: Sequence[str]=None, showFuncDetails=False):
  class FuncContainerPlugin(ParamEditorPlugin):
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

  return FuncContainerPlugin


class MultiPredictionsPlugin(ProcessorPlugin):
  name = 'Multi-Predictions'

  mgr: models.tablemodel.ComponentMgr

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.procEditor = PRJ_SINGLETON.multiPredClctn.createProcessorEditor(cls, cls.name + ' Processor')
    cls.dock.addEditors([cls.procEditor])

  def __init__(self):
    super().__init__()
    self.registerFunc(self.lastRunAnalytics)

  def attachWinRef(self, win):
    super().attachWinRef(win)
    self.mgr = win.compMgr
    self.mainImg = win.mainImg
    win.mainImg.toolsEditor.registerFunc(self.predictFromSelection, btnOpts=CNST.TOOL_PRED_SEL)

  def makePrediction(self, comps: pd.DataFrame):
    if self.win.mainImg.image is None:
      return
    vbRange = np.array(self.mainImg.getViewBox().viewRange()).T
    newComps = self.curProcessor.run(components=comps, image=self.win.mainImg.image,
                                     viewbox=vbRange)
    if not isinstance(newComps, ProcessIO):
      newComps = ProcessIO(components=newComps)
    compsToAdd = newComps['components']
    addType = newComps.get('addType', PRJ_ENUMS.COMP_ADD_AS_NEW)
    if newComps.get('deleteOrig', False):
      self.mgr.rmComps(comps.index)
    self.mgr.addComps(compsToAdd, addType)


  def predictFromSelection(self):
    self.makePrediction(self.mgr.compDf.loc[self.win.compDisplay.selectedIds])

  def lastRunAnalytics(self):
    raise NotImplementedError