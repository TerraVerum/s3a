from __future__ import annotations

from utilitys.params.parameditor import RunOpts

import sys
from typing import Callable, Sequence

import cv2 as cv
import numpy as np
import pandas as pd
from pyqtgraph import console as pg_console
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from s3a import models, XYVertices, ComplexXYVertices
from s3a.constants import PRJ_CONSTS as CNST, REQD_TBL_FIELDS as RTF, PRJ_ENUMS, GEN_PROPS_DIR, SCHEMES_DIR, \
  SHORTCUTS_DIR
from s3a.logger import getAppLogger
from s3a.models import s3abase
from s3a.parameditors.quickloader import QuickLoaderEditor
from s3a.plugins.base import ProcessorPlugin
from s3a.shared import SharedAppSettings
from utilitys import ParamEditorPlugin, ProcessIO, widgets as uw, ParamEditor, ParamContainer, ShortcutParameter


class MainImagePlugin(ParamEditorPlugin):
  name = __groupingName__ = 'Main Image'
  _makeMenuShortcuts = False

  def __initEditorParams__(self, shared: SharedAppSettings, **kwargs):
    self.props = ParamContainer()
    shared.generalProps.registerProp(CNST.PROP_MIN_COMP_SZ, container=self.props)
    self.tableData = shared.tableData
    super().__initEditorParams__(shared=shared, **kwargs)

  def attachWinRef(self, win: s3abase.S3ABase):
    self._hookupCopier(win)
    self._hookupDrawActions(win)
    self._hookupSelectionTools(win)

    win.mainImg.addTools(self.toolsEditor)
    # No need for a dropdown menu
    self.dock = None
    super().attachWinRef(win)

  def _hookupDrawActions(self, win):
    disp = win.compDisplay
    def actHandler(verts, param):
      # When editing, only want to select if nothing is already started
      if (param not in [CNST.DRAW_ACT_REM, CNST.DRAW_ACT_ADD]
          or len(self.win.vertsPlg.region.regionData) == 0
      ):
        # Special case: Selection with point shape should be a point
        if self.win.mainImg.shapeCollection.curShapeParam == CNST.DRAW_SHAPE_POINT:
          verts = verts.mean(0, keepdims=True)
        # Make sure to check vertices plugin regions since they suppress disp's regions for focused ids
        # Must be done first, since a no-find in disp's regions will deselect them
        with uw.makeDummySignal(win.compTbl, 'sigSelectionChanged'):
          # Second call should handle the true selection signal
          disp.reflectSelectionBoundsMade(verts, self.win.vertsPlg.region)
          disp.reflectSelectionBoundsMade(verts, clearExisting=False)

        nonUniqueIds = win.compTbl.ids_rows_colsFromSelection(excludeNoEditCols=False, warnNoneSelection=False)[:,0]
        selection = pd.unique(nonUniqueIds)
        win.compTbl.sigSelectionChanged.emit(selection)
    acts = [CNST.DRAW_ACT_ADD, CNST.DRAW_ACT_REM, CNST.DRAW_ACT_SELECT, CNST.DRAW_ACT_PAN]
    win.mainImg.registerDrawAction(acts, actHandler)
    win.mainImg.registerDrawAction(CNST.DRAW_ACT_CREATE, self.createComponent)

  def _hookupCopier(self, win):
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

  def _hookupSelectionTools(self, win):
    disp = win.compDisplay
    self.registerFunc(disp.mergeSelectedComps, btnOpts=CNST.TOOL_MERGE_COMPS, ignoreKeys=['keepId'])
    self.registerFunc(disp.splitSelectedComps, btnOpts=CNST.TOOL_SPLIT_COMPS)
    self.registerFunc(disp.removeSelectedCompOverlap, btnOpts=CNST.TOOL_REM_OVERLAP)

  @property
  def image(self):
    return self.win.mainImg.image

  def createComponent(self, roiVerts: XYVertices):
    verts = np.clip(roiVerts.astype(int), 0, self.image.shape[:2][::-1])

    if cv.contourArea(verts) < self.props[CNST.PROP_MIN_COMP_SZ]:
      # Use as selection instead of creation
      self.win.compDisplay.reflectSelectionBoundsMade(roiVerts[[0]])
      return

    # noinspection PyTypeChecker
    verts = ComplexXYVertices([verts])
    newComps = self.tableData.makeCompDf()
    newComps[RTF.VERTICES] = [verts]
    self.win.add_focusComps(newComps)

class CompTablePlugin(ParamEditorPlugin):
  name = 'Component Table'

  def __initEditorParams__(self, shared: SharedAppSettings):
    super().__initEditorParams__()
    self.dock.addEditors([shared.filter])

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
    self.tableData = win.sharedAttrs.tableData

class EditPlugin(ParamEditorPlugin):

  name = 'Edit'

  def attachWinRef(self, win: s3abase.S3ABase):
    super().attachWinRef(win)
    stack = win.sharedAttrs.actionStack

    self.registerFunc(stack.undo, name='Undo', btnOpts=CNST.TOOL_UNDO)
    self.registerFunc(stack.redo, name='Redo', btnOpts=CNST.TOOL_REDO)

    def updateUndoRedoTxts(_action=None):
      self.undoAct.setText(f'Undo: {stack.undoDescr}')
      self.redoAct.setText(f'Redo: {stack.redoDescr}')
    stack.stackChangedCallbacks.append(updateUndoRedoTxts)

    def showStatus(action):
      # Since this was the *already performed* action, what it reports is the opposite of what happens
      if action is None:
        return
      if action.treatAsUndo:
        msg = f'{stack.undoDescr}'
      else:
        msg = f'Undid {stack.redoDescr}'
      getAppLogger(__name__).info(msg)
    stack.stackChangedCallbacks.append(showStatus)

    updateUndoRedoTxts()

  @property
  def undoAct(self):
      return [a for a in self.menu.actions() if a.text().startswith('Undo')][0]

  @property
  def redoAct(self):
    return [a for a in self.menu.actions() if a.text().startswith('Redo')][0]

class RandomToolsPlugin(ParamEditorPlugin):
  name = 'Tools'
  _showFuncDetails = True

  def attachWinRef(self, win: s3abase.S3ABase):
    super().attachWinRef(win)

    self.registerFunc(self.showDevConsole_gui, name='Show Dev Console')
    self.registerFunc(win.clearBoundaries, btnOpts=CNST.TOOL_CLEAR_BOUNDARIES)
    self.registerFunc(win.compDisplay.exportCompOverlay, name='Export Component Overlay', toClipboard=True)
    self.registerFunc(lambda: win.setMainImg(None), name='Clear Current Image')

    self._hookupFieldDisplay(win)

  def _hookupFieldDisplay(self, win):
    display = win.compDisplay
    # This option shouldn't show in the menu dropdown, so register directly to the tools
    _, param = self.toolsEditor.registerFunc(
      display.fieldInfoProc,
      name='Show Field Info',
      returnParam=True,
      runOpts=RunOpts.ON_CHANGED,
    )

    # There should also be an option that *does* show in the menu, which displays field info
    # for every component
    def toggleAll():
      if display.fieldDisplay.inUseDelegates:
        display.fieldDisplay.callDelegateFunc('clear')
      else:
        display.fieldInfoProc(ids=win.compMgr.compDf.index, force=True)
    self.registerFunc(toggleAll, name='Toggle All Field Info')
    
    fieldsParam = param.child('fields')
    def updateLims():
      fieldsParam.setLimits(
        [str(f) for f in win.sharedAttrs.tableData.allFields
          if f not in display.fieldDisplay.ignoreCols]
      )
      fieldsParam.setValue(fieldsParam.opts['limits'])

    win.sharedAttrs.tableData.sigCfgUpdated.connect(updateLims)
    updateLims()

  def showDevConsole_gui(self):
    """
    Opens a console that allows dynamic interaction with current variables. If IPython
    is on your system, a qt console will be loaded. Otherwise, a (less capable) standard
    pyqtgraph console will be used.
    """
    namespace = dict(app=self.win, rtf=RTF)
    # "dict" default is to use repr instead of string for internal elements, so expanding
    # into string here ensures repr is not used
    nsPrintout = [f"{k}: {v}" for k, v in namespace.items()]
    text = f'Starting console with variables:\n' \
           f'{nsPrintout}'
    # Broad exception is fine, fallback is good enough. Too many edge cases to properly diagnose when Pycharm's event
    # loop is sync-able with the Jupyter dev console
    # noinspection PyBroadException
    try:
      # See https://intellij-support.jetbrains.com/hc/en-us/community/posts/205819799/comments/206004059
      # for detecting whether this is run in debug mode. PyCharm among other IDEs crash trying to spawn a jupyter console
      # without a stack trace, so attempt to catch this situation early
      if sys.gettrace() is None:
        console = uw.ConsoleWidget(parent=self.win, namespace=namespace, text=text)
      else:
        # Raising an error goes into the except clause
        raise RuntimeError('Cannot spawn Jupyter console in a debug environment')
    except Exception:
      # Ipy kernel can have issues for many different reasons. Always be ready to fall back to traditional console
      console = pg_console.ConsoleWidget(parent=self.win, namespace=namespace, text=text)
    console.setWindowFlags(QtCore.Qt.WindowType.Window)
    console.show()

class HelpPlugin(ParamEditorPlugin):
  name = 'Help'

  def attachWinRef(self, win: QtWidgets.QMainWindow):
    super().attachWinRef(win)
    self.registerFunc(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl('https://gitlab.com/ficsresearch/s3a/-/wikis/docs/user\'s-guide')),
                         name='Online User Guide')
    self.registerFunc(lambda: QtWidgets.QMessageBox.aboutQt(win, 'About Qt'), name='About Qt')
    self.registerFunc(self.iconAttributions_gui, name='Icon Attributions')

  def iconAttributions_gui(self):
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

  def __initEditorParams__(self, shared: SharedAppSettings):
    super().__initEditorParams__()
    self.procEditor = shared.multiPredClctn.createProcessorEditor(type(self), self.name + ' Processor')
    self.dock.addEditors([self.procEditor])

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
    self.mgr.addComps(compsToAdd, addType)


  def predictFromSelection(self):
    selectedIds = self.win.compDisplay.selectedIds
    if len(selectedIds) == 0:
      return
    self.makePrediction(self.mgr.compDf.loc[selectedIds])

  def lastRunAnalytics(self):
    raise NotImplementedError

class SettingsPlugin(ParamEditorPlugin):
  name = 'Settings'

  def __initEditorParams__(self, **kwargs):
    super().__initEditorParams__(**kwargs)

    self.generalProps = ParamEditor(saveDir=GEN_PROPS_DIR, fileType='genprops',
                                    name='App Settings')
    self.colorScheme = ParamEditor(saveDir=SCHEMES_DIR, fileType='scheme',
                                   name='Color Scheme')

    self.dock.addEditors([self.generalProps, self.colorScheme])


class ShortcutsPlugin(ParamEditorPlugin):
  name = 'Shortcuts'

  def __initEditorParams__(self, **kwargs):
    super().__initEditorParams__(**kwargs)

    self.shortcuts = ShortcutParameter.setRegistry(createIfNone=True, saveDir=SHORTCUTS_DIR)
    self.quickLoader = QuickLoaderEditor()

    self.dock.addEditors([self.shortcuts, self.quickLoader])

