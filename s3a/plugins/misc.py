from __future__ import annotations
from typing import Callable

from PyQt5 import QtWidgets, QtCore

from s3a import ParamEditor, models, FR_CONSTS as FRC, FR_SINGLETON, RunOpts, \
  REQD_TBL_FIELDS as RTF, FRParam
from s3a.graphicsutils import menuFromEditorActions, ConsoleWidget
from s3a.parameditors import ParamEditorPlugin
from s3a.processing import AtomicProcess
from s3a.views.imageareas import EditableImgBase


class MainImagePlugin(ParamEditorPlugin):
  name = 'Main Image'
  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls, 'Tools')

  def __init__(self):
    super().__init__()

  def attachS3aRef(self, s3a: models.s3abase.S3ABase):
    mainImg = s3a.mainImg

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

    funcForEditableImgPlugin(mainImg.clearCurRoi, self, mainImg, btnOpts=FRC.TOOL_CLEAR_ROI)
    funcForEditableImgPlugin(startMove, self, mainImg, btnOpts=FRC.TOOL_MOVE_REGIONS)
    funcForEditableImgPlugin(startCopy, self, mainImg, btnOpts=FRC.TOOL_COPY_REGIONS)

    if not hasattr(s3a, 'compDisplay'):
      return

    tbl = s3a.compDisplay
    # Wrap in process to ignore the default param
    toRegister = AtomicProcess(tbl.mergeSelectedComps, ignoreKeys=['keepId'])
    funcForEditableImgPlugin(toRegister, self, mainImg, btnOpts=FRC.TOOL_MERGE_COMPS)
    funcForEditableImgPlugin(tbl.splitSelectedComps, self, mainImg, btnOpts=FRC.TOOL_SPLIT_COMPS)
    mainImg.setMenuFromEditors([self.toolsEditor])
    # No need for a dropdown menu
    self.dock = None
    super().attachS3aRef(s3a)


class CompTablePlugin(ParamEditorPlugin):
  name = 'Component Table'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls, 'Tools')
    cls.dock.addEditors([FR_SINGLETON.filter])


  def attachS3aRef(self, s3a: models.s3abase.S3ABase):

    tbl = s3a.compTbl
    for func, param in zip(
        [lambda: tbl.setSelectedCellsAs_gui(), tbl.removeSelectedRows_gui, tbl.setSelectedCellsAsFirst],
        [FRC.TOOL_TBL_SET_AS, FRC.TOOL_TBL_DEL_ROWS, FRC.TOOL_TBL_SET_SAME_AS_FIRST]):
      param.opts['ownerObj'] = s3a
      self.toolsEditor.registerFunc(func, btnOpts=param)
    self.menu = tbl.menu = menuFromEditorActions([self.toolsEditor], menuParent=s3a)
    super().attachS3aRef(s3a)


class MiscFunctionsPluginBase(ParamEditorPlugin):
  """
  Base is defined separate from actual misc funcs plugin so the same structure can be
  easily subclassed. Otherwise, remnant registered functions etc. will stick around
  and be associated with the class
  """
  name = 'Tools'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls, 'Function Details...')
    cls.dock.addEditors([cls.toolsEditor])
    cls.menu = QtWidgets.QMenu('Functions')

  def registerFunc(self, func: Callable, name:str=None, runOpts=RunOpts.BTN, category:str=None):
    """See function signature for `ParamEditor.registerFunc`"""
    paramPath = []
    if category is not None:
      paramPath.append(category)
    proc = self.toolsEditor.registerFunc(func, name, runOpts, paramPath=tuple(paramPath))
    act = self.menu.addAction(proc.name)
    act.triggered.connect(lambda: proc(s3a=self.s3a))
    return proc

class MiscFunctionsPlugin(MiscFunctionsPluginBase):
  def attachS3aRef(self, s3a: models.s3abase.S3ABase):
    super().attachS3aRef(s3a)

    self.registerFunc(s3a.showModCompAnalytics)
    self.registerFunc(self.showDevConsole)

  def showDevConsole(self):
    """
    Opens a console that allows dynamic interaction with current variables. If IPython
    is on your system, a qt console will be loaded. Otherwise, a (less capable) standard
    pyqtgraph console will be used.
    """
    namespace = dict(app=self.s3a, rtf=RTF, singleton=FR_SINGLETON)
    # "dict" default is to use repr instead of string for internal elements, so expanding
    # into string here ensures repr is not used
    nsPrintout = [f"{k}: {v}" for k, v in namespace.items()]
    text = f'Starting console with variables:\n' \
           f'{nsPrintout}'
    console = ConsoleWidget(parent=self.s3a, namespace=namespace, text=text)
    console.setWindowFlags(QtCore.Qt.Window)
    console.show()


def funcForEditableImgPlugin(func: Callable, plugin: ParamEditorPlugin, editableImg: EditableImgBase, **kwargs):
  """See function signature for `FRParamEditor.registerFunc`"""
  origOpts = kwargs.pop('btnOpts', FRParam(''))
  origOpts.opts['ownerObj'] = editableImg

  proc = plugin.toolsEditor.registerFunc(func, **kwargs, btnOpts=origOpts)
  editableImg.toolsGrp.create_addBtn(origOpts,
                                     triggerFn=lambda *_args, **_kwargs: proc.run(),
                                     checkable=False, ownerObj=editableImg)