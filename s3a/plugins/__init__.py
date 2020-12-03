from typing import Callable

from s3a import models
from s3a.constants import FR_CONSTS as FRC
from s3a.graphicsutils import menuFromEditorActions
from s3a.parameditors import ParamEditorPlugin, ParamEditor, FR_SINGLETON
from s3a.processing import AtomicProcess
from s3a.structures import FRParam
from s3a.views.imageareas import EditableImgBase
from .project import ProjectsPlugin
from .tablefield import VerticesPlugin


def funcForEditableImgPlugin(func: Callable, plugin: ParamEditorPlugin, editableImg: EditableImgBase, **kwargs):
  """See function signature for `FRParamEditor.registerFunc`"""
  origOpts = kwargs.pop('btnOpts', FRParam(''))
  origOpts.opts['ownerObj'] = editableImg

  proc = plugin.toolsEditor.registerFunc(func, **kwargs, btnOpts=origOpts)
  editableImg.toolsGrp.create_addBtn(origOpts,
                                     triggerFn=lambda *_args, **_kwargs: proc.run(),
                                     checkable=False, ownerObj=editableImg)


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
      self.toolsEditor.registerFunc(func, btnOpts=param)
    self.menu = tbl.menu = menuFromEditorActions([self.toolsEditor], menuParent=s3a)
    super().attachS3aRef(s3a)

ALL_PLUGINS = [VerticesPlugin, ProjectsPlugin, MainImagePlugin, CompTablePlugin]
