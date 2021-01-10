from __future__ import annotations

from typing import Optional, Callable, Sequence, Union

import pandas as pd
from pyqtgraph.Qt import QtWidgets

from s3a import parameditors as pe
from s3a.constants import PRJ_CONSTS
from s3a.structures import FRParam, S3AException, AlgProcessorError
from ..graphicsutils import create_addMenuAct, paramWindow
from ..parameditors import EditorPropsMixin
from ..processing import GeneralProcWrapper


class ParamEditorPlugin(EditorPropsMixin):
  """
  Primitive plugin which can interface with S3A functionality. When this class is overloaded,
  the child class is given a reference to the main S3A window and S3A is made aware of the
  plugin's existence. For interfacing with table fields, see the special case of
  :class:`TableFieldPlugin`
  """
  name: str=None
  """
  Name of this plugin as it should appear in the plugin menu
  """

  menu: QtWidgets.QMenu=None
  """
  Menu of additional options that should appear under this plugin
  """

  dock: Optional[pe.ParamEditorDockGrouping]
  """
  Docks that should be shown in S3A's menu bar. By default, just the toolsEditor is shown.
  If multiple param editors must be visible, manually set this property to a
  :class:`FRParamEditorDockGrouping` as performed in :class:`XYVerticesPlugin`.
  """
  toolsEditor: pe.ParamEditor
  """Param Editor window which holds user-editable properties exposed by the programmer"""

  _showFuncDetails=False
  """If *True*, a menu option will be added to edit parameters for functions that need them"""

  win = None
  """Reference to the application main window"""

  @property
  def parentMenu(self):
    """
    When this plugin is added, its options will be visible under a certain menu or toolbar. Where it is placed is
    determined by this value, which is usually the window's menu bar
    """
    return self.win.menuBar()

  @classmethod
  def __initEditorParams__(cls):
    cls.dock = pe.ParamEditorDockGrouping(dockName=cls.name)
    cls.toolsEditor = pe.ParamEditor.buildClsToolsEditor(cls)
    if cls._showFuncDetails:
      cls.dock.addEditors([cls.toolsEditor])
    cls.menu = QtWidgets.QMenu(cls.name)

  def __init__(self, *args, **kwargs):
    if self.dock is not None:
      self.dock.createMenuOpt(parentMenu=self.menu)
      self.menu.addSeparator()

  def registerFunc(self, func: Callable, submenuName:str=None, editor:pe.ParamEditor=None, **kwargs):
    """
    :param func: Function to register
    :param submenuName: If provided, this function is placed under a breakout menu with this name
    :param editor: If provided, the function is registered here instead of the plugin's tool editor
    :param ownerObj: Registered functions with associated shortcuts must be scoped
      to an owner object. This is not needed if no shortcut is associated with the button opts.
    :param kwargs: Forwarded to `ParamEditor.registerFunc`
    """
    if editor is None:
      editor = self.toolsEditor
    paramPath = []
    if submenuName is not None:
      paramPath.append(submenuName)
      parentMenu = None
      for act in self.menu.actions():
        if act.text() == submenuName and act.menu():
          parentMenu = act.menu()
          break
      if parentMenu is None:
        parentMenu = create_addMenuAct(editor, self.menu, submenuName, True)
        editor.params.addChild(dict(name=submenuName, type='group'))
    else:
      parentMenu = self.menu
    opts = kwargs.get('btnOpts', {})
    if isinstance(opts, FRParam): opts = opts.toPgDict()
    opts.setdefault('ownerObj', self)
    kwargs['btnOpts'] = opts

    proc = editor.registerFunc(func, **kwargs)

    if opts.get('guibtn', True):
      if 'name' in kwargs:
        actName = kwargs['name']
      elif 'name' in opts:
        actName = opts['name']
      else:
        actName = proc.name
      act = parentMenu.addAction(actName)
      act.triggered.connect(lambda: proc(win=self.win))
    return proc

  def registerPopoutFuncs(self, funcList: Sequence[Callable], nameList: Sequence[str]=None, groupName:str=None, btnOpts: FRParam=None):
    # TODO: I really don't like this. Consider any refactoring option that doesn't
    #   have an import inside a function
    from s3a import FR_SINGLETON
    if groupName is None and btnOpts is None:
      raise S3AException('Must provide either group name or button options')
    if groupName is None:
      groupName = btnOpts.name
    act = self.menu.addAction(groupName, lambda: paramWindow(self.toolsEditor.params.child(groupName)))
    act.click = act.triggered.emit
    FR_SINGLETON.shortcuts.registerShortcut(btnOpts, btnOpts, overrideOwnerObj=self, baseBtn=act)
    if nameList is None:
      nameList = [None]*len(funcList)
    for title, func in zip(nameList, funcList):
      self.toolsEditor.registerFunc(func, name=title, namePath=(groupName,))
    self.menu.addSeparator()


  def attachWinRef(self, win):
    self.win = win
    self.menu.setParent(self.parentMenu, self.menu.windowFlags())


def dummyPluginFactory(name_: str=None, editors: Sequence[pe.ParamEditor]=None):
  class DummyPlugin(ParamEditorPlugin):
    name = name_

    @classmethod
    def __initEditorParams__(cls):
      super().__initEditorParams__()
      if editors is not None:
        cls.dock.addEditors(editors)
  return DummyPlugin


class ProcessorPlugin(ParamEditorPlugin):
  procCollection: pe.algcollection.AlgParamEditor = None
  """
  Most table field plugins will use some sort of processor to infer field data.
  This property holds spawned collections. See :class:`XYVerticesPlugin` for
  an example.
  """

  @property
  def curProcessor(self):
    return self.procCollection.curProcessor

  @curProcessor.setter
  def curProcessor(self, newProcessor: Union[str, GeneralProcWrapper]):
    self.procCollection.switchActiveProcessor(newProcessor)

class TableFieldPlugin(ProcessorPlugin):
  focusedImg = None
  """
  Holds a reference to the focused image and set when the s3a reference is set. This
  is useful for most table field plugins, since focusedImg will hold a reference to the
  component series that is modified by the plugins.
  """

  _active=False

  @property
  def parentMenu(self):
    return self.win.tblFieldToolbar

  def attachWinRef(self, win):
    super().attachWinRef(win)
    self.focusedImg = focusedImg = win.focusedImg
    win.sigRegionAccepted.connect(self.acceptChanges)
    focusedImg.sigUpdatedFocusedComp.connect(self.updateFocusedComp)
    self.active = True
    self.registerFunc(self.processorAnalytics, btnOpts=PRJ_CONSTS.TOOL_PROC_ANALYTICS)

  def processorAnalytics(self):
    proc = self.curProcessor
    try:
      proc.processor.stageSummary_gui()
    except NotImplementedError:
      raise AlgProcessorError(f'Processor type {type(proc)} does not implement'
                              f' summary analytics.')


  def updateFocusedComp(self, newComp: pd.Series = None):
    """
    This function is called when a new component is created or the focused image is updated
    from the main view. See :meth:`MainImage.updateFocusedComp` for parameters.
    """
    pass

  def acceptChanges(self):
    """
    This must be overloaded by each plugin so the set component data is properly stored
    in the focused component. Essentially, any changes made by this plugin are saved
    after a call to this method.
    """
    raise NotImplementedError

  @property
  def active(self):
    """Whether this plugin is currently in use by the focused image."""
    return self._active

  @active.setter
  def active(self, newActive: bool):
    if newActive == self._active:
      return
    if newActive:
      self._onActivate()
    else:
      self._onDeactivate()
    self._active = newActive

  def _onActivate(self):
    """Overloaded by plugin classes to set up the plugin for use"""

  def _onDeactivate(self):
    """Overloaded by plugin classes to tear down when the plugin is no longer in use"""