from __future__ import annotations

from typing import List, Type, Dict

from pyqtgraph.Qt import QtWidgets, QtCore
from utilitys import ActionStack
from utilitys import ParamEditor, ParamEditorDockGrouping, NestedProcWrapper, ParamEditorPlugin, dockPluginFactory

from s3a.constants import GEN_PROPS_DIR, SCHEMES_DIR, SHORTCUTS_DIR, CFG_DIR, IMG_PROC_DIR, MULT_PRED_DIR
from .algcollection import AlgCollection
from .quickloader import QuickLoaderEditor
from .table import TableData
from ..processing import ImgProcWrapper, ImageProcess
from utilitys.params import ShortcutParameter

Signal = QtCore.Signal


class _PrjSingleton(QtCore.QObject):
  sigPluginAdded = Signal(object) # List[QtWidgets.QDockWidget]

  def __init__(self, parent=None):
    super().__init__(parent)
    self.actionStack = ActionStack()
    self.clsToPluginMapping: Dict[Type[ParamEditorPlugin], ParamEditorPlugin] = {}

    self.tableData = TableData()
    self.filter = self.tableData.filter


    self.generalProps = ParamEditor(saveDir=GEN_PROPS_DIR, fileType='genprops',
                                    name='App Settings')
    self.colorScheme = ParamEditor(saveDir=SCHEMES_DIR, fileType='scheme',
                                   name='Color Scheme')
    self.shortcuts = ShortcutParameter.setRegistry(createIfNone=True, saveDir=SHORTCUTS_DIR)
    self.quickLoader = QuickLoaderEditor()
    self.imgProcClctn = AlgCollection(ImgProcWrapper, ImageProcess, saveDir=IMG_PROC_DIR)
    self.multiPredClctn = AlgCollection(saveDir=MULT_PRED_DIR)

    self.docks: List[QtWidgets.QDockWidget] = []
    self.addPlugin(dockPluginFactory('Settings', [self.generalProps, self.colorScheme]))
    self.addPlugin(dockPluginFactory('Shortcuts', [self.shortcuts, self.quickLoader]))

  @property
  def registerableEditors(self):
    outList = []
    for editor in self.docks:
      if isinstance(editor, ParamEditorDockGrouping):
        outList.extend(editor.editors)
      else:
        outList.append(editor)
    return outList

  def addPlugin(self, pluginCls: Type[ParamEditorPlugin], *args, **kwargs):
    """
    From a class inheriting the *PrjParamEditorPlugin*, creates a plugin object
    that will appear in the S3A toolbar. An entry is created with dropdown options
    for each editor in *pluginCls*'s *editors* attribute.

    :param pluginCls: Class containing plugin actions
    :param args: Passed to class constructor
    :param kwargs: Passed to class constructor
    """
    if pluginCls.name is not None and pluginCls.__groupingName__ is None:
      # The '&' character can be used in Qt to signal a shortcut, which is used internally
      # by plugins. When these plugins register shortcuts, the group name will contain
      # ampersands which shouldn't show up in human readable menus
      pluginCls.__groupingName__ = pluginCls.name.replace('&', '')
    plugin: ParamEditorPlugin = pluginCls(*args, **kwargs)
    self.clsToPluginMapping[pluginCls] = plugin
    self.sigPluginAdded.emit(plugin)
    if plugin.dock is not None and plugin.dock not in self.docks:
      self.docks.append(plugin.dock)
    return plugin

  def close(self):
    for editor in self.registerableEditors:
      editor.close()

INITIALIZED_GROUPINGS = set()
REGISTERED_GROUPINGS = set()
PRJ_SINGLETON = _PrjSingleton()