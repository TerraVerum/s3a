from __future__ import annotations

from typing import List, Type, Dict

from pyqtgraph.Qt import QtWidgets, QtCore

from s3a.constants import GEN_PROPS_DIR, SCHEMES_DIR, BASE_DIR
from s3a.models.actionstack import ActionStack
from s3a.plugins import base
from .algcollection import AlgCtorCollection
from .genericeditor import ParamEditor, ParamEditorDockGrouping, EditorPropsMixin
from .quickloader import QuickLoaderEditor
from .shortcut import ShortcutsEditor
from .table import TableData
from ..generalutils import pascalCaseToTitle, getAllBases
from ..processing import ImgProcWrapper, GeneralProcWrapper

Signal = QtCore.Signal


class _FRSingleton(QtCore.QObject):
  sigPluginAdded = Signal(object) # List[QtWidgets.QDockWidget]

  def __init__(self, parent=None):
    super().__init__(parent)
    self.actionStack = ActionStack()
    self.clsToPluginMapping: Dict[Type[base.ParamEditorPlugin], base.ParamEditorPlugin] = {}

    self.tableData = TableData()
    self.filter = self.tableData.filter


    self.generalProps = ParamEditor(saveDir=GEN_PROPS_DIR, fileType='genprops',
                                    name='App Settings')
    self.colorScheme = ParamEditor(saveDir=SCHEMES_DIR, fileType='scheme',
                                   name='Color Scheme')
    self.shortcuts = ShortcutsEditor()
    self.quickLoader = QuickLoaderEditor()
    self.imgProcClctn = AlgCtorCollection(ImgProcWrapper)
    self.globalPredClctn = AlgCtorCollection(GeneralProcWrapper)

    self.docks: List[QtWidgets.QDockWidget] = []
    self.addPlugin(base.dummyPluginFactory('&Settings', [self.generalProps, self.colorScheme]))
    self.addPlugin(base.dummyPluginFactory('Sho&rtcuts', [self.shortcuts, self.quickLoader]))

  @property
  def registerableEditors(self):
    outList = []
    for editor in self.docks:
      if isinstance(editor, ParamEditorDockGrouping):
        outList.extend(editor.editors)
      else:
        outList.append(editor)
    return outList

  def registerGroup(self, grpName: str=None):
    def deco(cls: type):
      nonlocal grpName
      if EditorPropsMixin not in getAllBases(cls):
        cls.__bases__ = (EditorPropsMixin,) + cls.__bases__
        if grpName is None:
          grpName = pascalCaseToTitle(cls.__name__)
      return cls
    return deco

  def addPlugin(self, pluginCls: Type[base.ParamEditorPlugin], *args, **kwargs):
    """
    From a class inheriting the *FRParamEditorPlugin*, creates a plugin object
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
    plugin: base.ParamEditorPlugin = pluginCls(*args, **kwargs)
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
FR_SINGLETON = _FRSingleton()