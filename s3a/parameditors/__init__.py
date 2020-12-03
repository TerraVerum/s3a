from typing import List, Union, Type, Sequence, Dict

from pyqtgraph.Qt import QtWidgets, QtCore

from s3a.constants import GEN_PROPS_DIR, SCHEMES_DIR, BASE_DIR
from s3a.models.actionstack import ActionStack
from s3a.structures import FRParam
from .algcollection import AlgCtorCollection
from .genericeditor import ParamEditor, ParamEditorDockGrouping, ParamEditorPlugin, \
  TableFieldPlugin, dummyPluginCreator
from .shortcut import ShortcutsEditor
from .quickloader import QuickLoaderEditor
from .project import ProjectData
from ..generalutils import pascalCaseToTitle
from ..processing import ImgProcWrapper

Signal = QtCore.Signal

__all__ = ['FR_SINGLETON', 'ParamEditor', 'ParamEditorDockGrouping', 'TableFieldPlugin',
           'ParamEditorPlugin']

class AppSettingsEditor(ParamEditor):
  def __init__(self, parent=None):
    super().__init__(parent, paramList=[], saveDir=GEN_PROPS_DIR, fileType='genprops')


class ColorSchemeEditor(ParamEditor):
  def __init__(self, parent=None):
    super().__init__(parent, paramList=[], saveDir=SCHEMES_DIR, fileType='scheme')


class _FRSingleton(QtCore.QObject):
  sigPluginAdded = Signal(object) # List[QtWidgets.QDockWidget]

  _dummyEditor = ParamEditor(saveDir=None)
  """
  The first parameter editor will not be properly initialized unless a dummy is first in
  the list of existing editors. TODO: Maybe some additional logic can change this
  """
  def __init__(self, parent=None):
    super().__init__(parent)
    self.actionStack = ActionStack()
    self.clsToPluginMapping: Dict[Type[ParamEditorPlugin], ParamEditorPlugin] = {}

    self.project = ProjectData()
    self.tableData = self.project.tableData
    self.tableData.loadCfg(BASE_DIR/'tablecfg.yml')
    self.filter = self.tableData.filter


    self.generalProps = AppSettingsEditor()
    self.colorScheme = ColorSchemeEditor()
    self.shortcuts = ShortcutsEditor()
    self.quickLoader = QuickLoaderEditor()
    self.imgProcClctn = AlgCtorCollection(ImgProcWrapper)

    self.docks: List[QtWidgets.QDockWidget] = []
    self.addPlugin(dummyPluginCreator('General Properties', [self.generalProps, self.colorScheme]))
    self.addPlugin(dummyPluginCreator('Shortcuts', [self.shortcuts, self.quickLoader]))

  @property
  def registerableEditors(self):
    outList = []
    for editor in self.docks:
      if isinstance(editor, ParamEditorDockGrouping):
        outList.extend(editor.editors)
      else:
        outList.append(editor)
    return outList

  def registerGroup(self, groupParam: FRParam, **opts):
    def multiEditorClsDecorator(cls):
      # Since all legwork is done inside the editors themselves, simply call each decorator from here as needed
      editorList = self.registerableEditors
      if len(editorList) == 0:
        editorList.append(self._dummyEditor)
      for editor in editorList:
        cls = editor.registerGroup(groupParam, **opts)(cls)
      return cls
    return multiEditorClsDecorator

  def addPlugin(self, pluginCls: Type[ParamEditorPlugin], *args, **kwargs):
    """
    From a class inheriting the *FRParamEditorPlugin*, creates a plugin object
    that will appear in the S3A toolbar. An entry is created with dropdown options
    for each editor in *pluginCls*'s *editors* attribute.

    :param pluginCls: Class containing plugin actions
    :param args: Passed to class constructor
    :param kwargs: Passed to class constructor
    """
    nameToUse = pluginCls.name
    if nameToUse is None:
      nameToUse = pascalCaseToTitle(pluginCls.__name__)
    deco = self.registerGroup(FRParam(nameToUse))
    plugin: ParamEditorPlugin = deco(pluginCls)(*args, **kwargs)
    self.clsToPluginMapping[pluginCls] = plugin
    self.sigPluginAdded.emit(plugin)
    if plugin.dock is not None and plugin.dock not in self.docks:
      self.docks.append(plugin.dock)
    return plugin

  def close(self):
    for editor in self.registerableEditors:
      editor.close()


FR_SINGLETON = _FRSingleton()
