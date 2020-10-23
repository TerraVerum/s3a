import weakref
from typing import List, Union, Type

from pyqtgraph.Qt import QtWidgets, QtCore

from s3a.constants import GEN_PROPS_DIR, SCHEMES_DIR, BASE_DIR
from s3a.models.actionstack import ActionStack
from s3a.structures import FRParam
from .genericeditor import ParamEditor, ParamEditorDockGrouping, ParamEditorPlugin, \
  TableFieldPlugin
from .algcollection import AlgCtorCollection
from .project import ProjectEditor
from .quickloader import QuickLoaderEditor
from .shortcut import ShortcutsEditor
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
  sigDocksAdded = Signal(object) # List[QtWidgets.QDockWidget]

  def __init__(self, parent=None):
    super().__init__(parent)
    self.actionStack = ActionStack()
    self.plugins: List[ParamEditorPlugin] = []
    self.tableFieldPlugins: List[TableFieldPlugin] = []

    self.project = ProjectEditor()
    self.tableData = self.project.projectData.tableData
    self.tableData.loadCfg(BASE_DIR/'tablecfg.yml')
    self.filter = self.tableData.filter


    self.shortcuts = ShortcutsEditor()
    self.generalProps = AppSettingsEditor()
    self.colorScheme = ColorSchemeEditor()
    self.imgProcClctn = AlgCtorCollection(ImgProcWrapper)
    self.quickLoader = QuickLoaderEditor()

    self.docks: List[QtWidgets.QDockWidget] = []
    propsGrouping = ParamEditorDockGrouping([self.generalProps, self.colorScheme], 'General Properties')
    shcGrouping = ParamEditorDockGrouping([self.shortcuts, self.quickLoader], 'Shortcuts')

    self.addDocks([self.filter, propsGrouping, shcGrouping])

  @property
  def registerableEditors(self):
    outList = []
    for editor in self.docks:
      if isinstance(editor, ParamEditorDockGrouping):
        outList.extend(editor.editors)
      else:
        outList.append(editor)
    return outList

  def addDocks(self, docks: Union[QtWidgets.QDockWidget, List[QtWidgets.QDockWidget]], blockEmit=False):
    if not isinstance(docks, List):
      docks = [docks]

    for dock in docks:
      if dock in self.docks or dock is self.quickLoader:
        # This logic is to add editors to quick loader, checking for it prevents recursion
        continue
      self.docks.append(dock)
      if isinstance(dock, ParamEditorDockGrouping):
        for editor in dock.editors:
          if editor in self.docks:
            # The editor will be accounted for as a group, so remove it as an individual
            self.docks.remove(editor)
        self.quickLoader.listModel.addEditors(dock.editors)
      else:
        self.quickLoader.listModel.addEditors([dock])

    if not blockEmit:
      self.sigDocksAdded.emit(docks)


  def registerGroup(self, groupParam: FRParam, **opts):
    def multiEditorClsDecorator(cls):
      # Since all legwork is done inside the editors themselves, simply call each decorator from here as needed
      for editor in self.registerableEditors:
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
    if isinstance(plugin, TableFieldPlugin):
      self.tableFieldPlugins.append(weakref.proxy(plugin))
    if plugin.docks is not None:
      self.addDocks(plugin.docks)
    self.plugins.append(plugin)
    return plugin

  def close(self):
    for editor in self.registerableEditors:
      editor.close()


FR_SINGLETON = _FRSingleton()
