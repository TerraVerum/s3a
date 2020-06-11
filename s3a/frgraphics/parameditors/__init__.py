from typing import List

from .genericeditor import FRParamEditor
from .processor import FRAlgPropsMgr
from .shortcut import FRShortcutsEditor
from .quickloader import FRQuickLoaderEditor
from .table import FRTableFilterEditor, FRTableData
from . import pgregistered
from s3a.projectvars import GEN_PROPS_DIR, SCHEMES_DIR, BASE_DIR
from s3a.structures import FRParam
from s3a.actionstack import FRActionStack


class FRGeneralPropertiesEditor(FRParamEditor):
  def __init__(self, parent=None):
    super().__init__(parent, paramList=[], saveDir=GEN_PROPS_DIR, fileType='regctrl')


class FRColorSchemeEditor(FRParamEditor):
  def __init__(self, parent=None):
    super().__init__(parent, paramList=[], saveDir=SCHEMES_DIR, fileType='scheme')


class _FRSingleton:

  def __init__(self):
    self.tableData = FRTableData()
    self.tableData.loadCfg(BASE_DIR/'tablecfg.yml')
    self.filter = self.tableData.filter

    self.shortcuts = FRShortcutsEditor()
    self.scheme = FRColorSchemeEditor()
    self.generalProps = FRGeneralPropertiesEditor()

    self._registerableEditors: List[FRParamEditor] = \
      [self.scheme, self.shortcuts, self.generalProps, self.filter]

    self.algParamMgr = FRAlgPropsMgr()
    self.quickLoader = FRQuickLoaderEditor(editorList=self.registerableEditors)
    self.algParamMgr.sigProcessorCreated.connect(lambda editor:
                                                 self.quickLoader.listModel.addEditors([editor]))

    self.actionStack = FRActionStack()
  @property
  def allEditors(self):
    return self.registerableEditors + [self.quickLoader]

  @property
  def registerableEditors(self):
    return self._registerableEditors + self.algParamMgr.spawnedCollections

  def registerGroup(self, clsParam: FRParam, **opts):
    def multiEditorClsDecorator(cls):
      # Since all legwork is done inside the editors themselves, simply call each decorator from here as needed
      for editor in self.registerableEditors:
        cls = editor.registerGroup(clsParam, **opts)(cls)
      return cls
    return multiEditorClsDecorator

  def close(self):
    for editor in self.registerableEditors:
      editor.close()


FR_SINGLETON = _FRSingleton()
