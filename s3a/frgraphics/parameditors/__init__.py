from typing import List, Union

from pyqtgraph.Qt import QtWidgets, QtCore

from s3a.actionstack import FRActionStack
from s3a.projectvars import GEN_PROPS_DIR, SCHEMES_DIR, BASE_DIR
from s3a.structures import FRParam
from . import pgregistered
from .genericeditor import FRParamEditor, FRParamEditorDockGrouping
from .processor import FRAlgPropsMgr
from .quickloader import FRQuickLoaderEditor
from .shortcut import FRShortcutsEditor
from .table import FRTableFilterEditor, FRTableData

Signal = QtCore.Signal


class FRGeneralPropertiesEditor(FRParamEditor):
  def __init__(self, parent=None):
    super().__init__(parent, paramList=[], saveDir=GEN_PROPS_DIR, fileType='regctrl')


class FRColorSchemeEditor(FRParamEditor):
  def __init__(self, parent=None):
    super().__init__(parent, paramList=[], saveDir=SCHEMES_DIR, fileType='scheme')


class _FRSingleton(QtCore.QObject):
  sigDocksAdded = Signal(object) # List[QtWidgets.QDockWidget]

  def __init__(self, parent=None):
    super().__init__(parent)
    self.tableData = FRTableData()
    self.tableData.loadCfg(BASE_DIR/'tablecfg.yml')
    self.filter = self.tableData.filter


    self.shortcuts = FRShortcutsEditor()
    self.scheme = FRColorSchemeEditor()
    self.generalProps = FRGeneralPropertiesEditor()

    self.algParamMgr = FRAlgPropsMgr()
    self.docks: List[FRParamEditor] = [self.scheme, self.generalProps, self.shortcuts]
    self.quickLoader = FRQuickLoaderEditor(editorList=self.registerableEditors)
    self.docks.append(self.quickLoader)
    addFn = self.quickLoader.listModel.addEditors
    self.algParamMgr.sigProcessorCreated.connect(lambda editor: addFn([editor]))

    self.actionStack = FRActionStack()

  @property
  def registerableEditors(self):
    outList = []
    for editor in self.docks + self.algParamMgr.spawnedCollections:
      if isinstance(editor, FRParamEditorDockGrouping):
        outList.extend(editor.editors)
      else:
        outList.append(editor)
    return outList

  def addDocks(self, docks: Union[QtWidgets.QDockWidget, List[QtWidgets.QDockWidget]], blockEmit=False):
    if not isinstance(docks, List):
      docks = [docks]

    for dock in docks:
      self.docks.append(dock)
      if isinstance(dock, FRParamEditorDockGrouping):
        self.quickLoader.listModel.addEditors(dock.editors)
      else:
        self.quickLoader.listModel.addEditors(dock)

    if not blockEmit:
      self.sigDocksAdded.emit(docks)


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
