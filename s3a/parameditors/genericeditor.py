from __future__ import annotations

from typing import List, Dict, Union, Type, Tuple, Optional, Sequence
from functools import partial

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import Parameter

from s3a.generalutils import pascalCaseToTitle
from s3a.graphicsutils import dialogGetSaveFileName, addDirItemsToMenu
from s3a.models.editorbase import ParamEditorBase
from s3a.structures import FRParam, FilePath

Signal = QtCore.Signal

def clearUnwantedParamVals(paramState: dict):
  for k, child in paramState.get('children', {}).items():
    clearUnwantedParamVals(child)
  if paramState.get('value', True) is None:
    paramState.pop('value')

_childTuple_asValue = Tuple[FRParam,...]
childTuple_asParam = Tuple[Tuple[FRParam,...], bool]
_keyType = Union[FRParam, Union[_childTuple_asValue, childTuple_asParam]]
class ParamEditor(ParamEditorBase):
  """
  GUI controls for user-interactive parameters within S3A. Each window consists of
  a parameter tree and basic saving capabilities.
  """
  def __init__(self, parent=None, paramList: List[Dict] = None,
               saveDir: Optional[FilePath] = '.', fileType='param', name=None,
               topTreeChild: Parameter = None, **kwargs):
    super().__init__(parent, paramList, saveDir, fileType, name, topTreeChild,
                     **kwargs)
    self.dock = self
    self.hide()
    self.setWindowTitle(self.name)
    self.setObjectName(self.name)

    # -----------
    # Additional widget buttons
    # -----------
    self.expandAllBtn = QtWidgets.QPushButton('Expand All')
    self.collapseAllBtn = QtWidgets.QPushButton('Collapse All')
    self.saveAsBtn = QtWidgets.QPushButton('Save As...')
    self.applyBtn = QtWidgets.QPushButton('Apply')

    # -----------
    # Widget layout
    # -----------
    self.dockContentsWidget = QtWidgets.QWidget(parent)
    self.setWidget(self.dockContentsWidget)
    expandCollapseBtnLayout = QtWidgets.QHBoxLayout()
    expandCollapseBtnLayout.addWidget(self.expandAllBtn)
    expandCollapseBtnLayout.addWidget(self.collapseAllBtn)
    paramStateBtns = QtWidgets.QHBoxLayout()
    paramStateBtns.addWidget(self.saveAsBtn)
    paramStateBtns.addWidget(self.applyBtn)

    self.centralLayout = QtWidgets.QVBoxLayout(self.dockContentsWidget)
    self.centralLayout.addLayout(expandCollapseBtnLayout)
    self.centralLayout.addWidget(self.tree)
    self.centralLayout.addLayout(paramStateBtns)
    # self.setLayout(centralLayout)
    self.tree.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    # -----------
    # UI Element Signals
    # -----------
    self.expandAllBtn.clicked.connect(lambda: self.setAllExpanded(True))
    self.collapseAllBtn.clicked.connect(lambda: self.setAllExpanded(False))
    self.saveAsBtn.clicked.connect(self.saveParamState_gui)
    self.applyBtn.clicked.connect(self.applyChanges)

  def __repr__(self):
    selfCls = type(self)
    oldName: str = super().__repr__()
    # Remove module name for brevity
    oldName = oldName.replace(f'{selfCls.__module__}.{selfCls.__name__}',
                              f'{selfCls.__name__} \'{self.name}\'')
    return oldName

  def show(self):
    if self.dock is self:
      return super().show()
    if isinstance(self.dock, ParamEditorDockGrouping):
      tabs: QtWidgets.QTabWidget = self.dock.tabs
      dockIdx = tabs.indexOf(self.dockContentsWidget)
      tabs.setCurrentIndex(dockIdx)
    self.tree.resizeColumnToContents(0)
    # Necessary on MacOS
    self.dock.setWindowState(QtCore.Qt.WindowActive)
    self.dock.raise_()
    self.dock.show()
    # Necessary on Windows
    self.activateWindow()
    self.applyBtn.setFocus()

  def reject(self):
    """
    If window is closed apart from pressing 'accept', restore pre-edit state
    """
    self.params.restoreState(self._stateBeforeEdit, removeChildren=False)
    super().reject()

  @staticmethod
  def buildClsToolsEditor(cls: type, name=None):
    groupName = pascalCaseToTitle(cls.__name__)
    lowerGroupName = groupName.lower()
    if name is None:
      name = groupName + ' Tools'
    toolsEditor = ParamEditor(saveDir=None,
                              fileType=lowerGroupName.replace(' ', '') + 'toolsEditor',
                              name=name, useNewInit=False)
    for btn in (toolsEditor.saveAsBtn, toolsEditor.applyBtn, toolsEditor.expandAllBtn,
                toolsEditor.collapseAllBtn):
      btn.hide()
    return toolsEditor

  def saveParamState_gui(self):
    saveName = dialogGetSaveFileName(self, 'Save As', self.lastAppliedName)
    self.saveParamState(saveName)

  def createMenuOpt(self, overrideName=None, parentMenu: QtWidgets.QMenu=None):
    def loadFunc(nameToLoad: str) -> Optional[dict]:
      with pg.BusyCursor():
        return self.loadParamState(nameToLoad)

    if overrideName is None:
      overrideName = self.name
    editAct = QtWidgets.QAction('Open ' + overrideName, self)
    if self.saveDir is None:
      # No save options are possible, just use an action instead of dropdown menu
      newMenuOrAct = editAct
      if parentMenu is not None:
        parentMenu.addAction(newMenuOrAct)
    else:
      newMenuOrAct = QtWidgets.QMenu(overrideName, self)
      newMenuOrAct.addAction(editAct)
      newMenuOrAct.addSeparator()
      def populateFunc():
        addDirItemsToMenu(newMenuOrAct,
                          self.saveDir.glob(f'*.{self.fileType}'),
                          loadFunc)
      self.sigParamStateCreated.connect(populateFunc)
      # Initialize default menus
      populateFunc()
      if parentMenu is not None:
        parentMenu.addMenu(newMenuOrAct)
    editAct.triggered.connect(self.show)
    return newMenuOrAct

class ParamEditorDockGrouping(QtWidgets.QDockWidget):
  """
  When multiple parameter editor windows should be grouped under the same heading,
  this class is responsible for performing that grouping.
  """
  def __init__(self, editors: List[ParamEditor]=None, dockName:str='', parent=None):
    super().__init__(parent)
    self.tabs = QtWidgets.QTabWidget(self)
    self.hide()

    if editors is None:
      editors = []

    if len(dockName) == 0 and len(editors) > 0:
      dockName = editors[0].name
    dockName = dockName.replace('&', '')
    self.name = dockName

    self.editors = []
    self.addEditors(editors)

    mainLayout = QtWidgets.QVBoxLayout()
    mainLayout.addWidget(self.tabs)
    centralWidget = QtWidgets.QWidget()
    centralWidget.setLayout(mainLayout)
    self.setWidget(centralWidget)
    self.setObjectName(dockName)
    self.setWindowTitle(dockName)

    self.biggestMinWidth = 0

  def addEditors(self, editors: Sequence[ParamEditor]):
    minWidth = 0
    for editor in editors:
      editor.tree.resizeColumnToContents(0)
      if editor.width() > minWidth:
        minWidth = editor.width()//2
      # "Main Image Settings" -> "Settings"
      tabName = self.getTabName(editor)
      self.tabs.addTab(editor.dockContentsWidget, tabName)
      editor.dock = self
      self.editors.append(editor)
    self.biggestMinWidth = minWidth

  def removeEditors(self, editors: Sequence[ParamEditor]):
    for editor in editors:
      idx = self.editors.index(editor)
      self.tabs.removeTab(idx)
      editor.dock = editor
      del self.editors[idx]

  def setParent(self, parent: QtWidgets.QWidget=None):
    super().setParent(parent)
    for editor in self.editors:
      editor.setParent(parent)

  def getTabName(self, editor: ParamEditor):
    if self.name in editor.name and len(self.name) > 0:
      tabName = editor.name.split(self.name)[1][1:]
      if len(tabName) == 0:
        tabName = editor.name
    else:
      tabName = editor.name
    return tabName

  def createMenuOpt(self, overrideName=None, parentMenu: QtWidgets.QMenu=None):
    if overrideName is None:
      overrideName = self.name
    if parentMenu is None:
      parentMenu = QtWidgets.QMenu(overrideName, self)
    # newMenu = create_addMenuAct(self, parentBtn, dockEditor.name, True)
    for editor in self.editors: # type: ParamEditor
      # "Main Image Settings" -> "Settings"
      tabName = self.getTabName(editor)
      nameWithoutBase = tabName
      editor.createMenuOpt(overrideName=nameWithoutBase, parentMenu=parentMenu)
    return parentMenu


class EditorPropsMixin:
  __groupingName__: str = None

  REGISTERED_GROUPINGS = set()
  def __new__(cls, *args, **kwargs):
    if cls.__groupingName__ is None:
      cls.__groupingName__ = pascalCaseToTitle(cls.__name__)
    if cls not in cls.REGISTERED_GROUPINGS:
      basePath = (cls.__groupingName__,)
      if basePath[0] == '':
        basePath = ()
      with ParamEditor.setBaseRegisterPath(*basePath):
        cls.__initEditorParams__()
      cls.REGISTERED_GROUPINGS.add(cls)
    return super().__new__(cls, *args, **kwargs)

  @classmethod
  def __initEditorParams__(cls):
    pass