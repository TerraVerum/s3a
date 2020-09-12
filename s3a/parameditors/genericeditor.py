from __future__ import annotations

from abc import ABC
from typing import List, Dict, Union, Type, Tuple, Optional

import pandas as pd
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import Parameter, ParameterItem

from s3a import models
from s3a.constants import MENU_OPTS_DIR
from s3a.generalutils import frPascalCaseToTitle
from s3a.graphicsutils import dialogGetSaveFileName
from s3a.models.editorbase import FRParamEditorBase
from s3a import parameditors
from s3a.processing import FRImgProcWrapper
from s3a.structures import FRParam, FilePath, NChanImg, FRVertices

Signal = QtCore.Signal

def clearUnwantedParamVals(paramState: dict):
  for k, child in paramState.get('children', {}).items():
    clearUnwantedParamVals(child)
  if paramState.get('value', True) is None:
    paramState.pop('value')

_childTuple_asValue = Tuple[FRParam,...]
childTuple_asParam = Tuple[Tuple[FRParam,...], bool]
_keyType = Union[FRParam, Union[_childTuple_asValue, childTuple_asParam]]
class FRParamEditor(FRParamEditorBase):
  """
  GUI controls for user-interactive parameters within S3A. Each window consists of
  a parameter tree and basic saving capabilities.
  """
  def __init__(self, parent=None, paramList: List[Dict]=None, saveDir: FilePath='.',
               fileType='param', name=None, topTreeChild: Parameter=None,
               registerCls: Type=None, registerParam: FRParam=None, **registerGroupOpts):
    super().__init__(parent, paramList, saveDir, fileType, name, topTreeChild,
                     registerCls, registerParam, **registerGroupOpts)
    self.dock = self
    self.hide()
    self.setWindowTitle(self.name)
    self.setObjectName(self.name)

    # This will be set to 'True' when an action for this editor is added to
    # the main window menu
    self.hasMenuOption = False

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

    if registerCls is not None:
      self.registerGroup(registerParam)(registerCls)

  def setAllExpanded(self, expandedVal=True):
    try:
      topTreeItem: ParameterItem = next(iter(self.params.items))
    except StopIteration:
      return
    for ii in range(topTreeItem.childCount()):
      topTreeItem.child(ii).setExpanded(expandedVal)

  def _expandCols(self):
    # totWidth = 0
    for colIdx in range(2):
      self.tree.resizeColumnToContents(colIdx)
    #   totWidth += self.tree.columnWidth(colIdx) + self.tree.margin
    # appInst.processEvents()
    # self.dockContentsWidget.setMinimumWidth(totWidth)
    self.tree.setColumnWidth(0, self.width()//2)
    self.resize(self.tree.width(), self.height())

  def show(self):
    self.setWindowState(QtCore.Qt.WindowActive)
    # Necessary on MacOS
    self.raise_()
    # Necessary on Windows
    self.activateWindow()
    self.applyBtn.setFocus()
    super().show()

  def reject(self):
    """
    If window is closed apart from pressing 'accept', restore pre-edit state
    """
    self.params.restoreState(self._stateBeforeEdit, removeChildren=False)
    super().reject()

  def applyChanges(self):
    # Don't emit any signals if nothing changed
    newState = self.params.saveState(filter='user')
    outDict = self.params.getValues()
    if self._stateBeforeEdit != newState:
      self._stateBeforeEdit = newState
      self.sigParamStateUpdated.emit(outDict)
    return outDict

  @staticmethod
  def buildClsToolsEditor(cls: type, name=None):
    groupName = frPascalCaseToTitle(cls.__name__)
    lowerGroupName = groupName.lower()
    toolsDir = MENU_OPTS_DIR / lowerGroupName
    if name is None:
      name = groupName + ' Tools'
    toolsEditor = FRParamEditor(
      saveDir=toolsDir, fileType=lowerGroupName.replace(' ', '') + 'tools',
      name=name, registerCls=cls, useNewInit=False
    )
    for btn in (toolsEditor.saveAsBtn, toolsEditor.applyBtn, toolsEditor.expandAllBtn,
                toolsEditor.collapseAllBtn):
      btn.hide()
    return toolsEditor


  def saveParamState_gui(self):
    saveName = dialogGetSaveFileName(self, 'Save As', self.lastAppliedName)
    self.saveParamState(saveName)

class FRParamEditorDockGrouping(QtWidgets.QDockWidget):
  """
  When multiple parameter editor windows should be grouped under the same heading,
  this class is responsible for performing that grouping.
  """
  def __init__(self, editors: List[FRParamEditor], dockName, parent=None):
    super().__init__(parent)
    self.tabs = QtWidgets.QTabWidget(self)
    self.hide()

    if dockName is None:
      dockName = editors[0].name
    self.name = dockName

    for editor in editors:
      # "Main Image Settings" -> "Settings"
      tabName = self.getTabName(editor)
      self.tabs.addTab(editor.dockContentsWidget, tabName)
      editor.dock = self
    mainLayout = QtWidgets.QVBoxLayout()
    mainLayout.addWidget(self.tabs)
    centralWidget = QtWidgets.QWidget()
    centralWidget.setLayout(mainLayout)
    self.setWidget(centralWidget)
    self.setObjectName(dockName)
    self.setWindowTitle(dockName)

    self.editors = editors

  def setParent(self, parent: QtWidgets.QWidget=None):
    super().setParent(parent)
    for editor in self.editors:
      editor.setParent(parent)

  def getTabName(self, editor: FRParamEditor):
    if self.name in editor.name:
      tabName = editor.name.split(self.name)[1][1:]
      if len(tabName) == 0:
        tabName = editor.name
    else:
      tabName = editor.name
    return tabName


class FRParamEditorPlugin(ABC):
  """
  Primitive plugin which can interface with S3A functionality. When this class is overloaded,
  the child class is given a reference to the main S3A window and S3A is made aware of the
  plugin's existence. For interfacing with table fields, see the special case of
  :class:`FRTableFieldPlugin`
  """
  name: str=None
  toolsEditor: FRParamEditor
  """Param Editor window which holds user-editable properties exposed by the programmer"""
  s3a: models.s3abase.S3ABase=None
  """Reference to the current S3A window"""

  docks: Union[FRParamEditorDockGrouping, FRParamEditor] = None
  """
  Docks that should be shown in S3A's menu bar. By default, just the toolsEditor is shown.
  If multiple param editors must be visible, manually set this property to a
  :class:`FRParamEditorDockGrouping` as performed in :class:`FRVerticesPlugin`.
  """

  @classmethod
  def __initEditorParams__(cls):
    pass

  def attachS3aRef(self, s3a: models.s3abase.S3ABase):
    self.s3a = s3a


class FRTableFieldPlugin(FRParamEditorPlugin):
  """
  Primary method for providing algorithmic refinement of table field data. For
  instance, the :class:`FRVerticesPlugin` class can refine initial bounding
  box estimates of component vertices using custom image processing algorithms.
  """

  procCollection: parameditors.algcollection.FRAlgParamEditor= None
  """
  Most table field plugins will use some sort of processor to infer field data.
  This property holds spawned collections. See :class:`FRVerticesPlugin` for
  an example.
  """

  focusedImg = None
  """
  Holds a reference to the focused image and set when the s3a reference is set. This
  is useful for most table field plugins, since focusedImg will hold a reference to the
  component series that is modified by the plugins.
  """

  _active=False

  @classmethod
  def __initEditorParams__(cls):
    """
    Initializes shared parameters accessible through the :meth:`FRParamEditor.registerProp`
    function
    """
    super().__initEditorParams__()
    cls.toolsEditor = FRParamEditor.buildClsToolsEditor(cls, 'Tools')

  def attachS3aRef(self, s3a: models.s3abase.S3ABase):
    super().attachS3aRef(s3a)
    self.focusedImg = s3a.focusedImg


  def updateAll(self, mainImg: Optional[NChanImg], newComp: Optional[pd.Series] = None):
    """
    This function is called when a new component is created or the focused image is updated
    from the main view. See :meth:`FRFocusedImage.updateAll` for parameters.
    """
    raise NotImplementedError

  def handleShapeFinished(self, roiVerts: FRVertices):
    """
    Called whenever a user completes a shape in the focused image. See
    :meth:`FRFocusedImage.handleShapeFinished` for parameters.
    """
    raise NotImplementedError

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

  @property
  def curProcessor(self):
    return self.procCollection.curProcessor
  @curProcessor.setter
  def curProcessor(self, newProcessor: Union[str, FRImgProcWrapper]):
    self.procCollection.switchActiveProcessor(newProcessor)
