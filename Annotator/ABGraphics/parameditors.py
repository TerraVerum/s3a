# -*- coding: utf-8 -*-
import pickle as pkl
import re
import sys
from dataclasses import dataclass
from functools import partial
from os.path import join
from typing import Sequence, Union, Callable, Any, Optional

from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.parametertree import (Parameter, ParameterTree, parameterTypes)

from Annotator.params import ABParam, ABParamGroup
from .graphicsutils import dialogSaveToFile
from ..constants import (
  SCHEMES_DIR, GEN_PROPS_DIR, FILTERS_DIR, SHORTCUTS_DIR,
  TEMPLATE_COMP as TC, TEMPLATE_COMP_TYPES as COMP_TYPES)

Signal = QtCore.pyqtSignal

def _genList(nameIter, paramType, defaultVal, defaultParam='value'):
  """Helper for generating children elements"""
  return [{'name': name, 'type': paramType, defaultParam: defaultVal} for name in nameIter]

@dataclass
class ABShortcutCtorGroup:
  constParam: ABParam
  func: Callable
  args: list

class ABEditableShortcut(QtWidgets.QShortcut):
  paramIdx: QtGui.QKeySequence

class ShortcutParameterItem(parameterTypes.WidgetParameterItem):
  """
  Class for creating custom shortcuts. Must be made here since pyqtgraph doesn't
  provide an implementation.
  """

  def __init__(self, param, depth):
    super().__init__(param, depth)
    self.item: Optional[QtGui.QKeySequence] = None

  def makeWidget(self):
    item = QtWidgets.QKeySequenceEdit()

    item.sigChanged = item.editingFinished
    item.value = lambda: item.keySequence().toString()
    item.setValue = item.setKeySequence
    self.item = item
    return self.item

  # def contextMenuEvent(self, ev: QtGui.QContextMenuEvent):
  #   menu = self.contextMenu
  #   delAct = QtWidgets.QAction('Set Blank')
  #   delAct.triggered.connect(lambda: self.widget.setValue(''))
  #   menu.addAction(delAct)
  #   menu.exec(ev.globalPos())

class ShortcutParameter(Parameter):
  itemClass = ShortcutParameterItem

parameterTypes.registerParameterType('shortcut', ShortcutParameter)

@dataclass
class ABBoundFnParams:
  param: ABParam
  func: Callable
  defaultFnArgs: list

class ConstParamWidget(QtWidgets.QDialog):
  sigParamStateCreated = Signal(str)
  sigParamStateUpdated = Signal(dict)

  def __init__(self, parent=None, paramDict=None, saveDir='.',
               saveExt='param', saveDlgName='Save As', createDefaultOutput=True):
    # Place in list so an empty value gets unpacked into super constructor
    if paramDict is None:
      paramDict = []

    super().__init__(parent)
    self.resize(500, 400)

    self.paramsPerClass = {}
    self.classToParamMapping = {}

    # -----------
    # Construct parameter tree
    # -----------
    self.params = Parameter(name='Parameters', type='group', children=paramDict)
    self.tree = ParameterTree()
    self.tree.setParameters(self.params, showTop=False)

    # Allow the user to change column widths
    for colIdx in range(2):
      self.tree.header().setSectionResizeMode(colIdx, QtWidgets.QHeaderView.Interactive)

    # -----------
    # Internal parameters for saving settings
    # -----------
    self.saveDir = saveDir
    self.fileType = saveExt
    self._saveDlgName = saveDlgName
    self._stateBeforeEdit = self.params.saveState()

    # -----------
    # Save default state if desired
    # -----------
    if createDefaultOutput:
      with open(join(saveDir, f'Default.{saveExt}'), 'wb') as ofile:
        pkl.dump(self.params.saveState(), ofile)

    # -----------
    # Additional widget buttons
    # -----------
    self.saveAsBtn = QtWidgets.QPushButton('Save As...')
    self.applyBtn = QtWidgets.QPushButton('Apply')
    self.closeBtn = QtWidgets.QPushButton('Close')

    # -----------
    # Widget layout
    # -----------
    btnLayout = QtWidgets.QHBoxLayout()
    btnLayout.addWidget(self.saveAsBtn)
    btnLayout.addWidget(self.applyBtn)
    btnLayout.addWidget(self.closeBtn)
    QtWidgets.QWidget.setTabOrder(self.applyBtn, self.saveAsBtn)

    centralLayout = QtWidgets.QVBoxLayout()
    centralLayout.addWidget(self.tree)
    centralLayout.addLayout(btnLayout)
    self.setLayout(centralLayout)
    # -----------
    # UI Element Signals
    # -----------
    self.saveAsBtn.clicked.connect(self.saveAsBtnClicked)
    self.closeBtn.clicked.connect(self.close)
    self.applyBtn.clicked.connect(self.applyBtnClicked)

  # Helper method for accessing simple parameter values
  def __getitem__(self, keys: Union[tuple, ABParam, Sequence]):
    """
    Convenience function for accessing child parameters within a parameter editor.
      - If :param:`keys` is a single :class:`ABParam`, the value at that parameter is
        extracted and returned to the user.
      - If :param:`keys` is a :class:`tuple`:

        * The first element of the tuple must correspond to the base name within the
          parameter grouping in order to properly extract the corresponding children.
          For instance, to extract MARGIN from :class:`RegionControlsEditor`,
              you must first specify the group parent for that parameter:
              >>> seedThresh = TEMPLATE_REG_CTRLS[REG_CTRLS.FOCUSED_IMG_PARAMS,
              >>>   REG_CTRLS.MARGIN]
        * The second parameter must be a signle :class:`ABParam` objects or a sequence
          of :class:`ABParam` objects. If a sequence is given, a list of output values
          respecting input order is provided.
        * The third parameter is optional. If provided, the :class:`Parameter<pyqtgraph.Parameter>`
          object is returned instead of the :func:`value()<Parameter.value>` data
          *within* the object.

    :param keys: One of of the following:
    :return:
    """
    returnSingle = False
    extractObj = False
    if isinstance(keys, tuple):
      if len(keys) > 2:
        extractObj = True
      baseParam = [keys[0].name]
      keys = keys[1]
    else:
      baseParam = []
    if not hasattr(keys, '__iter__'):
      keys = [keys]
      returnSingle = True
    outVals = []
    extractFunc = lambda name: self.params.child(*baseParam, name)
    if not extractObj:
      oldExtractFunc = extractFunc
      extractFunc = lambda name: oldExtractFunc(name).value()
    for curKey in keys: # type: ABParam
      outVals.append(extractFunc(curKey.name))
    if returnSingle:
      return outVals[0]
    else:
      return outVals

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
    self.params.restoreState(self._stateBeforeEdit)
    super().reject()

  def keyPressEvent(self, ev: QtGui.QKeyEvent):
    pressedKey = ev.key()
    if pressedKey == QtCore.Qt.Key_Enter or pressedKey == QtCore.Qt.Key_Return:
      self.applyBtnClicked()
    super().keyPressEvent(ev)

  def applyBtnClicked(self):
    self._stateBeforeEdit = self.params.saveState()
    outDict = self.params.getValues()
    self.sigParamStateUpdated.emit(outDict)
    return outDict

  def saveAsBtnClicked(self, saveName=None):
    """
    :return: List where each index corresponds to the tree's parameter.
    This is suitable for extending with an ID and vertex list, after which
    it can be placed into the component table.
    """
    paramState = self.params.saveState()
    if saveName is False or saveName is None:
      saveName = dialogSaveToFile(self, paramState, self._saveDlgName,
                                  self.saveDir, self.fileType, allowOverwriteDefault=False)
    else:
      with open(saveName, 'wb') as saveFile:
        pkl.dump(paramState, saveFile)
    if saveName is not None:
      # Accept new param state after saving
      self.applyBtnClicked()
      outDict = self.params.getValues()
      self.sigParamStateCreated.emit(saveName)
      return outDict
    # If no name specified
    return None

  def loadState(self, newStateDict):
    self.params.restoreState(newStateDict, addChildren=False)

  def registerProp(self, constParam: ABParam):
    # First add registered property to self list
    def funcWrapper(func):
      func, clsName = self.registerMethod(constParam)(func, True)

      @property
      def paramGetter(clsObj):
        # Use function wrapper instead of directly returning so no errors are thrown when class isn't fully instantiated
        return self[self.classToParamMapping[clsName], constParam]
      return paramGetter
    return funcWrapper

  def registerMethod(self, constParam: ABParam, fnArgs=None):
    """
    Designed for use as a function decorator. Registers the decorated function into a list
    of methods known to the :class:`ShortcutsEditor`. These functions are then accessable from
    customizeable shortcuts.
    """
    if fnArgs is None:
      fnArgs = []

    def registerMethodDecorator(func: Callable, returnClsName=False):
      boundFnParam = ABBoundFnParams(param=constParam, func=func, defaultFnArgs=fnArgs)
      fullFuncName = func.__qualname__
      lastDotIdx = fullFuncName.find('.')
      if lastDotIdx < 0:
        # This function isn't inside a class, so defer
        # to the global namespace
        fnParentClass = 'Global'
      else:
        # Get name of class containing this function
        fnParentClass = fullFuncName[:lastDotIdx]

      self._addParamToList(fnParentClass, boundFnParam)
      if returnClsName:
        return func, fnParentClass
      else:
        return func
    return registerMethodDecorator

  def _addParamToList(self, clsName: str, param: Union[ABParam, ABBoundFnParams]):
    clsParams = self.paramsPerClass.get(clsName, [])
    clsParams.append(param)
    self.paramsPerClass[clsName] = clsParams

  def registerClass(self, clsParam: ABParam):
    """
    Intended for use as a class decorator. Registers a class as able to hold
    customizable shortcuts.
    """
    def classDecorator(cls):
      clsName = cls.__qualname__
      self.addParamsFromClass(clsName, clsParam)
      # Now that class params are registered, save off default file
      with open(join(self.saveDir, f'Default.{self.fileType}'), 'wb') as ofile:
        pkl.dump(self.params.saveState(), ofile)
      self.classToParamMapping[clsName] = clsParam
      oldClsInit = cls.__init__
      def newClassInit(clsObj, *args, **kwargs):
        retVal = oldClsInit(clsObj, *args, **kwargs)
        self._extendedClassInit(clsObj, clsParam)
        return retVal
      cls.__init__ = newClassInit
      return cls
    return classDecorator

  def _extendedClassInit(self, clsObj: Any, clsParam: ABParam):
    """
    For editors that need to perform any initializations within the decorated class,
      they must be able to access the decorated class' *init* function and modify it.
      Allow this by providing an overloadable stub that is inserted into the decorated
      class *init*.
    """
    return

  def addParamsFromClass(self, clsName, clsParam: ABParam):
    """
    Once the top-level widget is set, we can construct the
    parameter editor widget. Set the parent of each shortcut so they
    can be used when focusing the main window, then construct the
    editor widget.

    :param clsName: Fully qualified name of the class

    :param clsParam: :class:`ABParam` value encapsulating the human readable class name.
           This is how the class will be displayed in the :class:`ShortcutsEditor`.

    :return: None
    """
    classParamList = self.paramsPerClass.get(clsName, [])
    # Don't add a category unless at least one list element is present
    if len(classParamList) == 0: return
    # If a human-readable name was given, replace class name with human name
    paramChildren = []
    paramGroup = {'name': clsParam.name, 'type': 'group',
                  'children': paramChildren}
    for boundFn in classParamList:
      paramForTree = {'name': boundFn.param.name,
                       'type': boundFn.param.valType,
                       'value': boundFn.param.value}
      paramChildren.append(paramForTree)
    self.params.addChild(paramGroup)
    # Make sure all new names are properly displayed
    self.tree.resizeColumnToContents(0)
    self._stateBeforeEdit = self.params.saveState()

class GeneralPropertiesEditor(ConstParamWidget):
  def __init__(self, parent=None):
    super().__init__(parent, paramDict=[], saveDir=GEN_PROPS_DIR, saveExt='regctrl')

class TableFilterEditor(ConstParamWidget):
  def __init__(self, parent=None):
    minMaxParam = _genList(['min', 'max'], 'int', 0)
    # Make max 'infinity'
    minMaxParam[1]['value'] = sys.maxsize
    validatedParms = _genList(['Validated', 'Not Validated'], 'bool', True)
    devTypeParam = _genList((param.name for param in COMP_TYPES), 'bool', True)
    xyVerts = _genList(['X Bounds', 'Y Bounds'], 'group', minMaxParam, 'children')
    _FILTER_DICT = [
        {'name': TC.INST_ID.name, 'type': 'group', 'children': minMaxParam},
        {'name': TC.VALIDATED.name, 'type': 'group', 'children': validatedParms},
        {'name': TC.DEV_TYPE.name, 'type': 'group', 'children': devTypeParam},
        {'name': TC.LOGO.name, 'type': 'str', 'value': '.*'},
        {'name': TC.NOTES.name, 'type': 'str', 'value': '.*'},
        {'name': TC.BOARD_TEXT.name, 'type': 'str', 'value': '.*'},
        {'name': TC.DEV_TEXT.name, 'type': 'str', 'value': '.*'},
        {'name': TC.VERTICES.name, 'type': 'group', 'children': xyVerts}
      ]
    super().__init__(parent, paramDict=_FILTER_DICT, saveDir=FILTERS_DIR, saveExt='filter')

class TmpTblFilterEditor(ConstParamWidget):
  def __init__(self):
    for param in TC:
      paramType = type(param.value)
      if paramType == int:
        curChild = self.createMinMaxFilter()
      elif paramType == ABParamGroup:
        pass

class ShortcutsEditor(ConstParamWidget):

  def __init__(self, parent=None):

    self.shortcuts = []
    # Unlike other param editors, these children don't get filled in until
    # after the top-level widget is passed to the shortcut editor
    super().__init__(parent, [], saveDir=SHORTCUTS_DIR, saveExt='shortcut')

  def _extendedClassInit(self, clsObj: Any, clsParam: ABParam):
    clsName = type(clsObj).__qualname__
    boundParamList = self.paramsPerClass.get(clsName, [])
    for boundParam in boundParamList:
      seqCopy = QtGui.QKeySequence(boundParam.param.value)
      shortcut = ABEditableShortcut(seqCopy, clsObj)
      shortcut.paramIdx = (clsParam, boundParam.param)
      shortcut.activated.connect(partial(boundParam.func, clsObj, *boundParam.defaultFnArgs))
      shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
      self.shortcuts.append(shortcut)

  def applyBtnClicked(self):
    for shortcut in self.shortcuts: #type: ABEditableShortcut
      shortcut.setKey(self[shortcut.paramIdx])
    super().applyBtnClicked()


class SchemeEditor(ConstParamWidget):
  def __init__(self, parent=None):
    super().__init__(parent, paramDict=[], saveDir=SCHEMES_DIR,
                     saveExt='scheme')

class _ABSingleton:
  shortcuts = ShortcutsEditor()
  scheme = SchemeEditor()
  generalProps = GeneralPropertiesEditor()
  filter = TableFilterEditor()

  def __init__(self):
    # Code retrieved from https://stackoverflow.com/a/20214464/9463643
    editors = []
    editorNames = []
    for prop in dir(self):
      propObj = getattr(self, prop)
      if not prop.startswith('__') and not callable(propObj):
        editors.append(propObj)
        # Strip 'editor', space at capital letter
        propClsName = type(propObj).__name__
        name = propClsName[:propClsName.index('Editor')]
        name = re.sub(r'(\w)([A-Z])', r'\1 \2', name)
        editorNames.append(name)
    self.editors = editors
    self.editorNames = editorNames

  def registerClass(self, clsParam: ABParam):
    def multiEditorClsDecorator(cls):
      # Since all legwork is done inside the editors themselves, simply call each decorator from here as needed
      for editor in self.editors:
        cls = editor.registerClass(clsParam)(cls)
      return cls
    return multiEditorClsDecorator

  def close(self):
    for editor in self.editors:
      editor.close()
# Encapsulate scheme within class so that changes to the scheme propagate to all GUI elements
AB_SINGLETON = _ABSingleton()