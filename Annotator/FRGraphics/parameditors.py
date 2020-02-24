# -*- coding: utf-8 -*-
from __future__ import annotations
import pickle as pkl
import re
import sys
from dataclasses import dataclass
from functools import partial
from os.path import join
from pathlib import Path
from typing import Sequence, Union, Callable, Any, Optional, List, Dict

import numpy as np
from asn1crypto.core import Any

from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.parametertree import (Parameter, ParameterTree, parameterTypes)

from Annotator.constants import MENU_OPTS_DIR
from Annotator.interfaces import FRImageProcessor
from .graphicsutils import dialogSaveToFile
from .. import appInst
from ..constants import (
  SCHEMES_DIR, GEN_PROPS_DIR, FILTERS_DIR, SHORTCUTS_DIR, CLICK_MODIFIERS_DIR,
  TEMPLATE_COMP as TC, TEMPLATE_COMP_TYPES as COMP_TYPES, FR_CONSTS)
from ..exceptions import FRIllRegisteredPropError
from ..params import FRParam

Signal = QtCore.pyqtSignal

def _genList(nameIter, paramType, defaultVal, defaultParam='value'):
  """Helper for generating children elements"""
  return [{'name': name, 'type': paramType, defaultParam: defaultVal} for name in nameIter]


def _camelCaseToTitle(name: str) -> str:
  """
  Helper utility to turn a CamelCase name to a 'Title Case' title
  :param name: camel-cased name
  :return: Space-separated, properly capitalized version of :param:`Name`
  """
  if not name:
    return name
  else:
    name = re.sub(r'(\w)([A-Z])', r'\1 \2', name)
    return name.title()

def _class_fnNamesFromFnQualname(qualname: str) -> (str, str):
  """
  From the fully qualified function name (e.g. module.class.fn), return the function
  name and class name (module.class, fn).
  :param qualname: output of fn.__qualname__
  :return: (clsName, fnName)
  """
  lastDotIdx = qualname.find('.')
  fnName = qualname
  if lastDotIdx < 0:
    # This function isn't inside a class, so defer
    # to the global namespace
    fnParentClass = 'Global'
  else:
    # Get name of class containing this function
    fnParentClass = qualname[:lastDotIdx]
    fnName = qualname[lastDotIdx:]
  return fnParentClass, fnName


@dataclass
class FRShortcutCtorGroup:
  constParam: FRParam
  func: Callable
  args: list

class FREditableShortcut(QtWidgets.QShortcut):
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
    item.value = item.keySequence
    item.setValue = item.setKeySequence
    self.item = item
    return self.item

  def updateDisplayLabel(self, value=None):
    # Make sure the key sequence is human readable
    self.displayLabel.setText(self.widget.keySequence().toString())

  # def contextMenuEvent(self, ev: QtGui.QContextMenuEvent):
  #   menu = self.contextMenu
  #   delAct = QtWidgets.QAction('Set Blank')
  #   delAct.triggered.connect(lambda: self.widget.setValue(''))
  #   menu.addAction(delAct)
  #   menu.exec(ev.globalPos())

class ShortcutParameter(Parameter):
  itemClass = ShortcutParameterItem

  def __init__(self, **opts):
    # Before initializing super, turn the string keystroke into a key sequence
    value = opts.get('value', '')
    keySeqVal = QtGui.QKeySequence(value)
    opts['value'] = keySeqVal
    super().__init__(**opts)

parameterTypes.registerParameterType('shortcut', ShortcutParameter)

@dataclass
class FRBoundFnParams:
  param: FRParam
  func: Callable
  defaultFnArgs: list

class ConstParamWidget(QtWidgets.QDialog):
  sigParamStateCreated = Signal(str)
  sigParamStateUpdated = Signal(dict)

  def __init__(self, parent=None, paramList: List[Dict]=None, saveDir='.',
               saveExt='param', saveDlgName='Save As', name=None):
    # Place in list so an empty value gets unpacked into super constructor
    if paramList is None:
      paramList = []

    super().__init__(parent)
    self.setWindowTitle('Parameter Editor')
    self.resize(500, 400)

    self.boundFnsPerClass: Dict[str, List[FRBoundFnParams]] = {}
    self.classNameToParamMapping: Dict[str, FRParam] = {}
    self.classInstToEditorMapping: Dict[Any, ConstParamWidget] = {}


    # -----------
    # Construct parameter tree
    # -----------
    self.params = Parameter(name='Parameters', type='group', children=paramList)
    self.tree = ParameterTree()
    self.tree.setParameters(self.params, showTop=False)

    # Allow the user to change column widths
    for colIdx in range(2):
      self.tree.header().setSectionResizeMode(colIdx, QtWidgets.QHeaderView.Interactive)

    # -----------
    # Human readable name (for settings menu)
    # -----------
    self.name = name

    # -----------
    # Internal parameters for saving settings
    # -----------
    self.saveDir = saveDir
    self.fileType = saveExt
    self._saveDlgName = saveDlgName
    self._stateBeforeEdit = self.params.saveState()

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
  def __getitem__(self, keys: Union[tuple, FRParam, Sequence[FRParam]]):
    """
    Convenience function for accessing child parameters within a parameter editor.
      - If :param:`keys` is a single :class:`FRParam`, the value at that parameter is
        extracted and returned to the user.
      - If :param:`keys` is a :class:`tuple`:

        * The first element of the tuple must correspond to the base name within the
          parameter grouping in order to properly extract the corresponding children.
          For instance, to extract MARGIN from :class:`GeneralPropertiesEditor`,
              you must first specify the group parent for that parameter:
              >>> margin = FR_SINGLETON.generalProps[FR_CONSTS.CLS_FOCUSED_IMG_AREA,
              >>>   FR_CONSTS.MARGIN]
        * The second parameter must be a signle :class:`FRParam` objects or a sequence
          of :class:`FRParam` objects. If a sequence is given, a list of output values
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
    for curKey in keys: # type: FRParam
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

  def registerProp(self, constParam: FRParam):
    # First add registered property to self list
    def funcWrapper(func):
      func, clsName = self.registerMethod(constParam)(func, True)

      @property
      def paramGetter(*args, **kwargs):
        # Use function wrapper instead of directly returning so no errors are thrown when class isn't fully instantiated
        xpondingEditor = self.classInstToEditorMapping[args[0]]
        return xpondingEditor[self.classNameToParamMapping[clsName], constParam]
      return paramGetter
    return funcWrapper

  def registerMethod(self, constParam: FRParam, fnArgs=None):
    """
    Designed for use as a function decorator. Registers the decorated function into a list
    of methods known to the :class:`ShortcutsEditor`. These functions are then accessable from
    customizeable shortcuts.
    """
    if fnArgs is None:
      fnArgs = []

    def registerMethodDecorator(func: Callable, returnClsName=False):
      boundFnParam = FRBoundFnParams(param=constParam, func=func, defaultFnArgs=fnArgs)
      fnParentClass, _ = _class_fnNamesFromFnQualname(func.__qualname__)

      self._addParamToList(fnParentClass, boundFnParam)
      if returnClsName:
        return func, fnParentClass
      else:
        return func
    return registerMethodDecorator

  def _addParamToList(self, clsName: str, param: Union[FRParam, FRBoundFnParams]):
    clsParams = self.boundFnsPerClass.get(clsName, [])
    clsParams.append(param)
    self.boundFnsPerClass[clsName] = clsParams

  def registerClass(self, clsParam: FRParam, **opts):
    """
    Intended for use as a class decorator. Registers a class as able to hold
    customizable shortcuts.
    """
    def classDecorator(cls):
      clsName = cls.__qualname__
      self.addParamsFromClass(clsName, clsParam)
      # Now that class params are registered, save off default file
      if opts.get('saveDefault', True):
        Path(self.saveDir).mkdir(parents=True, exist_ok=True)
        with open(join(self.saveDir, f'Default.{self.fileType}'), 'wb') as ofile:
          pkl.dump(self.params.saveState(), ofile)
      self.classNameToParamMapping[clsName] = clsParam
      oldClsInit = cls.__init__
      self._extendedClassDecorator(cls, clsParam, **opts)
      def newClassInit(clsObj, *args, **kwargs):
        self.classInstToEditorMapping[clsObj] = self
        retVal = oldClsInit(clsObj, *args, **kwargs)
        self._extendedClassInit(clsObj, clsParam)
        return retVal
      cls.__init__ = newClassInit
      return cls
    return classDecorator

  def _extendedClassInit(self, clsObj: Any, clsParam: FRParam):
    """
    For editors that need to perform any initializations within the decorated class,
      they must be able to access the decorated class' *init* function and modify it.
      Allow this by providing an overloadable stub that is inserted into the decorated
      class *init*.
    """
    return

  def _extendedClassDecorator(self, cls: Any, clsParam: FRParam, **opts):
    """
    Editors needing additional class decorator boilerplates will place it in this overloaded function
    """

  def addParamsFromClass(self, clsName, clsParam: FRParam):
    """
    Once the top-level widget is set, we can construct the
    parameter editor widget. Set the parent of each shortcut so they
    can be used when focusing the main window, then construct the
    editor widget.

    :param clsName: Fully qualified name of the class

    :param clsParam: :class:`FRParam` value encapsulating the human readable class name.
           This is how the class will be displayed in the :class:`ShortcutsEditor`.

    :return: None
    """
    classParamList = self.boundFnsPerClass.get(clsName, [])
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
    # If this group already exists, append the children to the existing group
    # instead of adding a new child
    paramExists = False
    existingParamIdx = None
    for ii, param in enumerate(self.params.childs):
      if param.name() == clsParam.name:
        paramExists = True
        existingParamIdx = ii
        break
    if paramExists:
      self.params.childs[ii].addChildren(paramChildren)
    else:
      self.params.addChild(paramGroup)
    # Make sure all new names are properly displayed
    self.tree.resizeColumnToContents(0)
    self._stateBeforeEdit = self.params.saveState()

class GeneralPropertiesEditor(ConstParamWidget):
  def __init__(self, parent=None):
    super().__init__(parent, paramList=[], saveDir=GEN_PROPS_DIR, saveExt='regctrl')

class ClickModifiersEditor(ConstParamWidget):
  def __init__(self, parent=None):
    super().__init__(parent, saveDir=CLICK_MODIFIERS_DIR, saveExt='modifier')

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
    super().__init__(parent, paramList=_FILTER_DICT, saveDir=FILTERS_DIR, saveExt='filter')

class ShortcutsEditor(ConstParamWidget):

  def __init__(self, parent=None):

    self.shortcuts = []
    # Unlike other param editors, these children don't get filled in until
    # after the top-level widget is passed to the shortcut editor
    super().__init__(parent, [], saveDir=SHORTCUTS_DIR, saveExt='shortcut')

  def _extendedClassInit(self, clsObj: Any, clsParam: FRParam):
    clsName = type(clsObj).__qualname__
    boundParamList = self.boundFnsPerClass.get(clsName, [])
    for boundParam in boundParamList:
      appInst.topLevelWidgets()
      seqCopy = QtGui.QKeySequence(boundParam.param.value)
      # If the registered class is not a graphical widget, the shortcut
      # needs a global context
      allWidgets = appInst.topLevelWidgets()
      isGlobalWidget = [isinstance(o, QtWidgets.QMainWindow) for o in allWidgets]
      mainWin = allWidgets[np.argmax(isGlobalWidget)]

      try:
        shortcut = FREditableShortcut(seqCopy, clsObj)
      except TypeError:
        shortcut = FREditableShortcut(seqCopy, mainWin)
      shortcut.paramIdx = (clsParam, boundParam.param)
      shortcut.activated.connect(partial(boundParam.func, clsObj, *boundParam.defaultFnArgs))
      shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
      self.shortcuts.append(shortcut)

  def registerProp(self, constParam: FRParam):
    """
    Properties should never be registered as shortcuts, so make sure this is disallowed
    """
    raise FRIllRegisteredPropError('Cannot register property/attribute as a shortcut')

  def applyBtnClicked(self):
    for shortcut in self.shortcuts: #type: FREditableShortcut
      shortcut.setKey(self[shortcut.paramIdx])
    super().applyBtnClicked()

class AlgorithmPropertiesEditor(ConstParamWidget):
  def __init__(self, saveDir, algMgr: AlgPropsMgr, name=None, parent=None):
    self.algMgr = algMgr
    super().__init__(parent, saveDir=saveDir, saveExt='alg', name=name)
    algOptDict = {
      'name': 'Algorithm', 'type': 'list', 'values': [], 'value': 'N/A'
    }
    self.treeAlgOpts: Parameter = Parameter(name='Algorithm Selection', type='group', children=[algOptDict])
    self.algOpts = self.treeAlgOpts.children()[0]
    # Since constructor forces self.params to be top level item, we need to reconstruct
    # the tree to avoid this
    self.tree.setParameters(self.algOpts, showTop=False)
    self.tree.addParameters(self.params, showTop=False)
    # self.params.addChild(self.algOpts)
    self.algOpts.sigValueChanged.connect(self.changeActiveAlg)

    self.build_attachParams(algMgr)

    self.curProcessor: Optional[FRImageProcessor] = None
    self.processors: List[FRImageProcessor] = []

  def build_attachParams(self, algMgr: AlgPropsMgr):
    # Step 1: Construct parameter tree
    params = algMgr.params.opts.copy()
    self.params.clearChildren()
    self.params.addChildren(params['children'])

    # Step 2: Instantiate all processor algorithms
    for processorCtor in algMgr.processorCtors:
      processor = processorCtor()
      self.processors.append(processor)
      # Step 3: For each instantiated process, hook up accessor functions to self's
      #         parameter tree
      algMgr.classInstToEditorMapping[processor] = self

      procName = type(processor).__qualname__
      clsName, procName = _class_fnNamesFromFnQualname(procName)
      procParam = algMgr.classNameToParamMapping[clsName]
      procBoundFnList = algMgr.boundFnsPerClass[clsName]

    # Step 4: Determine the active processor object
    pass

  def changeActiveAlg(self, _param: Parameter, newAlgName: FRParam):
    # Copy from opts intead of directly accessing the parameter to avoid
    # overwriting the values in the original manager
    newChildren: list = self.algMgr.params.child(newAlgName).opts['children']
    self.params.clearChildren()
    self.params = Parameter(name='Parameters', type='group',
                            children=newChildren)
    # self.params.addChild(self.algOpts)
    # self.params.addChildren(newChildren)
    self.tree.addParameters(self.params, showTop=False)

class AlgPropsMgr(ConstParamWidget):

  def __init__(self, parent=None):
    super().__init__(parent, saveExt='', saveDir='')
    # self.algPropEditors = [AlgorithmPropertiesEditor(ALG_FOC_IMG_DIR, self),
    #                        AlgorithmPropertiesEditor(ALG_MAIN_IMG_DIR, self)]
    self.algPropEditors: List[AlgorithmPropertiesEditor] = []
    self.processorCtors : List[Callable[[Any,...], FRImageProcessor]] = []

  def registerClass(self, clsParam: FRParam, **opts):
    # Don't save a default file for this class
    return super().registerClass(clsParam, saveDefault=False)

  def _extendedClassDecorator(self, cls: Any, clsParam: FRParam, **opts):
    ctorArgs = opts.get('args', [])
    procCtor = partial(cls.__init__, *ctorArgs)
    self.processorCtors.append(procCtor)

  # def _extendedClassDecorator(self, cls: Any, clsParam: FRParam):
  #   # When an algorithm is added to the manager, attach dropdown options
  #   # to each class using editable algorithms
  #   for editor in self.algPropEditors:  # type: AlgorithmPropertiesEditor
  #     newLimits = editor.algOpts.opts.get('limits', []) + [clsParam.name]
  #     editor.algOpts.setLimits(newLimits)
  #     editor.algOpts.setDefault(newLimits[0])

  def createProcessorForClass(self, cls, ctorArgs: List[Any]=None) -> FRImageProcessor:
    if ctorArgs is None:
      ctorArgs = []
    clsName = cls.__name__
    editorDir = join(MENU_OPTS_DIR, clsName)
    newEditor = AlgorithmPropertiesEditor(editorDir, self, name=_camelCaseToTitle(clsName))
    self.algPropEditors.append(newEditor)
    return newEditor.curProcessor



class SchemeEditor(ConstParamWidget):
  def __init__(self, parent=None):
    super().__init__(parent, paramList=[], saveDir=SCHEMES_DIR, saveExt='scheme')

class _FRSingleton:
  algParamMgr_ = AlgPropsMgr()

  shortcuts = ShortcutsEditor()
  scheme = SchemeEditor()
  generalProps = GeneralPropertiesEditor()
  filter = TableFilterEditor()
  clickModifiers = ClickModifiersEditor()

  annotationAuthor = None

  def __init__(self):
    # editors = []
    # editorNames = []
    # Code retrieved from https://stackoverflow.com/a/20214464/9463643
    # for prop in dir(self):
    #   propObj = getattr(self, prop)
    #   if isinstance(propObj, ConstParamWidget) \
    #       and prop[-1] != '_':
    #     editors.append(propObj)
    #     # Strip 'editor', space at capital letter
    #     propClsName = type(propObj).__name__
    #     name = propClsName[:propClsName.index('Editor')]
    #     name = re.sub(r'(\w)([A-Z])', r'\1 \2', name)
    #     editorNames.append(name)
    # self.editors = editors
    # self.editorNames = editorNames
    self.editors: List[ConstParamWidget] =\
      [self.scheme, self.shortcuts, self.generalProps, self.filter, self.clickModifiers,
       *self.algParamMgr_.algPropEditors]
    self.editorNames: List[str] = []
    for editor in self.editors:
      if editor.name is not None:
        self.editorNames.append(editor.name)
      else:
        propClsName = type(editor).__name__
        name = propClsName[:propClsName.index('Editor')]
        name = _camelCaseToTitle(name)
        self.editorNames.append(name)

  def registerClass(self, clsParam: FRParam):
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
FR_SINGLETON = _FRSingleton()