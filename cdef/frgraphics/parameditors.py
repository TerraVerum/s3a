from __future__ import annotations

import re
import sys
import weakref
from dataclasses import dataclass
from functools import partial
from os.path import join
from pathlib import Path
from typing import Collection, Union, Callable, Any, Optional, List, Dict, Tuple, Set, Type

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.parametertree import (Parameter, ParameterTree, parameterTypes)
from pyqtgraph.parametertree.parameterTypes import ListParameter

from cdef.structures import NChanImg, ContainsSharedProps, FRComplexVertices, FRVertices, \
  BlackWhiteImg
from imageprocessing.processing import ImageProcess, Process
from .graphicsutils import dialogGetSaveFileName, saveToFile
from .. import appInst
from ..procwrapper import FRImgProcWrapper
from ..projectvars import (
  MENU_OPTS_DIR, SCHEMES_DIR, GEN_PROPS_DIR, FILTERS_DIR, SHORTCUTS_DIR,
  LAYOUTS_DIR, USER_PROFILES_DIR,
  TEMPLATE_COMP as TC, TEMPLATE_COMP_CLASSES as COMP_CLASSES, FR_CONSTS)
from ..structures import FRIllRegisteredPropError
from ..structures import FRParam

Signal = QtCore.pyqtSignal

def _genList(nameIter, paramType, defaultVal, defaultParam='value'):
  """Helper for generating children elements"""
  return [{'name': name, 'type': paramType, defaultParam: defaultVal} for name in nameIter]


def _frPascalCaseToTitle(name: str) -> str:
  """
  Helper utility to turn a FRPascaleCase name to a 'Title Case' title
  :param name: camel-cased name
  :return: Space-separated, properly capitalized version of :param:`Name`
  """
  if not name:
    return name
  if name.startswith('FR'):
    name = name[2:]
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

def _getAllBases(cls):
  baseClasses = [cls]
  nextClsPtr = 0
  # Get all bases of bases, too
  while nextClsPtr < len(baseClasses):
    curCls = baseClasses[nextClsPtr]
    curBases = curCls.__bases__
    # Only add base classes that haven't already been added to prevent infinite recursion
    baseClasses.extend([tmpCls for tmpCls in curBases if tmpCls not in baseClasses])
    nextClsPtr += 1
  return baseClasses

@dataclass
class FRShortcutCtorGroup:
  constParam: FRParam
  func: Callable
  args: list

class FREditableShortcut(QtWidgets.QShortcut):
  paramIdx: Tuple[FRParam, FRParam]

class FRShortcutParameterItem(parameterTypes.WidgetParameterItem):
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

class FRShortcutParameter(Parameter):
  itemClass = FRShortcutParameterItem

  def __init__(self, **opts):
    # Before initializing super, turn the string keystroke into a key sequence
    value = opts.get('value', '')
    keySeqVal = QtGui.QKeySequence(value)
    opts['value'] = keySeqVal
    super().__init__(**opts)

class FRProcGroupParameter(parameterTypes.GroupParameter):
  def __init__(self, **opts):
    super().__init__(**opts)
    disableFont = QtGui.QFont()
    disableFont.setStrikeOut(True)
    self.enabledFontMap = {True: None, False: disableFont}

  def makeTreeItem(self, depth):
    item = super().makeTreeItem(depth)
    item.contextMenuEvent = lambda ev: item.contextMenu.popup(ev.globalPos())
    act = item.contextMenu.addAction('Toggle Enable')
    self.enabledFontMap[True] = QtGui.QFont(item.font(0))
    def setter():
      # Toggle 'enable' on click
      disabled = self.opts['enabled']
      enabled = not disabled
      item.setFont(0, self.enabledFontMap[enabled])
      for ii in range(item.childCount()):
        item.child(ii).setDisabled(disabled)
      self.opts['enabled'] = enabled
    act.triggered.connect(setter)
    return item

class FRAtomicGroupParameter(parameterTypes.GroupParameter):
  def makeTreeItem(self, depth):
    item = super().makeTreeItem(depth)
    font = QtGui.QFont()
    font.setBold(False)
    item.setFont(0, font)
    return item


class FRNoneParameter(parameterTypes.SimpleParameter):

  def __init__(self, **opts):
    opts['type'] = 'str'
    super().__init__(**opts)
    self.setWritable(False)


parameterTypes.registerParameterType('NoneType', FRNoneParameter)
parameterTypes.registerParameterType('shortcut', FRShortcutParameter)
parameterTypes.registerParameterType('procgroup', FRProcGroupParameter)
parameterTypes.registerParameterType('atomicgroup', FRAtomicGroupParameter)

@dataclass
class FRBoundFnParams:
  param: FRParam
  func: Callable
  defaultFnArgs: list

class FRParamEditor(QtWidgets.QDockWidget):
  sigParamStateCreated = Signal(str)
  sigParamStateUpdated = Signal(dict)

  def __init__(self, parent=None, paramList: List[Dict]=None, saveDir='.',
               fileType='param', saveDlgName='Save As', name=None):
    # Place in list so an empty value gets unpacked into super constructor
    if paramList is None:
      paramList = []

    if name is None:
      try:
        propClsName = type(self).__name__
        name = propClsName[:propClsName.index('Editor')]
        name = _frPascalCaseToTitle(name)
      except ValueError:
        name = "Parameter Editor"

    super().__init__(parent)
    self.hide()
    self.setWindowTitle(name)
    self.setObjectName(name)

    self.boundFnsPerClass: Dict[str, List[FRBoundFnParams]] = {}
    """Holds the parameters associated with this registered class"""

    self.classNameToParamMapping: Dict[str, FRParam] = {}
    """
    Allows the editor to associate a class name with its human-readable parameter
    name
    """

    self.classInstToEditorMapping: Dict[Any, FRParamEditor] = {}
    """
    For editors that register parameters for *other* editors,
     this allows parameters to be updated from the correct editor
    """

    self.instantiatedClassTypes = set()
    """
    Records whether classes with registered parameters have been instantiated. This way,
    base classes with registered parameters but no instances will not appear in the
    parameter editor.
    """

    # -----------
    # Construct parameter tree
    # -----------
    self.params = Parameter(name='Parameters', type='group', children=paramList)
    self.params.sigStateChanged.connect(self._paramTreeChanged)
    self.tree = ParameterTree()
    self.tree.setParameters(self.params, showTop=False)
    self.tree.header().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)

    # -----------
    # Human readable name (for settings menu)
    # -----------
    self.name = name

    # -----------
    # Internal parameters for saving settings
    # -----------
    self.saveDir = saveDir
    self.fileType = fileType
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
    self.dockContentsWidget = QtWidgets.QWidget(parent)
    self.setWidget(self.dockContentsWidget)
    btnLayout = QtWidgets.QHBoxLayout()
    btnLayout.addWidget(self.saveAsBtn)
    btnLayout.addWidget(self.applyBtn)
    btnLayout.addWidget(self.closeBtn)

    centralLayout = QtWidgets.QVBoxLayout(self.dockContentsWidget)
    centralLayout.addWidget(self.tree)
    centralLayout.addLayout(btnLayout)
    # self.setLayout(centralLayout)
    self.tree.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    self.tree.setSizeAdjustPolicy(QtWidgets.QScrollArea.AdjustToContents)
    # -----------
    # UI Element Signals
    # -----------
    self.saveAsBtn.clicked.connect(self.saveAsBtnClicked)
    self.closeBtn.clicked.connect(self.close)
    self.applyBtn.clicked.connect(self.applyBtnClicked)

  def _paramTreeChanged(self, param, child, idx):
    self._stateBeforeEdit = self.params.saveState()

  def _expandCols(self):
    # self.resize(self.tree.width(), self.height())
    self.tree.setColumnWidth(0, self.width()//2)
    # totWidth = 0
    # for colIdx in range(2):
    #   self.tree.resizeColumnToContents(colIdx)
    #   totWidth += self.tree.columnWidth(colIdx) + self.tree.margin
    # appInst.processEvents()
    # self.dockContentsWidget.setMinimumWidth(totWidth)

  # Helper method for accessing simple parameter values
  def __getitem__(self, keys: Union[tuple, FRParam, Collection[FRParam]]):
    """
    Convenience function for accessing child parameters within a parameter editor.
      - If :param:`keys` is a single :class:`FRParam`, the value at that parameter is
        extracted and returned to the user.
      - If :param:`keys` is a :class:`tuple`:

        * The first element of the tuple must correspond to the base name within the
          parameter grouping in order to properly extract the corresponding children.
          For instance, to extract MARGIN from :class:`FRGeneralPropertiesEditor`,
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
    self.params.restoreState(self._stateBeforeEdit, removeChildren=False)
    super().reject()

  def applyBtnClicked(self):
    # Don't emit any signals if nothing changed
    newState = self.params.saveState()
    if self._stateBeforeEdit == newState:
      return
    self._stateBeforeEdit = newState
    outDict = self.params.getValues()
    self.sigParamStateUpdated.emit(outDict)
    return outDict

  def saveAsBtnClicked(self):
    paramState = self.params.saveState(filter='user')
    saveName = dialogGetSaveFileName(self, self._saveDlgName)
    errMsg = self.saveAs(saveName, paramState)
    if isinstance(errMsg, str):
      QtWidgets.QMessageBox().information(self, 'Error During Import', errMsg)


  def saveAs(self, saveName: str=None, paramState: dict=None,
             allowOverwriteDefault=False):
    """
    * Returns dict on successful parameter save and emits sigParamStateCreated.
    * Returns string representing error if the file name was invalid.
    * Returns None if no save name was given
    """
    if saveName is None:
      return None
    if paramState is None:
      paramState = self.params.saveState()
    Path(self.saveDir).mkdir(parents=True, exist_ok=True)
    errMsg = saveToFile(paramState, self.saveDir, saveName, self.fileType,
                        allowOverwriteDefault=allowOverwriteDefault)
    if errMsg is None:
      self.applyBtnClicked()
      outDict: dict = self.params.getValues()
      self.sigParamStateCreated.emit(saveName)
      return outDict
    else:
      return errMsg

  def loadState(self, newStateDict: dict):
    self.params.restoreState(newStateDict, addChildren=False)

  def registerProps(self, clsObj, constParams: List[FRParam]):
    outProps = []
    for param in constParams:
      outProps.append(self.registerProp(clsObj, param))
    return outProps

  def _addParamGroup(self, groupName: str):
    paramForCls = Parameter.create(name=groupName, type='group')
    paramForCls.sigStateChanged.connect(self._paramTreeChanged)
    self.params.addChild(paramForCls)
    return paramForCls

  def registerProp(self, groupingName: Union[type, str], constParam: FRParam,
                   parentParamPath:Collection[str]=None, asProperty=True, **etxraOpts):
    """
    Registers a property defined by *constParam* that will appear in the respective
    parameter editor.

    :param groupingName: If *type* it must be a class. In this case, this parameter will
      be listed under the registered class of the same exact name.
      If *str*, *groupingName* must match the exact same string name passed in during
      :func:`registerGroup <FRParamEditor.registerGroup>`. Otherwise, an error will
      occur.
    :param constParam: Object holding parameter attributes such as name, type,
      help text, etc.
    :param parentParamPath: If None, defaults to the top level of the parameters for the
      current class (or paramHolder). *parentParamPath* represents the parent group
      to whom the newly registered parameter should be added
    :param asProperty: If True, creates a property object bound to getter and setter
      for the new param. Otherwise, returns the param itself. If asProperty is false,
      the returned parameter must be evaluated to obtain a value, e.g.
      x = registerProp(..., asProperty=False); myVal = x.value()
    :param etxraOpts: Extra options passed directly to the created pyqtgraph.Parameter
    :return: Property bound to this value in the parameter editor
    """
    paramOpts = dict(name=constParam.name, type=constParam.valType, tip=constParam.helpText)
    if constParam.valType == 'group':
      paramOpts.update(children=constParam.value)
    else:
      paramOpts.update(value=constParam.value)
    paramOpts.update(etxraOpts)
    paramForEditor = Parameter.create(**paramOpts)

    if isinstance(groupingName, type):
      clsName = groupingName.__qualname__
    else:
      clsName = groupingName
    paramName = self.classNameToParamMapping[clsName].name
    if paramName in self.params.names:
      paramForCls = self.params.child(paramName)
    else:
      paramForCls = self._addParamGroup(paramName)

    if parentParamPath is not None and len(parentParamPath) > 0:
      paramForCls = paramForCls.param(*parentParamPath)
    if constParam.name not in paramForCls.names:
      paramForCls.addChild(paramForEditor)

    self._expandCols()

    if not asProperty:
      return paramForEditor

    def _paramAccessHelper(clsObj):
      # when class isn't fully instantiated
      # Retrieve class name from the class instance, since this function call may
      # have resulted from an inhereted class. This only matters if the class name was
      # used instead of a generic string parameter value
      nonlocal clsName
      if not isinstance(groupingName, str):
        clsName = type(clsObj).__qualname__
      if clsObj in self.classInstToEditorMapping:
        xpondingEditor = self.classInstToEditorMapping[clsObj]
      else:
        xpondingEditor = self
      return clsName, xpondingEditor

    # Else, create a property and return that instead
    @property
    def paramAccessor(clsObj):
      trueClsName, xpondingEditor = _paramAccessHelper(clsObj)
      return xpondingEditor[self.classNameToParamMapping[trueClsName], constParam]

    @paramAccessor.setter
    def paramAccessor(clsObj, newVal):
      trueClsName, xpondingEditor = _paramAccessHelper(clsObj)

      param = xpondingEditor[self.classNameToParamMapping[trueClsName], constParam, True]
      param.setValue(newVal)

    return paramAccessor

  def registerGroup(self, groupParam: FRParam, **opts):
    """
    Intended for use as a class decorator. Registers a class as able to hold
    customizable shortcuts.

    :param groupParam: Parameter holding the name of this class as it should appear
      in this parameter editor. As such, it should be human readable.
    :param opts: Additional registration options. Accepted values:
      - nameFromParam: This is available so objects without a dedicated class can
      also be registered. In that case, a spoof class is registered under the name
      'clsParam.name' instead of '[decorated class].__qualname__'.
      - forceCreate: Normally, the class is not registered until at least one property
      (or method) is registered to it. That way, classes registered for other editors
      don't erroneously appear. *forceCreate* will override this rule, creating this
      parameter immediately.
    :return: Undecorated class, but with a new __init__ method which initializes
      all shared properties contained in the '__initEditorParams__' method, if it exists
      in the class.
    """
    opts['nameFromParam'] = opts.get('nameFromParam', False)
    def classDecorator(cls: Union[Type, Any]=None):
      if cls is None:
        # In this case nameFromParam must be provided. Use a dummy class for the
        # rest of the proceedings
        cls = type('DummyClass', (), {})
      if not isinstance(cls, type):
        # Instance was passed, not class
        cls = type(cls)
      if opts['nameFromParam']:
        clsName = groupParam.name
      else:
        clsName = cls.__qualname__
      oldClsInit = cls.__init__
      self._extendedClassDecorator(cls, groupParam, **opts)

      if opts.get('forceCreate', False):
        self._addParamGroup(clsName)

      self.classNameToParamMapping[clsName] = groupParam
      def newClassInit(clsObj, *args, **kwargs):
        if (cls not in _INITIALIZED_CLASSES
            and issubclass(cls, ContainsSharedProps)):
          _INITIALIZED_CLASSES.add(cls)
          cls.__initEditorParams__()
          superObj = super(cls, clsObj)
          if isinstance(superObj, ContainsSharedProps):
           superObj.__initEditorParams__()

        if opts.get('saveDefault', True):
          self.saveAs(saveName='Default', allowOverwriteDefault=True)
        self.classInstToEditorMapping[clsObj] = self
        retVal = oldClsInit(clsObj, *args, **kwargs)
        self._extendedClassInit(clsObj, groupParam)
        return retVal
      cls.__init__ = newClassInit
      return cls
    if opts['nameFromParam']:
      classDecorator()
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

class FRGeneralPropertiesEditor(FRParamEditor):
  def __init__(self, parent=None):
    super().__init__(parent, paramList=[], saveDir=GEN_PROPS_DIR, fileType='regctrl')

class FRUserProfileEditor(FRParamEditor):
  def __init__(self, parent=None, singletonObj: _FRSingleton=None):
    super().__init__(parent, paramList=[],
                     saveDir=USER_PROFILES_DIR, fileType='cdefprofile')
    optsFromSingletonEditors = []
    for editor in singletonObj.editors:
      curValues = self.getSettingsFiles(editor.saveDir, editor.fileType)
      curParam = ListParameter(name=editor.name, value='Default', values=curValues)
      updateFunc = lambda newName, listParam=curParam: \
        listParam.setLimits(listParam.opts['limits'] + [newName])
      editor.sigParamStateCreated.connect(updateFunc)
      optsFromSingletonEditors.append(curParam)

    _USER_PROFILE_PARAMS = [
      {'name': 'Image', 'type': 'str'},
      {'name': 'Annotations', 'type': 'str'},
      {'name': 'Layout', 'type': 'list', 'values': self.getSettingsFiles(LAYOUTS_DIR, 'dockstate'),
       'value': 'Default'},
    ]
    _USER_PROFILE_PARAMS.extend(optsFromSingletonEditors)
    self.params.addChildren(_USER_PROFILE_PARAMS)

  @staticmethod
  def getSettingsFiles(settingsDir: str, ext: str) -> List[str]:
    files = Path(settingsDir).glob(f'*.{ext}')
    return [file.stem for file in files]

class FRTableFilterEditor(FRParamEditor):
  def __init__(self, parent=None):
    minMaxParam = _genList(['min', 'max'], 'int', 0)
    # Make max 'infinity'
    minMaxParam[1]['value'] = sys.maxsize
    validatedParms = _genList(['Validated', 'Not Validated'], 'bool', True)
    devTypeParam = _genList((param.name for param in COMP_CLASSES), 'bool', True)
    xyVerts = _genList(['X Bounds', 'Y Bounds'], 'group', minMaxParam, 'children')
    _FILTER_PARAMS = [
        {'name': TC.INST_ID.name, 'type': 'group', 'children': minMaxParam},
        {'name': TC.VALIDATED.name, 'type': 'group', 'children': validatedParms},
        {'name': TC.COMP_CLASS.name, 'type': 'group', 'children': devTypeParam},
        {'name': TC.LOGO.name, 'type': 'str', 'value': '.*'},
        {'name': TC.NOTES.name, 'type': 'str', 'value': '.*'},
        {'name': TC.BOARD_TEXT.name, 'type': 'str', 'value': '.*'},
        {'name': TC.DEV_TEXT.name, 'type': 'str', 'value': '.*'},
        {'name': TC.VERTICES.name, 'type': 'group', 'children': xyVerts}
      ]
    super().__init__(parent, paramList=_FILTER_PARAMS, saveDir=FILTERS_DIR, fileType='filter')

class FRShortcutsEditor(FRParamEditor):

  def __init__(self, parent=None):

    self.shortcuts = []
    # Unlike other param editors, these children don't get filled in until
    # after the top-level widget is passed to the shortcut editor
    super().__init__(parent, [], saveDir=SHORTCUTS_DIR, fileType='shortcut')

    # If the registered class is not a graphical widget, the shortcut
    # needs a global context
    allWidgets = pg.mkQApp().topLevelWidgets()
    isGlobalWidget = [isinstance(o, QtWidgets.QMainWindow) for o in allWidgets]
    self.mainWinRef = weakref.proxy(allWidgets[np.argmax(isGlobalWidget)])

  def registerMethod(self, constParam: FRParam, fnArgs=None):
    """
    Designed for use as a function decorator. Registers the decorated function into a list
    of methods known to the :class:`FRShortcutsEditor`. These functions are then accessable from
    customizeable shortcuts.
    """
    if fnArgs is None:
      fnArgs = []

    def registerMethodDecorator(func: Callable, returnClsName=False, fnParentClass=None):
      boundFnParam = FRBoundFnParams(param=constParam, func=func, defaultFnArgs=fnArgs)
      if fnParentClass is None:
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

  def _extendedClassDecorator(self, cls: Any, clsParam: FRParam, **opts):
    self.addRegisteredFuncsFromClass(cls, clsParam)
    super()._extendedClassDecorator(cls, clsParam, **opts)

  def _extendedClassInit(self, clsObj: Any, clsParam: FRParam):
    clsName = type(clsObj).__qualname__
    boundParamList = self.boundFnsPerClass.get(clsName, [])
    for boundParam in boundParamList:
      seqCopy = QtGui.QKeySequence(boundParam.param.value)
      try:
        shortcut = FREditableShortcut(seqCopy, clsObj)
      except TypeError:
        # Occurs when the requested class is not a widget
        shortcut = FREditableShortcut(seqCopy, self.mainWinRef)
      shortcut.paramIdx = (clsParam, boundParam.param)
      shortcut.activated.connect(partial(boundParam.func, clsObj, *boundParam.defaultFnArgs))
      shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
      self.shortcuts.append(shortcut)

  def addRegisteredFuncsFromClass(self, cls: Any, clsParam: FRParam):
    """
    For a given class, adds the registered parameters from that class to the respective
    editor. This is how the dropdown menus in the editors are populated with the
    user-specified variables.

    :param cls: Current class

    :param clsParam: :class:`FRParam` value encapsulating the human readable class name.
           This is how the class will be displayed in the :class:`FRShortcutsEditor`.

    :return: None
    """
    # Make sure to add parameters from registered base classes, too
    iterClasses = []
    baseClasses = _getAllBases(cls)

    for baseCls in baseClasses:
      iterClasses.append(baseCls.__qualname__)

    for clsName in iterClasses:
      classParamList = self.boundFnsPerClass.get(clsName, [])
      # Don't add a category unless at least one list element is present
      if len(classParamList) == 0: continue
      # If a human-readable name was given, replace class name with human name
      paramChildren = []
      paramGroup = {'name': clsParam.name, 'type': 'group',
                    'children': paramChildren}
      for boundFn in classParamList:
        paramForTree = {'name' : boundFn.param.name,
                        'type' : boundFn.param.valType,
                        'value': boundFn.param.value,
                        'tip'  : boundFn.param.helpText}
        paramChildren.append(paramForTree)
      # If this group already exists, append the children to the existing group
      # instead of adding a new child
      if clsParam.name in self.params.names:
        self.params.child(clsParam.name).addChildren(paramChildren)
      else:
        self.params.addChild(paramGroup)

  def registerProp(self, *args, **etxraOpts):
    """
    Properties should never be registered as shortcuts, so make sure this is disallowed
    """
    raise FRIllRegisteredPropError('Cannot register property/attribute as a shortcut')

  def applyBtnClicked(self):
    for shortcut in self.shortcuts: #type: FREditableShortcut
      shortcut.setKey(self[shortcut.paramIdx])
    super().applyBtnClicked()

class FRAlgCollectionEditor(FRParamEditor):
  def __init__(self, saveDir, algMgr: FRAlgPropsMgr, name=None, parent=None):
    self.algMgr = algMgr
    super().__init__(parent, saveDir=saveDir, fileType='alg', name=name)
    algOptDict = {
      'name': 'Algorithm', 'type': 'list', 'values': [], 'value': 'N/A'
    }
    self.treeAlgOpts: Parameter = Parameter(name='Algorithm Selection', type='group', children=[algOptDict])
    self.algOpts: ListParameter = self.treeAlgOpts.children()[0]
    # Since constructor forces self.params to be top level item, we need to reconstruct
    # the tree to avoid this
    self.tree.setParameters(self.algOpts)
    self.algOpts.sigValueChanged.connect(lambda param, proc: self.changeActiveAlg(proc))

    Path(self.saveDir).mkdir(parents=True, exist_ok=True)

    # Allows only the current processor params to be shown in the tree
    #self.tree.addParameters(self.params, showTop=False)

    self.curProcessor: Optional[FRImgProcWrapper] = None
    self.nameToProcMapping: Dict[str, FRImgProcWrapper] = {}
    self._image = np.zeros((1,1), dtype='uint8')

    self.VERT_LST_NAMES = ['fgVerts', 'bgVerts']
    self.vertBuffers: Dict[str, FRComplexVertices] = {
      vType: FRComplexVertices() for vType in self.VERT_LST_NAMES
    }

    wrapped : Optional[FRImgProcWrapper] = None
    for processorCtor in algMgr.processorCtors:
      # Retrieve proc so default can be set after
      wrapped = self.addImageProcessor(processorCtor())
    self.algOpts.setDefault(wrapped)
    self.changeActiveAlg(proc=wrapped)
    self.saveAs('Default', allowOverwriteDefault=True)

  def run(self, **kwargs):
    # for vertsName in self.VERT_LST_NAMES:
    #   curVerts = kwargs[vertsName]
    #   if curVerts is not None:
    #     self.vertBuffers[vertsName].append(curVerts)
    # for vertsName in self.VERT_LST_NAMES:
    #   arg = self.vertBuffers[vertsName].stack()
    #   kwargs[vertsName] = arg
    for name in 'fgVerts', 'bgVerts':
      if kwargs[name] is None:
        kwargs[name] = FRVertices()
    retVal = self.curProcessor.run(**kwargs)
    # self.vertBuffers = {name: FRComplexVertices() for name in self.VERT_LST_NAMES}
    return retVal

  def resultAsVerts(self, localEstimate=True):
    return self.curProcessor.resultAsVerts(localEstimate=localEstimate)

  @property
  def image(self):
    return self._image
  @image.setter
  def image(self, newImg: NChanImg):
    if self.curProcessor is not None:
      self.curProcessor.image = newImg
    self._image = newImg

  def addImageProcessor(self, newProc: ImageProcess):
    processor = FRImgProcWrapper(newProc, self)
    self.tree.addParameters(self.params.child(processor.algName))

    self.nameToProcMapping.update({processor.algName: processor})
    self.algOpts.setLimits(self.nameToProcMapping.copy())
    return processor

  def saveAs(self, saveName: str=None, paramState: dict=None,
             allowOverwriteDefault=False):
    """
    The algorithm editor also needs to store information about the selected algorithm, so lump
    this in with the other parameter information before calling default save.
    """
    paramState = [self.algOpts.value().algName, self.params.saveState()]
    return super().saveAs(saveName, paramState, allowOverwriteDefault)

  def loadState(self, selection_newStatePair: Tuple[str, dict]):
    selectedOpt = selection_newStatePair[0]
    # Get the impl associated with this option name
    isLegit = selectedOpt in self.algOpts.opts['limits']
    if not isLegit:
      selectedImpl = self.algOpts.value()
      msgBox = QtWidgets.QMessageBox
      msgBox.information(self, 'Invalid Selection', f'Selection {selectedOpt} does'
           f' not match the list of available algorithms. Defaulting to {selectedImpl}')
    else:
      selectedImpl = self.algOpts.opts['limits'][selectedOpt]
    self.algOpts.setValue(selectedImpl)
    super().loadState(selection_newStatePair[1])

  def changeActiveAlg(self, proc: FRImgProcWrapper):
    # Hide all except current selection
    # TODO: Find out why hide() isn't working. Documentation indicates it should
    # Instead, use the parentChanged utility as a hacky workaround
    selectedParam = self.params.child(proc.algName)
    for ii, child in enumerate(self.params.children()):
      shouldHide = child is not selectedParam
      # Offset by 1 to account for self.algOpts
      self.tree.setRowHidden(1 + ii, QtCore.QModelIndex(), shouldHide)
    # selectedParam.show()
    self.curProcessor = proc
    self.curProcessor.image = self.image

class FRAlgPropsMgr(FRParamEditor):

  def __init__(self, parent=None):
    super().__init__(parent, fileType='', saveDir='')
    self.processorCtors : List[Callable[[], ImageProcess]] = []
    self.spawnedCollections : List[FRAlgCollectionEditor] = []

  def registerGroup(self, groupParam: FRParam, **opts):
    # Don't save a default file for this class
    return super().registerGroup(groupParam, saveDefault=False, **opts)

  def createProcessorForClass(self, clsObj) -> FRAlgCollectionEditor:
    clsName = type(clsObj).__name__
    editorDir = join(MENU_OPTS_DIR, clsName, '')
    # Strip "FR" from class name before retrieving name
    settingsName = _frPascalCaseToTitle(clsName[2:]) + ' Processor'
    newEditor = FRAlgCollectionEditor(editorDir, self, name=settingsName)
    FR_SINGLETON.editors.append(newEditor)
    self.spawnedCollections.append(weakref.proxy(newEditor))
    # Wrap in property so changes propagate to the calling class
    lims = newEditor.algOpts.opts['limits']
    defaultKey = next(iter(lims))
    defaultAlg = lims[defaultKey]
    newEditor.algOpts.setDefault(defaultAlg)
    newEditor.changeActiveAlg(proc=defaultAlg)
    return newEditor

  def addProcessCtor(self, procCtor: Callable[[], ImageProcess]):
    self.processorCtors.append(procCtor)
    for algCollection in self.spawnedCollections:
      algCollection.addImageProcessor(procCtor())


class FRColorSchemeEditor(FRParamEditor):
  def __init__(self, parent=None):
    super().__init__(parent, paramList=[], saveDir=SCHEMES_DIR, fileType='scheme')


_INITIALIZED_CLASSES: Set[Type[ContainsSharedProps]] = set()
class _FRSingleton:
  algParamMgr = FRAlgPropsMgr()

  shortcuts = FRShortcutsEditor()
  scheme = FRColorSchemeEditor()
  generalProps = FRGeneralPropertiesEditor()
  filter = FRTableFilterEditor()

  annotationAuthor = None

  def __init__(self):
    self.editors: List[FRParamEditor] = \
      [self.scheme, self.shortcuts, self.generalProps, self.filter]
    self.userProfile = FRUserProfileEditor(singletonObj=self)

  def registerGroup(self, clsParam: FRParam, **opts):
    def multiEditorClsDecorator(cls):
      # Since all legwork is done inside the editors themselves, simply call each decorator from here as needed
      for editor in self.editors:
        cls = editor.registerGroup(clsParam, **opts)(cls)
      return cls
    return multiEditorClsDecorator

  def close(self):
    for editor in self.editors:
      editor.close()
# Encapsulate scheme within class so that changes to the scheme propagate to all GUI elements
FR_SINGLETON = _FRSingleton()