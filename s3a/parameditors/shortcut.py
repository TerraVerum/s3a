from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from inspect import isclass
from typing import Tuple, Callable, Any, Dict, List, DefaultDict

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from pyqtgraph.parametertree import Parameter

from s3a.constants import SHORTCUTS_DIR
from s3a.structures import FRParam, FRParamEditorError
from .genericeditor import FRParamEditor
from .pgregistered import FRShortcutParameter
from s3a.generalutils import helpTextToRichText


def _class_fnNamesFromFnQualname(qualname: str) -> (str, str):
  """
  From the fully qualified function name (e.g. module.class.fn), return the function
  name and class name (module.class, fn).
  :param qualname: output of fn.__qualname__
  :return: (clsName, fnName)
  """
  lastDotIdx = qualname.rfind('.')
  fnName = qualname
  if lastDotIdx < 0:
    # This function isn't inside a class, so defer
    # to the global namespace
    fnParentClass = 'Global'
  else:
    # Get name of class containing this function
    fnParentClass = qualname[:lastDotIdx]
    fnName = qualname[lastDotIdx+1:]
  return fnParentClass, fnName


def _getAllBases(cls):
  baseClasses = [cls]
  nextClsPtr = 0
  if not isclass(cls):
    return baseClasses
  # Get all bases of bases, too
  while nextClsPtr < len(baseClasses):
    curCls = baseClasses[nextClsPtr]
    curBases = curCls.__bases__
    # Only add base classes that haven't already been added to prevent infinite recursion
    baseClasses.extend([tmpCls for tmpCls in curBases if tmpCls not in baseClasses])
    nextClsPtr += 1
  return baseClasses


class FREditableShortcut(QtWidgets.QShortcut):
  paramIdx: Tuple[FRParam, FRParam]


@dataclass
class FRBoundFnParams:
  param: FRParam
  func: Callable
  defaultFnArgs: list


class FRShortcutsEditor(FRParamEditor):

  def __init__(self, parent=None):

    self.paramToShortcutMapping: Dict[FRParam, QtWidgets.QShortcut] = {}
    # Unlike other param editors, these children don't get filled in until
    # after the top-level widget is passed to the shortcut editor
    super().__init__(parent, [], saveDir=SHORTCUTS_DIR, fileType='shortcut',
                     name='Tool Shortcuts')

    # If the registered class is not a graphical widget, the shortcut
    # needs a global context
    self.mainWinRef = None
    self.boundFnsPerQualname: Dict[FRParam, List[FRBoundFnParams]] = defaultdict(list)
    self.objToShortcutParamMapping: DefaultDict[Any, List[FRParam]] = defaultdict(list)
    """Holds which objects have what shortcuts. Useful for avoiding duplicate shortcuts
    for the same action on one class"""
    # Allow global shortcuts
    self.globalParam = FRParam('Global')
    self.groupingToParamMapping[type(None)] = self.globalParam

  def registerMethod_cls(self, constParam: FRParam, fnArgs: List[Any]=None, forceCreate=False):
    """
    Designed for use as a function decorator. Registers the decorated function into a list
    of methods known to the :class:`FRShortcutsEditor`. These functions are then accessable from
    customizeable shortcuts.
    """
    if fnArgs is None:
      fnArgs = []

    def registerMethodDecorator(func: Callable, ownerObj: Any=None):
      boundFnParam = FRBoundFnParams(param=constParam, func=func, defaultFnArgs=fnArgs)
      ownerCls = ownerObj if isclass(ownerObj) else type(ownerObj)
      if ownerObj is None:
        qualname, _ = _class_fnNamesFromFnQualname(func.__qualname__)
      else:
        qualname = ownerCls.__qualname__
      self.boundFnsPerQualname[qualname].append(boundFnParam)
      # Make sure it's an object not just a class before forcing extended init
      groupParam = self.groupingToParamMapping[ownerCls]
      if ownerObj is not None and ownerCls != ownerObj:
        self._extendedClassInit(ownerObj, groupParam)
      elif forceCreate:
        self.addRegisteredFuncsFromCls(ownerCls, groupParam)
      # Global shortcuts won't get created since no 'global' widget init will be
      # called. In those cases, create right away
      if ownerCls == type(None):
        self.hookupShortcutForBoundFn(boundFnParam, None)
      return func
    return registerMethodDecorator

  def _extendedClassInit(self, clsObj: Any, groupParam: FRParam):
    grouping = type(clsObj)
    try:
      self.addRegisteredFuncsFromCls(grouping, groupParam)
    except Exception as ex:
      # Already added previously from a different class.
      # This is the easiest way to check error type since pg throws a raw exception,
      # not a subclass
      if 'Already have child' not in str(ex):
        raise
    boundParamList = self.boundFnsPerQualname.get(grouping.__qualname__, [])
    for boundParam in boundParamList:
      self.hookupShortcutForBoundFn(boundParam, clsObj)

  def createRegisteredButton(self, btnParam: FRParam, ownerObj: Any, doRegister=True,
                             baseBtn: QtWidgets.QAbstractButton=None):
    if baseBtn is not None:
      newBtn = baseBtn
      tooltipText = btnParam.helpText
    elif 'icon' in btnParam.opts:
      newBtn = QtWidgets.QPushButton(QtGui.QIcon(btnParam.opts['icon']), '', self)
      tooltipText = helpTextToRichText(btnParam.helpText, btnParam.name)
    else:
      newBtn = QtWidgets.QPushButton(btnParam.name, self)
      tooltipText = btnParam.helpText
    if btnParam.value is None or not doRegister:
      # Either the shortcut wasn't given a value or wasn't requested, or already exists
      return newBtn

    if isclass(ownerObj):
      self.registerMethod_cls(btnParam, forceCreate=True)(
        lambda *args: newBtn.clicked.emit(), ownerObj)
      clsParam = self.groupingToParamMapping[ownerObj]
      param = self[clsParam, btnParam, True]
    else:
      param = self.registerMethod_obj(lambda *args: newBtn.clicked.emit(), btnParam, ownerObj)

    param.opts['tip'] = tooltipText

    def shcChanged(_param, newSeq: str):
      newTooltipText = f'Shortcut: {newSeq}'
      tip = param.opts["tip"]
      tip = helpTextToRichText(tip, newTooltipText)
      newBtn.setToolTip(tip)

    param.sigValueChanged.connect(shcChanged)
    shcChanged(None, btnParam.value)

    return newBtn

  def registerMethod_obj(self, func: Callable[[], Any], funcFrParam: FRParam, ownerObj: Any,
                         *funcArgs, **funcKwargs):
    cls = type(ownerObj)
    try:
      groupParam = self.groupingToParamMapping[cls]
    except KeyError:
      # Not yet registered
      raise FRParamEditorError(f'{cls} must be registered as a group before any buttons'
                               f' can be reigstered to {ownerObj}')
    shortcutParam: FRShortcutParameter = Parameter.create(name=funcFrParam.name,
                                                          type='shortcut', value=funcFrParam.value,
                                                          tip=funcFrParam.helpText,
                                                          frParam=funcFrParam)
    if groupParam.name in self.params.names:
      pgParam = self.params.child(groupParam.name)
    else:
      pgParam = Parameter.create(name=groupParam.name, type='group')
      self.params.addChild(pgParam)
    if funcFrParam.name not in pgParam.names:
      pgParam.addChild(shortcutParam)
    else:
      shortcutParam = pgParam.child(shortcutParam.name())

    seq = shortcutParam.seqEdit.keySequence()
    if ownerObj is None or not isinstance(ownerObj, QtWidgets.QWidget):
      newShortcut = QtWidgets.QShortcut(seq, self.parent())
      ctx = QtCore.Qt.ApplicationShortcut
    else:
      newShortcut = QtWidgets.QShortcut(seq, ownerObj)
      ctx = QtCore.Qt.WidgetWithChildrenShortcut

    newShortcut.setContext(ctx)
    partialFn = partial(func, *funcArgs, **funcKwargs)
    newShortcut.activated.connect(partialFn)
    # newShortcut.activatedAmbiguously.connect(lambda: print(f'{ownerObj} shc ambiguous: {newShortcut.key().toString()}'))
    shortcutParam.sigValueChanged.connect(lambda param: newShortcut.setKey(param.seqEdit.keySequence()))
    if funcFrParam in self.paramToShortcutMapping:
      # Already found this value before, make sure to remove it
      self.paramToShortcutMapping[funcFrParam].deleteLater()
    self.paramToShortcutMapping[funcFrParam] = newShortcut
    return shortcutParam

  def deleteShortcut(self, shortcutFrParam: FRParam):
    shc = self.mappin


  def addRegisteredFuncsFromCls(self, grouping: Any, groupParam: FRParam):
    """
    For a given class, adds the registered parameters from that class to the respective
    editor. This is how the dropdown menus in the editors are populated with the
    user-specified variables.

    :param grouping: Current class

    :param groupParam: :class:`FRParam` value encapsulating the human readable class name.
           This is how the class will be displayed in the :class:`FRShortcutsEditor`.

    :return: None
    """
    # Make sure to add parameters from registered base classes, too, if they exist
    iterGroupings = []
    baseClasses = _getAllBases(grouping)
    for baseCls in baseClasses:
      iterGroupings.append(baseCls.__qualname__)
    boundFns_includingBases = []

    # If this group already exists, append the children to the existing group
    # instead of adding a new child
    if groupParam.name in self.params.names:
      ownerParam = self.params.child(groupParam.name)
      shouldAdd = False
    else:
      ownerParam = Parameter.create(name=groupParam.name, type='group')
      shouldAdd = True
    for curGrouping in iterGroupings:
      groupParamList = self.boundFnsPerQualname.get(curGrouping, [])
      boundFns_includingBases.extend(groupParamList)
      for boundFn in groupParamList:
        self.addBoundFn(boundFn, ownerParam)
    if shouldAdd and len(ownerParam.children()) > 0:
      self.params.addChild(ownerParam)

    # Now make sure when props are registered for this grouping, they include
    # base class shortcuts too
    self.boundFnsPerQualname[grouping.__qualname__] = boundFns_includingBases

  def addBoundFn(self, boundFn: FRBoundFnParams, parentParam: Parameter):
    if boundFn.param.name in parentParam.names:
      # Already registered
      return
    paramForTree = {'name' : boundFn.param.name,
                    'type' : 'shortcut',
                    'value': boundFn.param.value,
                    'tip'  : boundFn.param.helpText}
    parentParam.addChild(paramForTree)

  def hookupShortcutForBoundFn(self, boundFn: FRBoundFnParams, ownerObj: Any):
    if boundFn.param in self.objToShortcutParamMapping[ownerObj]:
      return
    self.objToShortcutParamMapping[ownerObj].append(boundFn.param)
    groupParam = self.groupingToParamMapping[type(ownerObj)]
    seqCopy = QtGui.QKeySequence(boundFn.param.value)
    if ownerObj is None or not isinstance(ownerObj, QtWidgets.QWidget):
      shortcut = QtWidgets.QShortcut(seqCopy, self.parent())
      ctx = QtCore.Qt.ApplicationShortcut
    else:
      shortcut = QtWidgets.QShortcut(seqCopy, ownerObj)
      ctx = QtCore.Qt.WidgetWithChildrenShortcut
    keySeqParam: FRShortcutParameter = self[groupParam, boundFn.param, True]
    shortcut.paramIdx = (groupParam, boundFn.param)
    shortcut.activated.connect(partial(boundFn.func, ownerObj, *boundFn.defaultFnArgs))
    shortcut.activatedAmbiguously.connect(lambda: print('ambiguous'))
    shortcut.setContext(ctx)
    keySeqParam.sigValueChanged.connect(lambda item, value: shortcut.setKey(value))
    self.paramToShortcutMapping[boundFn.param] = shortcut
    return shortcut

  def setParent(self, parent: QtWidgets.QWidget, *args):
    super().setParent(parent, *args)
    # When this parent is set, make sure application-level shortcuts have that parent
    for shortcut in self.paramToShortcutMapping.values():
      if shortcut.context() == QtCore.Qt.ApplicationShortcut:
        shortcut.setParent(parent)

  def registerProp(self, *args, **etxraOpts):
    """
    Properties should never be registered as shortcuts, so make sure this is disallowed
    """
    raise FRParamEditorError('Cannot register property/attribute as a shortcut')