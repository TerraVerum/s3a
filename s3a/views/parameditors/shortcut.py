from dataclasses import dataclass
from functools import partial
from inspect import isclass
from typing import Tuple, Callable, Union, Any, Dict, List

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from s3a.graphicsutils import findMainWin
from s3a.projectvars import SHORTCUTS_DIR
from s3a.structures import FRParam, FRParamEditorError
from .genericeditor import FRParamEditor


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

    self.shortcuts = []
    # Unlike other param editors, these children don't get filled in until
    # after the top-level widget is passed to the shortcut editor
    super().__init__(parent, [], saveDir=SHORTCUTS_DIR, fileType='shortcut')

    # If the registered class is not a graphical widget, the shortcut
    # needs a global context
    self.mainWinRef = findMainWin()
    self.boundFnsPerGroup: Dict[FRParam, List[FRBoundFnParams]] = {}
    self._objsWithShortcuts = set()
    """Holds the parameters associated with this registered class"""

  def registerMethod(self, constParam: FRParam, fnArgs: List[Any]=None):
    """
    Designed for use as a function decorator. Registers the decorated function into a list
    of methods known to the :class:`FRShortcutsEditor`. These functions are then accessable from
    customizeable shortcuts.
    """
    if fnArgs is None:
      fnArgs = []

    def registerMethodDecorator(func: Callable):
      boundFnParam = FRBoundFnParams(param=constParam, func=func, defaultFnArgs=fnArgs)
      grouping, _ = _class_fnNamesFromFnQualname(func.__qualname__)
      self._addParamToList(grouping, boundFnParam)
      return func
    return registerMethodDecorator

  def _addParamToList(self, grouping: Any, param: Union[FRParam, FRBoundFnParams]):
    groupParams = self.boundFnsPerGroup.get(grouping, [])
    groupParams.append(param)
    self.boundFnsPerGroup[grouping] = groupParams

  def _extendedClassInit(self, clsObj: Any, groupParam: FRParam):
    grouping = type(clsObj)
    if clsObj in self._objsWithShortcuts:
      return
    self._objsWithShortcuts.add(clsObj)
    try:
      self.addRegisteredFuncsFromGroup(grouping, groupParam)
    except Exception as ex:
      # Already added previously from a different class.
      # This is the easiest way to check error type since pg throws a raw exception,
      # not a subclass
      if 'Already have child' not in str(ex):
        raise
    boundParamList = self.boundFnsPerGroup.get(grouping.__qualname__, [])
    for boundParam in boundParamList:
      seqCopy = QtGui.QKeySequence(boundParam.param.value)
      try:
        shortcut = FREditableShortcut(seqCopy, clsObj)
      except TypeError:
        # Occurs when the requested class is not a widget
        shortcut = FREditableShortcut(seqCopy, self.mainWinRef)
      shortcut.paramIdx = (groupParam, boundParam.param)
      shortcut.activated.connect(partial(boundParam.func, clsObj, *boundParam.defaultFnArgs))
      shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
      self.shortcuts.append(shortcut)

  def addRegisteredFuncsFromGroup(self, grouping: Any, groupParam: FRParam):
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

    for curGrouping in iterGroupings:
      groupParamList = self.boundFnsPerGroup.get(curGrouping, [])
      boundFns_includingBases.extend(groupParamList)
      # Don't add a category unless at least one list element is present
      if len(groupParamList) == 0: continue
      # If a human-readable name was given, replace class name with human name
      paramChildren = []
      paramGroup = {'name': groupParam.name, 'type': 'group',
                    'children': paramChildren}
      for boundFn in groupParamList:
        paramForTree = {'name' : boundFn.param.name,
                        'type' : boundFn.param.valType,
                        'value': boundFn.param.value,
                        'tip'  : boundFn.param.helpText}
        paramChildren.append(paramForTree)
      # If this group already exists, append the children to the existing group
      # instead of adding a new child
      if groupParam.name in self.params.names:
        self.params.child(groupParam.name).addChildren(paramChildren)
      else:
        self.params.addChild(paramGroup)

    # Now make sure when props are registered for this grouping, they include
    # base class shortcuts too
    self.boundFnsPerGroup[grouping.__qualname__] = boundFns_includingBases

  def registerProp(self, *args, **etxraOpts):
    """
    Properties should never be registered as shortcuts, so make sure this is disallowed
    """
    raise FRParamEditorError('Cannot register property/attribute as a shortcut')

  def applyChanges(self):
    for shortcut in self.shortcuts: #type: FREditableShortcut
      shortcut.setKey(self[shortcut.paramIdx])
    super().applyChanges()