import weakref
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Callable, Union, Any, Dict, List

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from .genericeditor import FRParamEditor
from s3a.projectvars import SHORTCUTS_DIR
from s3a.structures import FRParam, FRIllRegisteredPropError
from ..graphicsutils import findMainWin


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
    self.mainWinRef = weakref.proxy(findMainWin())
    self.boundFnsPerClass: Dict[str, List[FRBoundFnParams]] = {}
    """Holds the parameters associated with this registered class"""

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