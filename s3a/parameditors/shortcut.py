from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from inspect import isclass
from typing import Tuple, Callable, Any, Dict, List, DefaultDict, Sequence
from warnings import warn

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from pyqtgraph.parametertree import Parameter

from s3a.constants import SHORTCUTS_DIR
from s3a.structures import FRParam, ParamEditorError, S3AWarning
from .genericeditor import ParamEditor
from .pgregistered import ShortcutParameter
from s3a.generalutils import helpTextToRichText, getParamChild, pascalCaseToTitle, \
  getAllBases


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


class EditableShortcut(QtWidgets.QShortcut):
  paramIdx: Tuple[FRParam, FRParam]


@dataclass
class BoundFnParams:
  param: FRParam
  func: Callable
  defaultFnArgs: list


class ShortcutsEditor(ParamEditor):

  def __init__(self, parent=None):

    self.paramToShortcutMapping: Dict[Tuple[FRParam, Any], QtWidgets.QShortcut] = {}
    """
    Each shortcut must have a unique name, otherwise the registration process will
    leave hanging, untriggerable shortcuts. However, shortcuts with the same name
    under a *different* parent are fine. To catch these cases, the unique key for
    each shortcut is a combination of its FRParam and parent/ownerObj.
    """
    # Unlike other param editors, these children don't get filled in until
    # after the top-level widget is passed to the shortcut editor
    super().__init__(parent, [], saveDir=SHORTCUTS_DIR, fileType='shortcut',
                     name='Tool Shortcuts')

    # If the registered class is not a graphical widget, the shortcut
    # needs a global context
    self.mainWinRef = None
    self.boundFnsPerQualname: Dict[FRParam, List[BoundFnParams]] = defaultdict(list)
    self.objToShortcutParamMapping: DefaultDict[Any, List[FRParam]] = defaultdict(list)
    """Holds which objects have what shortcuts. Useful for avoiding duplicate shortcuts
    for the same action on one class"""
    # Allow global shortcuts
    self.globalParam = FRParam('Global')

  def registerMethod_cls(self, constParam: FRParam, fnArgs: List[Any]=None, forceCreate=False):
    """
    Designed for use as a function decorator. Registers the decorated function into a list
    of methods known to the :class:`FRShortcutsEditor`. These functions are then accessable from
    customizeable shortcuts.
    """
    if fnArgs is None:
      fnArgs = []

    def registerMethodDecorator(func: Callable, ownerObj: Any=None, namePath: Sequence[str]=()):
      boundFnParam = BoundFnParams(param=constParam, func=func, defaultFnArgs=fnArgs)
      ownerCls = ownerObj if isclass(ownerObj) else type(ownerObj)
      if ownerObj is None:
        qualname, _ = _class_fnNamesFromFnQualname(func.__qualname__)
      else:
        qualname = ownerCls.__qualname__
      self.boundFnsPerQualname[qualname].append(boundFnParam)
      # Make sure it's an object not just a class before forcing extended init
      if ownerObj is not None and ownerCls != ownerObj:
        self.addRegisteredFuncsFromCls(ownerCls, namePath)
        boundParamList = self.boundFnsPerQualname.get(ownerCls.__qualname__, [])
        for boundParam in boundParamList:
          self.hookupShortcutForBoundFn(boundParam, ownerObj)
      elif forceCreate:
        self.addRegisteredFuncsFromCls(ownerCls, namePath)
      # Global shortcuts won't get created since no 'global' widget init will be
      # called. In those cases, create right away
      if ownerCls == type(None):
        self.hookupShortcutForBoundFn(boundFnParam, None)
      return func
    return registerMethodDecorator

  def createRegisteredButton(self, btnParam: FRParam, ownerObj: Any, doRegister=True,
                             baseBtn: QtWidgets.QAbstractButton=None):
    """Check if this shortcut was already made globally or for this owner"""
    for prevRegisteredOwner in ownerObj, None:
      if (btnParam, prevRegisteredOwner) in self.paramToShortcutMapping:
        doRegister = False
        ownerObj = prevRegisteredOwner
        break

    if baseBtn is not None:
      newBtn = baseBtn
      tooltipText = btnParam.helpText
    elif 'icon' in btnParam.opts:
      newBtn = QtWidgets.QPushButton(QtGui.QIcon(btnParam.opts['icon']), '', self)
      tooltipText = helpTextToRichText(btnParam.helpText, btnParam.name)
    else:
      newBtn = QtWidgets.QPushButton(btnParam.name, self)
      tooltipText = btnParam.helpText
    if btnParam.value is None:
      # Either the shortcut wasn't given a value or wasn't requested, or already exists
      newBtn.setToolTip(tooltipText)
      return newBtn

    param = None
    if isclass(ownerObj) and doRegister:
      self.registerMethod_cls(btnParam, forceCreate=True)(
        lambda *args: newBtn.clicked.emit(), ownerObj)
    elif doRegister:
      param = self.registerMethod_obj(lambda *args: newBtn.clicked.emit(), btnParam, ownerObj)
    else:
      ownerObj = type(ownerObj)

    if param is None:
      param = self.params.child(pascalCaseToTitle(ownerObj.__name__), btnParam.name)
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
    shortcutParam: ShortcutParameter = Parameter.create(name=funcFrParam.name,
                                                        type='shortcut', value=funcFrParam.value,
                                                        tip=funcFrParam.helpText,
                                                        frParam=funcFrParam)
    clsName = pascalCaseToTitle(cls.__name__)
    pgParam = getParamChild(self.params, clsName)
    if funcFrParam.name not in pgParam.names:
      pgParam.addChild(shortcutParam)
    else:
      shortcutParam = pgParam.child(shortcutParam.name())

    seq = shortcutParam.seqEdit.keySequence()
    if ownerObj is None or not isinstance(ownerObj, QtWidgets.QWidget):
      newShortcut = QtWidgets.QShortcut(seq, QtWidgets.QApplication.desktop())
      ctx = QtCore.Qt.ApplicationShortcut
    else:
      newShortcut = QtWidgets.QShortcut(seq, ownerObj)
      ctx = QtCore.Qt.WidgetWithChildrenShortcut

    newShortcut.setContext(ctx)
    partialFn = partial(func, *funcArgs, **funcKwargs)
    newShortcut.activated.connect(partialFn)
    newShortcut.activatedAmbiguously.connect(lambda: self.ambigWarning(ownerObj, newShortcut))
    shortcutParam.sigValueChanged.connect(lambda param: newShortcut.setKey(param.seqEdit.keySequence()))
    if (funcFrParam, ownerObj) in self.paramToShortcutMapping:
      # Already found this value before, make sure to remove it
      self.paramToShortcutMapping[funcFrParam, ownerObj].deleteLater()
    self.paramToShortcutMapping[funcFrParam, ownerObj] = newShortcut
    return shortcutParam

  @staticmethod
  def ambigWarning(ownerObj: QtWidgets.QWidget, shc: QtWidgets.QShortcut):
    warn(f'{ownerObj.__class__} shortcut ambiguously activated: {shc.key().toString()}\n'
         f'Perhaps multiple shortcuts are assigned the same key sequence?',
         S3AWarning)


  def addRegisteredFuncsFromCls(self, grouping: Any, namePath: Sequence[str]=()):
    """
    For a given class, adds the registered parameters from that class to the respective
    editor. This is how the dropdown menus in the editors are populated with the
    user-specified variables.

    :param grouping: Current class
    :param namePath: Sequence of groups to traverse to find this parameter
    """
    # Make sure to add parameters from registered base classes, too, if they exist
    iterGroupings = []
    baseClasses = getAllBases(grouping)
    for baseCls in baseClasses:
      iterGroupings.append(baseCls.__qualname__)
    boundFns_includingBases = []

    # If this group already exists, append the children to the existing group
    # instead of adding a new child
    ownerParam = getParamChild(self.params, *namePath)
    for curGrouping in iterGroupings:
      groupParamList = self.boundFnsPerQualname.get(curGrouping, [])
      boundFns_includingBases.extend(groupParamList)
      for boundFn in groupParamList:
        self.addBoundFn(boundFn, ownerParam)

    # Now make sure when props are registered for this grouping, they include
    # base class shortcuts too
    self.boundFnsPerQualname[grouping.__qualname__] = boundFns_includingBases

  @classmethod
  def addBoundFn(cls, boundFn: BoundFnParams, parentParam: Parameter):
    if boundFn.param.name in parentParam.names:
      # Already registered
      return
    paramForTree = {'name' : boundFn.param.name,
                    'type' : 'shortcut',
                    'value': boundFn.param.value,
                    'tip'  : boundFn.param.helpText}
    parentParam.addChild(paramForTree)

  def hookupShortcutForBoundFn(self, boundFn: BoundFnParams, ownerObj: Any):
    if boundFn.param in self.objToShortcutParamMapping[ownerObj]:
      return
    self.objToShortcutParamMapping[ownerObj].append(boundFn.param)
    self.registerMethod_obj(boundFn.func, boundFn.param, ownerObj, *boundFn.defaultFnArgs)

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
    raise ParamEditorError('Cannot register property/attribute as a shortcut')