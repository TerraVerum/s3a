from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Callable, Any, Dict, List, DefaultDict, Sequence, Union
from warnings import warn

import pandas as pd
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from s3a.constants import SHORTCUTS_DIR
from s3a.generalutils import helpTextToRichText, getParamChild, pascalCaseToTitle
from s3a.structures import FRParam, ParamEditorError, S3AWarning
from .genericeditor import ParamEditor


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

def _clsNameOrGroup(cls: type):
  if hasattr(cls, '__groupingName__'):
    return cls.__groupingName__
  return pascalCaseToTitle(cls.__name__)


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
    self.recordsDf = pd.DataFrame(columns=['param', 'owner', 'namePath', 'shortcut'])
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

  def registerShortcut(self, func: Callable, shortcutOpts: Union[FRParam, dict],
                   funcArgs: tuple=(), funcKwargs: dict=None,
                   namePath:Tuple[str, ...]=(),
                   overrideBasePath: Sequence[str]=None,
                   overrideOwnerObj: Any=None,
                   createBtn=True,
                   baseBtn: QtWidgets.QPushButton=None,
                   **kwargs):
    if funcKwargs is None:
      funcKwargs = {}
    if overrideBasePath is None:
      namePath = tuple(self._baseRegisterPath) + tuple(namePath)
    else:
      namePath = tuple(overrideBasePath) + tuple(namePath)
    if isinstance(shortcutOpts, dict): shortcutOpts = FRParam(**shortcutOpts)
    shcForCreate = shortcutOpts.toPgDict()
    shcForCreate['type'] = 'shortcut'
    param = getParamChild(self.params, *namePath, chOpts=shcForCreate)

    if createBtn:
      self.createRegisteredButton(shortcutOpts, baseBtn)
      return param

    if overrideOwnerObj is None:
      overrideOwnerObj = shortcutOpts.opts.get('ownerObj', None)

    seq = param.seqEdit.keySequence()
    key = (shortcutOpts, overrideOwnerObj)
    if key in self.paramToShortcutMapping:
      # Can't recycle old key sequence signals because C++ lifecycle is not in sync
      # Without `disconnect`, "wrapped c/c++ object deleted" errors appear
      # TODO: Find way to preserve this, in case multuple other operations
      #   were bound to this shortcut and lost
      newShortcut = self.paramToShortcutMapping[key]
      newShortcut.disconnect()
      ctx = newShortcut.context()
    elif overrideOwnerObj is None or not isinstance(overrideOwnerObj, QtWidgets.QWidget):
      newShortcut = QtWidgets.QShortcut(seq, self.parent())
      ctx = QtCore.Qt.ApplicationShortcut
    else:
      newShortcut = QtWidgets.QShortcut(seq, overrideOwnerObj)
      ctx = QtCore.Qt.WidgetWithChildrenShortcut
    newShortcut.setContext(ctx)
    self.paramToShortcutMapping[key] = newShortcut

    def onActivate():
      func(*funcArgs, **funcKwargs)
    newShortcut.activated.connect(onActivate)
    newShortcut.activatedAmbiguously.connect(lambda: self.ambigWarning(overrideOwnerObj, newShortcut))
    param.sigValueChanged.connect(lambda param: newShortcut.setKey(param.seqEdit.keySequence()))
    return param


  def createRegisteredButton(self, btnOpts: FRParam,
                             baseBtn: QtWidgets.QAbstractButton=None,
                             **kwargs):
    tooltipText = btnOpts.helpText
    if baseBtn is not None:
      newBtn = baseBtn
    elif 'icon' in btnOpts.opts:
      newBtn = QtWidgets.QPushButton(QtGui.QIcon(btnOpts.opts['icon']), '', self)
      tooltipText = helpTextToRichText(btnOpts.helpText, btnOpts.name)
    else:
      newBtn = QtWidgets.QPushButton(btnOpts.name, self)
    if btnOpts.value is None:
      # Either the shortcut wasn't given a value or wasn't requested, or already exists
      newBtn.setToolTip(tooltipText)
      return newBtn
    btnOpts.opts['tip'] = tooltipText
    param = self.registerShortcut(newBtn.click, btnOpts, createBtn=False, **kwargs)

    def shcChanged(_param, newSeq: str):
      newTooltipText = f'Shortcut: {newSeq}'
      tip = param.opts.get('tip', '')
      tip = helpTextToRichText(tip, newTooltipText)
      newBtn.setToolTip(tip)

    param.sigValueChanged.connect(shcChanged)
    shcChanged(None, btnOpts.value)
    return newBtn

  @staticmethod
  def ambigWarning(ownerObj: QtWidgets.QWidget, shc: QtWidgets.QShortcut):
    warn(f'{ownerObj.__class__} shortcut ambiguously activated: {shc.key().toString()}\n'
         f'Perhaps multiple shortcuts are assigned the same key sequence?',
         S3AWarning)

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

  def registerFunc(self, *args, **kwargs):
    """Functions should not be registered as shortcuts"""
    raise ParamEditorError('Cannot register function as a shortcut. See `registerShortcut`'
                           ' instead.')