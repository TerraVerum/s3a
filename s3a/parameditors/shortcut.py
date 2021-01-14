from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Callable, Any, Dict, List, DefaultDict, Sequence, Union
from warnings import warn

import pandas as pd
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from s3a.constants import SHORTCUTS_DIR
from s3a.generalutils import helpTextToRichText, getParamChild
from s3a.models.editorbase import params_flattened
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

  def _createSeq(self, shortcutOpts: Union[FRParam, dict],
                 namePath: Sequence[str]=(),
                 overrideBasePath: Sequence[str]=None,
                 **kwargs):
    if overrideBasePath is None:
      namePath = tuple(self._baseRegisterPath) + tuple(namePath)
    else:
      namePath = tuple(overrideBasePath) + tuple(namePath)
    if isinstance(shortcutOpts, dict): shortcutOpts = FRParam(**shortcutOpts)
    shcForCreate = shortcutOpts.toPgDict()
    shcForCreate['type'] = 'shortcut'
    return getParamChild(self.params, *namePath, chOpts=shcForCreate)

  def registerShortcut(self, func: Callable, shortcutOpts: Union[FRParam, dict],
                   funcArgs: tuple=(), funcKwargs: dict=None,
                   overrideOwnerObj: Any=None,
                   **kwargs):
    if funcKwargs is None:
      funcKwargs = {}
    if overrideOwnerObj is None:
      overrideOwnerObj = shortcutOpts.opts.get('ownerObj', None)
    param = self._createSeq(shortcutOpts, **kwargs)

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
      newShortcut = QtWidgets.QShortcut(seq, QtWidgets.QApplication.desktop())
      ctx = QtCore.Qt.ApplicationShortcut
    else:
      newShortcut = QtWidgets.QShortcut(seq, overrideOwnerObj)
      ctx = QtCore.Qt.WidgetWithChildrenShortcut
    newShortcut.setContext(ctx)
    self.paramToShortcutMapping[key] = newShortcut

    # Disconnect before connect in case this was already hooked up previously
    param.sigValueChanged.disconnect(self.maybeAmbigWarning)
    param.sigValueChanged.connect(self.maybeAmbigWarning)

    def onActivate():
      func(*funcArgs, **funcKwargs)
    newShortcut.activated.connect(onActivate)
    param.sigValueChanged.connect(lambda param: newShortcut.setKey(param.seqEdit.keySequence()))
    return param

  def registerMenuAction(self, btnOpts: FRParam, action: QtWidgets.QAction, **kwargs):
    param = self._createSeq(btnOpts, **kwargs)
    action.setToolTip(btnOpts.helpText)
    def shcChanged(_param, newSeq: str):
      action.setShortcut(newSeq)
      self.maybeAmbigWarning(_param, newSeq)

    param.sigValueChanged.connect(shcChanged)
    shcChanged(None, btnOpts.value)
    return action

  def createRegisteredButton(self, btnOpts: FRParam,
                             baseBtn: QtWidgets.QAbstractButton=None,
                             asToolBtn=False, **kwargs):
    if asToolBtn:
      btnType = QtWidgets.QToolButton
    else:
      btnType = QtWidgets.QPushButton
    tooltipText = btnOpts.helpText
    if baseBtn is not None:
      newBtn = baseBtn
    elif 'icon' in btnOpts.opts:
      newBtn = btnType(self)
      newBtn.setIcon(QtGui.QIcon(btnOpts.opts['icon']))
      tooltipText = helpTextToRichText(btnOpts.helpText, btnOpts.name)
    else:
      newBtn = btnType(self)
      newBtn.setText(btnOpts.name)
    if btnOpts.value is None:
      # Either the shortcut wasn't given a value or wasn't requested, or already exists
      newBtn.setToolTip(tooltipText)
      return newBtn
    btnOpts.opts['tip'] = tooltipText

    param = self._createSeq(btnOpts, **kwargs)

    def shcChanged(_param, newSeq: str):
      newTooltipText = f'Shortcut: {newSeq}'
      tip = param.opts.get('tip', '')
      tip = helpTextToRichText(tip, newTooltipText)
      newBtn.setToolTip(tip)
      newBtn.setShortcut(newSeq)
      self.maybeAmbigWarning(param, newSeq)

    param.sigValueChanged.connect(shcChanged)
    shcChanged(None, btnOpts.value)
    return newBtn

  def maybeAmbigWarning(self, _param, shortcut: str):
    conflicts = [p.name() for p in params_flattened(self.params) if p.value() == shortcut]
    if len(conflicts) > 1:
      warn(f'Ambiguous shortcut: {shortcut}\n'
           f'Perhaps multiple shortcuts are assigned the same key sequence? Possible conflicts:\n'
           f'{conflicts}',
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