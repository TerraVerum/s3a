from __future__ import annotations
from typing import Tuple, Collection, Callable, Union, Iterable, Any, Dict, Sequence, \
  List, Optional

from pyqtgraph.Qt import QtWidgets, QtGui, QtCore

from s3a.structures import FRParam
from s3a import parameditors

__all__ = ['ButtonCollection']

class _DEFAULT_OWNER: pass
"""None is a valid owner, so create a sentinel that's not valid"""
btnCallable = Callable[[FRParam], Any]
class ButtonCollection(QtWidgets.QGroupBox):
  def __init__(self, parent=None, title: str=None, btnParams: Collection[FRParam]=(),
               btnTriggerFns: Union[btnCallable, Collection[btnCallable]]=(),
               exclusive=True, checkable=True):
    super().__init__(parent)
    self.lastTriggered: Optional[FRParam] = None
    self.uiLayout = QtWidgets.QHBoxLayout(self)
    self.btnGroup = QtWidgets.QButtonGroup(self)
    self.paramToFuncMapping: Dict[FRParam, btnCallable] = dict()
    self.paramToBtnMapping: Dict[FRParam, QtWidgets.QPushButton] = dict()
    if title is not None:
      self.setTitle(title)
    self.btnGroup.setExclusive(exclusive)

    if not isinstance(btnTriggerFns, Iterable):
      btnTriggerFns = [btnTriggerFns]*len(btnParams)
    for param, fn in zip(btnParams, btnTriggerFns):
      self.create_addBtn(param, fn, checkable)

  def create_addBtn(self, btnParam: FRParam, triggerFn: btnCallable, checkable=True, **registerOpts):
    if btnParam in self.paramToBtnMapping or not btnParam.opts.get('guibtn', True):
      # Either already exists or wasn't designed to be a button
      return
    newBtn = parameditors.FR_SINGLETON.shortcuts.createRegisteredButton(btnParam, **registerOpts)
    if checkable:
      newBtn.setCheckable(True)
      oldTriggerFn = triggerFn
      # If the button is chekcable, only call this function when the button is checked
      def newTriggerFn(param: FRParam):
        if newBtn.isChecked():
          oldTriggerFn(param)
      triggerFn = newTriggerFn
    newBtn.clicked.connect(lambda: self.callFuncByParam(btnParam))

    self.addBtn(btnParam, newBtn, triggerFn)

  def clear(self):
    for button in self.paramToBtnMapping.values():
      self.btnGroup.removeButton(button)
      self.uiLayout.removeWidget(button)
      button.deleteLater()

    self.paramToBtnMapping.clear()
    self.paramToFuncMapping.clear()

  def addFromExisting(self, other: ButtonCollection, which: Collection[FRParam]=None):
    for (param, btn), func in zip(other.paramToBtnMapping.items(), other.paramToFuncMapping.values()):
      if which is None or param in which:
        self.addBtn(param, btn, func)

  def addBtn(self, param: FRParam, btn: QtWidgets.QPushButton, func: btnCallable):
    self.btnGroup.addButton(btn)
    self.uiLayout.addWidget(btn)
    self.paramToFuncMapping[param] = func
    self.paramToBtnMapping[param] = btn


  def callFuncByParam(self, param: FRParam):
    if param is None:
      return
    # Ensure function is called in the event it requires a button to be checked
    btn = self.paramToBtnMapping[param]
    if btn.isCheckable():
      btn.setChecked(True)
    self.paramToFuncMapping[param](param)
    self.lastTriggered = param

  @classmethod
  def fromToolsEditors(cls,
                       toolsEditors: Union[parameditors.ParamEditor,
                       Sequence[parameditors.ParamEditor]],
                       title='Tools',
                       checkable=True, ownerClctn: ButtonCollection=None):
    toolParams = []
    toolFns = []
    sepIdxs = []
    curSepIdx = 0
    if not isinstance(toolsEditors, Sequence):
      toolsEditors = [toolsEditors]
    for toolsEditor in toolsEditors:
      for param in toolsEditor.params.childs:
        if 'action' in param.opts['type'] and param.opts.get('guibtn', True):
          toolParams.append(FRParam(**param.opts))
          toolFns.append(lambda *_args, _param=param: _param.sigActivated.emit(_param))
          curSepIdx += 1
      sepIdxs.append(curSepIdx)
    if ownerClctn is None:
      ownerClctn = ButtonCollection(title=title, exclusive=True)
      # Don't create shortcuts since this will be done by the tool editor
      returnClctn = True
    else:
      returnClctn = False

    # TODO: Figure out separators inside a box layout
    # numFns = len(toolFns)
    for ii, (param, fn) in enumerate(zip(toolParams, toolFns)):
      ownerClctn.create_addBtn(param, fn, checkable)
    #   if ii in sepIdxs and (0 < ii < numFns-1):
    #     # Add qframe as separator since separator doesn't exist for layouts
    #     sep = QtWidgets.QFrame(ownerClctn)
    #     sep.setFrameShape(sep.HLine)
    #     sep.setFrameShadow(sep.Sunken)
    #     sep.setFixedHeight(ownerClctn.height())
    #     ownerClctn.uiLayout.addWidget(sep)
    if returnClctn:
      return ownerClctn
    else:
      return toolParams, toolFns

  def toolbarFormat(self):
    """
    Returns a list of buttons + title in a format that's easier to add to a toolbar, e.g.
    doesn't require as much horizontal space
    """
    title = self.title()
    out: List[QtWidgets.QWidget] = [] if title is None else [QtWidgets.QLabel(self.title())]
    for btn in self.paramToBtnMapping.values():
      out.append(btn)
    return out