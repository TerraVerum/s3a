from __future__ import annotations

from typing import Tuple, Collection, Callable, Union, Iterable, Any, Dict

from bidict import bidict
from pyqtgraph.Qt import QtWidgets, QtGui

from s3a.projectvars import FR_CONSTS
from s3a.structures import FRParam

__all__ = ['FRDrawOpts']

# class FRDrawOpts_old(QtWidgets.QWidget):
#   def __init__(self, parent=None, shapes: Tuple[FRParam, ...]=None, actions: Tuple[FRParam, ...]=None):
#     """
#     Creates a draw options widget hosting both shape and action selection buttons.
#     :param parent: UI widget whose destruction will also destroy these widgets
#     :param shapes: Shape options that will appear in the widget
#     :param actions: Action optiosn that will appear in the widget
#     """
#     super().__init__(parent)
#     # Create 2 layout versions so on resize the group boxes can 'wrap'
#     self.topLayout = QtWidgets.QHBoxLayout()
#     self.setLayout(self.topLayout)
#
#     # SHAPES
#     shapeUiGroup, self.shapeBtnGroup, self.shapeBtnParamMap = self._create_addBtnToGroup(shapes, "Shapes")
#     self.topLayout.addWidget(shapeUiGroup)
#     # ACTIONS
#     actionUiGroup, self.actionBtnGroup, self.actionBtnParamMap = self._create_addBtnToGroup(actions, "Actions")
#     self.topLayout.addWidget(actionUiGroup)
#     self.topLayout.setDirection(self.topLayout.LeftToRight)
#     self.horizWidth = self.layout().minimumSize().width()
#
#   def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
#     if self.width() < self.horizWidth + 30:
#       self.topLayout.setDirection(self.topLayout.TopToBottom)
#     else:
#       self.topLayout.setDirection(self.topLayout.LeftToRight)
#     super().resizeEvent(ev)
#
#   def _create_addBtnToGroup(self, whichBtns: FR_CONSTS, groupTitle: str) \
#       -> Tuple[QtWidgets.QGroupBox, QtWidgets.QButtonGroup, bidict]:
#     uiGroup = QtWidgets.QGroupBox(self)
#     uiGroup.setTitle(groupTitle)
#     uiLayout = QtWidgets.QHBoxLayout(uiGroup)
#     btnGroup = QtWidgets.QButtonGroup(uiGroup)
#     # btnDict: Dict[QtWidgets.QPushButton, FRParam] = {}
#     btnDict = bidict()
#     for btnParam in whichBtns: # type: FRParam
#       if btnParam.value is not None:
#         newBtn = QtWidgets.QPushButton(QtGui.QIcon(btnParam.value), '', uiGroup)
#         newBtn.setToolTip(btnParam.name)
#       else:
#         newBtn = QtWidgets.QPushButton(btnParam.name, uiGroup)
#       newBtn.setCheckable(True)
#       btnGroup.addButton(newBtn)
#       uiLayout.addWidget(newBtn)
#       btnDict[newBtn] = btnParam
#     return uiGroup, btnGroup, btnDict
#
#   def selectOpt(self, shapeOrAction: FRParam):
#     """
#     Programmatically selects a shape or action from the existing button group.
#     Whether a shape or action is passed in is inferred from which button group
#     :param:`shapeOrAction` belongs to
#
#     :param shapeOrAction: The button to select
#     :return: None
#     """
#     # TODO: This should probably be more robust
#     if shapeOrAction in self.shapeBtnParamMap.inverse:
#       self.shapeBtnParamMap.inverse[shapeOrAction].setChecked(True)
#     elif shapeOrAction in self.actionBtnParamMap.inverse:
#       self.actionBtnParamMap.inverse[shapeOrAction].setChecked(True)

class FRDrawOpts(QtWidgets.QWidget):
  def __init__(self, shapeGrp: FRButtonCollection, actGrp: FRButtonCollection,
               parent: QtWidgets.QWidget=None):
    """
    Creates a draw options widget hosting both shape and action selection buttons.
    :param parent: UI widget whose destruction will also destroy these widgets
    :param shapeGrp: Shape options that will appear in the widget
    :param actGrp: Action optiosn that will appear in the widget
    """
    super().__init__(parent)
    # Create 2 layout versions so on resize the group boxes can 'wrap'
    self.topLayout = QtWidgets.QHBoxLayout()
    self.setLayout(self.topLayout)

    # SHAPES

    self.topLayout.addWidget(shapeGrp)
    # ACTIONS
    self.topLayout.addWidget(actGrp)
    self.topLayout.setDirection(self.topLayout.LeftToRight)
    self.horizWidth = self.layout().minimumSize().width()

  def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
    if self.width() < self.horizWidth + 30:
      self.topLayout.setDirection(self.topLayout.TopToBottom)
    else:
      self.topLayout.setDirection(self.topLayout.LeftToRight)
    super().resizeEvent(ev)

  def _create_addBtnToGroup(self, whichBtns: FR_CONSTS, groupTitle: str) \
      -> Tuple[QtWidgets.QGroupBox, QtWidgets.QButtonGroup, bidict]:
    uiGroup = QtWidgets.QGroupBox(self)
    uiGroup.setTitle(groupTitle)
    uiLayout = QtWidgets.QHBoxLayout(uiGroup)
    btnGroup = QtWidgets.QButtonGroup(uiGroup)
    # btnDict: Dict[QtWidgets.QPushButton, FRParam] = {}
    btnDict = bidict()
    for btnParam in whichBtns: # type: FRParam
      if btnParam.value is not None:
        newBtn = QtWidgets.QPushButton(QtGui.QIcon(btnParam.value), '', uiGroup)
        tooltipText = btnParam.name
        if len(btnParam.helpText) > 0:
          tooltipText += f'\n{btnParam.helpText}'
        newBtn.setToolTip(tooltipText)
      else:
        newBtn = QtWidgets.QPushButton(btnParam.name, uiGroup)
      newBtn.setCheckable(True)
      btnGroup.addButton(newBtn)
      uiLayout.addWidget(newBtn)
      btnDict[newBtn] = btnParam
    return uiGroup, btnGroup, btnDict

  def selectOpt(self, shapeOrAction: FRParam):
    """
    Programmatically selects a shape or action from the existing button group.
    Whether a shape or action is passed in is inferred from which button group
    :param:`shapeOrAction` belongs to

    :param shapeOrAction: The button to select
    :return: None
    """
    # TODO: This should probably be more robust
    if shapeOrAction in self.shapeBtnParamMap.inverse:
      self.shapeBtnParamMap.inverse[shapeOrAction].setChecked(True)
    elif shapeOrAction in self.actionBtnParamMap.inverse:
      self.actionBtnParamMap.inverse[shapeOrAction].setChecked(True)

btnCallable = Callable[[FRParam], Any]
class FRButtonCollection(QtWidgets.QGroupBox):
  def __init__(self, parent=None, title: str=None, btnParams: Collection[FRParam]=(),
               btnTriggerFns: Union[btnCallable, Collection[btnCallable]]=(),
               exclusive=True, checkable=True):
    super().__init__(parent)
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

  def create_addBtn(self, btnParam: FRParam, triggerFn: btnCallable, checkable=True):
    if btnParam.value is not None:
      newBtn = QtWidgets.QPushButton(QtGui.QIcon(btnParam.value), '', self)
      tooltipText = btnParam.name
      if len(btnParam.helpText) > 0:
        tooltipText += f'\n{btnParam.helpText}'
      newBtn.setToolTip(tooltipText)
    else:
      newBtn = QtWidgets.QPushButton(btnParam.name, self)
    if checkable:
      newBtn.setCheckable(True)
      oldTriggerFn = triggerFn
      # If the button is chekcable, only call this function when the button is checked
      def newTriggerFn(param: FRParam):
        if newBtn.isChecked():
          oldTriggerFn(param)
      triggerFn = newTriggerFn
    newBtn.clicked.connect(lambda: triggerFn(btnParam))
    self.btnGroup.addButton(newBtn)
    self.uiLayout.addWidget(newBtn)
    self.paramToFuncMapping[btnParam] = triggerFn
    self.paramToBtnMapping[btnParam] = newBtn

  def callFuncByParam(self, param: FRParam):
    # Ensure function is called in the event it requires a button to be checked
    btn = self.paramToBtnMapping[param]
    if btn.isCheckable():
      btn.setChecked(True)
    self.paramToFuncMapping[param](param)

def _create_addActionsToToolbar(toolbar: QtWidgets.QToolBar, whichBtns: FR_CONSTS) -> QtWidgets.QActionGroup:
  uiGroup = QtWidgets.QGroupBox('Test', toolbar)
  actionGroup = QtWidgets.QActionGroup(uiGroup)
  actionGroup.setExclusive(True)
  for btnParam in whichBtns: # type: FRParam
    # TODO: Use picture instead of name
    newAction = actionGroup.addAction(btnParam.name)
    newAction.setCheckable(True)
  toolbar.addWidget(uiGroup)
  return actionGroup