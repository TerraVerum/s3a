"""
Inspired by 'undo' on pypi (https://bitbucket.org/aquavitae/undo/src/default/)
but there are many flaws and the project is not under active development. It is
also less pythonic than it could be, using functions where properties are more
appropriate.
"""
from __future__ import annotations

import contextlib
from collections import deque
from functools import wraps
from typing import Callable, Generator, Deque, Union, Type, Any, List

from typing_extensions import Protocol

from cdef.structures import FRActionStackError, FRCdefException


class _FRAction:
  """
  This represents an action which can be done and undone.
  """
  def __init__(self, generator: Callable[[...], Generator], args:tuple=None,
               kwargs:dict=None, descr: str=None, treatAsUndo=False):
    if args is None:
      args = []
    if kwargs is None:
      kwargs = {}
    if descr is None:
      descr = generator.__name__
    self._generator = generator
    self.args = args
    self.kwargs = kwargs
    self.descr = descr
    self._runner = None

    self.treatAsUndo = treatAsUndo
    if treatAsUndo:
      # Need to swap how methods are treated
      tmp = self.forward
      self.forward = self.backward
      self.backward = tmp


  def do(self):
    if self.treatAsUndo:
      ret = self.backward()
    else:
      ret = self.forward()
    self.treatAsUndo = not self.treatAsUndo
    return ret

  def forward(self):
    """Do or redo the action"""
    self._runner = self._generator(*self.args, **self.kwargs)
    # Forward use is expired, so treat as backward now
    return next(self._runner)

  def backward(self):
    """Undo the action"""
    ret = None
    try:
      ret = next(self._runner)
    except StopIteration:
      pass
    # Delete it so that its not accidentally called again
    del self._runner
    return ret

class EMPTY: pass
EmptyType = Type[EMPTY]
_FRONT = -1
_BACK = 0

class Appendable(Protocol):
  def append(self):
    raise NotImplementedError

class _FRBufferOverride:
  def __init__(self, stack: FRActionStack, newActQueue: deque=None):
    if newActQueue is None:
      newActQueue = deque()
    self.newActQueue = newActQueue
    self.stack = stack

    self.oldStackActions = None

  def __enter__(self):
    stack = self.stack
    # Deisgned for internal use, so OK to use protected member
    # noinspection PyProtectedMember
    self.oldStackActions = stack._curReceiver
    stack._curReceiver = self.newActQueue
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stack._curReceiver = self.oldStackActions

class FRActionStack:
  """ The main undo stack.

  The two key features are the :func:`redo` and :func:`undo` methods. If an
  exception occurs during doing or undoing a undoable, the undoable
  aborts and the stack is cleared to avoid any further data corruption.

  The stack provides two properties for tracking actions: *docallback*
  and *undocallback*. Each of these allow a callback function to be set
  which is called when an action is done or undone repectively. By default,
  they do nothing.
  """

  def __init__(self, maxlen:int=50):
    if maxlen is not None:
      maxlen *= 2
    self.actions: Deque[_FRAction] = deque(maxlen=maxlen)
    self._curReceiver = self.actions
    self._savepoint: Union[EmptyType, _FRAction] = EMPTY
    self.undoCallbacks: List[Callable] = []
    self.doCallbacks: List[Callable] = []

  @contextlib.contextmanager
  def group(self, descr: str=None, flushRedos=False):
    """ Return a context manager for grouping undoable actions.

    All actions which occur within the group will be undone by a single call
    of `stack.undo`."""
    groupStack = FRActionStack()
    newActBuffer = groupStack._curReceiver
    with _FRBufferOverride(self, newActBuffer):
      yield
    actIter = range(len(newActBuffer))
    def grpAct():
        for _ in actIter:
          groupStack.undo()
        yield
        for _ in actIter:
          groupStack.redo()
    self._curReceiver.append(_FRAction(grpAct, descr=descr, treatAsUndo=True))
    if flushRedos:
      self.flushUnusedRedos()

  def undoable(self, descr=None):
    """ Decorator which creates a new undoable action type.

    This decorator should be used on a generator of the following format::

      @undoable
      def operation(*args):
        do_operation_code
        yield returnval
        undo_operator_code
    """
    def decorator(generatorFn: Callable[[...], Generator]):
      nonlocal descr
      if descr is None:
        descr = generatorFn.__name__
      @wraps(generatorFn)
      def inner(*args, **kwargs):
        action = _FRAction(generatorFn, args, kwargs, descr)
        self._curReceiver.appendleft(action)
        ret = self.redo()

        self.flushUnusedRedos()
        return ret
      return inner
    return decorator

  @property
  def canUndo(self):
    """ Return *True* if undos are available """
    try:
      return self._curReceiver[-1].treatAsUndo
    except IndexError:
      return False

  @property
  def canRedo(self):
    """ Return *True* if redos are available """
    try:
      return not self._curReceiver[0].treatAsUndo
    except IndexError:
      return False

  def flushUnusedRedos(self):
    while (len(self._curReceiver) > 0
           and not self._curReceiver[0].treatAsUndo):
      self._curReceiver.popleft()

  def flushUntilSavepoint(self):
    while self._curReceiver[-1] is not self._savepoint:
      self._curReceiver.pop()

  def redo(self):
    """
    Redo the last undone action.

    This is only possible if no other actions have occurred since the
    last undo call.
    """
    ret = None
    if not self.canRedo:
      raise FRActionStackError('Nothing to redo')

    self._curReceiver.rotate(-1)
    try:
      ret = self.processAct()
    except StopIteration as ex:
      # Nothing to undo when this happens, remove the action from the stack.
      ret = ex.value
      self._curReceiver.popleft()
    except FRCdefException:
      # These are intentionally thrown, recoverable exceptions. Don't clear
      # the stack when they happen
      pass
    if self._curReceiver is self.actions:
      for callback in self.doCallbacks:
        callback()
    return ret

  def undo(self):
    """
    Undo the last action.
    """
    if not self.canUndo:
      raise FRActionStackError('Nothing to undo')

    ret = self.processAct()
    self._curReceiver.rotate(1)
    if self._curReceiver is self.actions:
      for callback in self.undoCallbacks:
        callback()
    return ret

  def processAct(self):
    """ Undo the last action. """
    act = self._curReceiver[-1]
    with self.ignoreActions():
      try:
        return act.do()
      except Exception as ex:
        # In general exceptions are recoverable, so don't obliterate the undo stack
        # self.clear()
        raise

  def clear(self):
    """ Clear the undo list. """
    self._savepoint = EMPTY
    self._curReceiver.clear()

  def setSavepoint(self):
    """ Set the savepoint. """
    if self.canUndo:
      self._savepoint = self._curReceiver[-1]
    else:
      self._savepoint = EMPTY

  @property
  def hasChanged(self):
    """ Return *True* if the state has changed since the savepoint. 
    
    This will always return *True* if the savepoint has not been set.
    """
    if self.canUndo:
      cmpAction = self._curReceiver[-1]
    else:
      cmpAction = EMPTY
    return self._savepoint is not cmpAction

  def ignoreActions(self):
    return _FRBufferOverride(self)