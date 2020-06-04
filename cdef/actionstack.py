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

from cdef.frgraphics.graphicsutils import raiseErrorLater
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
      # Need to init runner for when backward is called
      self._runner = self._generator(*args, **kwargs)

  def forward(self):
    """Do or redo the action"""
    self._runner = self._generator(*self.args, **self.kwargs)
    # Forward use is expired, so treat as backward now
    self.treatAsUndo = True
    return next(self._runner)

  def backward(self):
    """Undo the action"""
    # It's OK if this raises StopIteration, since we don't need anything after
    # calling it. Therefore call graceful next.
    ret = gracefulNext(self._runner)
    # Delete it so that its not accidentally called again
    del self._runner
    self.treatAsUndo = False
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

def gracefulNext(generator: Generator):
  try:
    return next(generator)
  except StopIteration as ex:
    return ex.value

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
    self.actions: Deque[_FRAction] = deque(maxlen=maxlen)
    self._curReceiver = self.actions
    self._savepoint: Union[EmptyType, _FRAction] = EMPTY
    self.undoCallbacks: List[Callable] = []
    self.doCallbacks: List[Callable] = []

  @contextlib.contextmanager
  def group(self, descr: str=None, flushUnusedRedos=False):
    """ Return a context manager for grouping undoable actions.

    All actions which occur within the group will be undone by a single call
    of `stack.undo`."""
    newActBuffer: deque[_FRAction] = deque()
    with _FRBufferOverride(self, newActBuffer):
      yield
    def grpAct():
      for _ in range(2):
        for act in newActBuffer:
          if act.treatAsUndo:
            act.backward()
          else:
            act.forward()
        yield
    if self._curReceiver is not None:
      self._curReceiver.append(_FRAction(grpAct, descr=descr, treatAsUndo=True))
    if flushUnusedRedos:
      self.flushUnusedRedos()

  def undoable(self, descr=None):
    """ Decorator which creates a new undoable action type.

    This decorator should be used on a generator of the following format::
      @undoable(descr)
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
        shouldAppend = True
        action = _FRAction(generatorFn, args, kwargs, descr)
        try:
          with self.ignoreActions():
            ret = action.forward()
        except StopIteration as ex:
          ret = ex.value
          shouldAppend = False
        if self._curReceiver is not None and shouldAppend:
          self._curReceiver.append(action)
        if self._curReceiver is self.actions:
          # State change of application means old redos are invalid
          self.flushUnusedRedos()
        # Else: doesn't get added to the queue
        return ret
      return inner
    return decorator

  @property
  def canUndo(self):
    """ Return *True* if undos are available """
    return len(self.actions) > 0 and self.actions[-1].treatAsUndo

  @property
  def canRedo(self):
    """ Return *True* if redos are available """
    return len(self.actions) > 0 and not self.actions[0].treatAsUndo

  def flushUnusedRedos(self):
    while self.canRedo:
      if self.actions[0] is self._savepoint:
        self._savepoint = EMPTY
      self.actions.popleft()

  def revertToSavepoint(self):
    if self._savepoint is EMPTY:
      raise FRActionStackError('Attempted to revert to empty savepoint. Perhaps you'
                               ' performed several \'undo\' operations, then performed'
                               ' a forward operation that flushed your savepoint?')
    if self._savepoint.treatAsUndo:
      actFn = self.undo
    else:
      actFn = self.redo
    while self.changedSinceLastSave:
      actFn()

  def redo(self):
    """
    Redo the last undone action.

    This is only possible if no other actions have occurred since the
    last undo call.
    """
    if not self.canRedo:
      raise FRActionStackError('Nothing to redo')

    self.actions.rotate(-1)
    with self.ignoreActions():
      ret = self.actions[-1].forward()
    if self.actions is self.actions:
      for callback in self.doCallbacks:
        callback()
    return ret

  def undo(self):
    """
    Undo the last action.
    """
    if not self.canUndo:
      raise FRActionStackError('Nothing to undo')

    with self.ignoreActions():
      ret = self.actions[-1].backward()
    self.actions.rotate(1)
    if self.actions is self.actions:
      for callback in self.undoCallbacks:
        callback()
    return ret

  def clear(self):
    """ Clear the undo list. """
    self._savepoint = EMPTY
    self.actions.clear()

  def setSavepoint(self):
    """ Set the savepoint. """
    if self.canUndo:
      self._savepoint = self.actions[-1]
    else:
      self._savepoint = EMPTY

  @property
  def changedSinceLastSave(self):
    """ Return *True* if the state has changed since the savepoint. 
    
    This will always return *True* if the savepoint has not been set.
    """
    if self._savepoint is EMPTY: return False
    elif self._savepoint.treatAsUndo:
      cmpAction = self.actions[-1]
    else:
      cmpAction = self.actions[0]
    return self._savepoint is not cmpAction

  def ignoreActions(self):
    return _FRBufferOverride(self)