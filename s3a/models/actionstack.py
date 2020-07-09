"""
Inspired by 'undo' on pypi (https://bitbucket.org/aquavitae/undo/src/default/)
but there are many flaws and the project is not under active development. It is
also less pythonic than it could be, using functions where properties are more
appropriate.
"""
from __future__ import annotations

import contextlib
import copy
from collections import deque
from functools import wraps
from typing import Callable, Generator, Deque, Union, Type, Any, List

from typing_extensions import Protocol

from s3a.structures import FRActionStackError

__all__ = ['FRActionStack', 'FRAction']

# _generatorCallable = Callable[[...], Union[Generator, Any]]
class FRAction:
  """
  This represents an action which can be done and undone.
  """
  def __init__(self, generator: Callable[[...], Union[Generator, Any]], args:tuple=None,
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

  def reassignBackward(self, backwardFn: Callable[[...], Any], backwardArgs=(),
                       backwardKwargs=None):

    if backwardKwargs is None:
      backwardKwargs = {}
    oldGenerator = self._generator
    def newGenerator(*args, **kwargs):
      # Keep forward
      yield next(oldGenerator(*args, **kwargs))
      # Alter backwards
      yield backwardFn(*backwardArgs, **backwardKwargs)
    self._generator = newGenerator
    if self.treatAsUndo:
      # Already in current runner, so change it
      def newRunner():
        yield backwardFn(*backwardArgs, **backwardKwargs)
      self._runner = newRunner()

  def forward(self, graceful=False):
    """
    Do or redo the action

    :param graceful: Whether to show an error on stop iteration or not. If a function
      is registered as undoable but doesn't contain a yield expression this is useful,
      i.e. performing a redo when that redo may not have a corresponding undo again
    """
    self._runner = self._generator(*self.args, **self.kwargs)
    # Forward use is expired, so treat as backward now
    self.treatAsUndo = True
    if not graceful:
      return next(self._runner)
    else:
      return gracefulNext(self._runner)

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
    self.actions: Deque[FRAction] = deque(maxlen=maxlen)
    self._curReceiver = self.actions
    self._savepoint: Union[EmptyType, FRAction] = EMPTY
    self.stackChangedCallbacks: List[Callable] = []

  @contextlib.contextmanager
  def group(self, descr: str=None, flushUnusedRedos=False):
    """ Return a context manager for grouping undoable actions.

    All actions which occur within the group will be undone by a single call
    of `stack.undo`.
    """
    newActBuffer: Deque[FRAction] = deque()
    with _FRBufferOverride(self, newActBuffer):
      yield
    def grpAct():
      for _ in range(2):
        for act in newActBuffer:
          if act.treatAsUndo:
            act.backward()
          else:
            act.forward(graceful=True)
        yield
    if self._curReceiver is not None:
      self._curReceiver.append(FRAction(grpAct, descr=descr, treatAsUndo=True))
    if flushUnusedRedos:
      self.flushUnusedRedos()

  def undoable(self, descr=None, asGroup=False, copyArgs=False):
    """ Decorator which creates a new undoable action type.

    Parameters
    ___________
    :param descr: Description of this action, e.g. "add components", etc.
    :param asGroup: If *True* assumes this undoable function is a composition
      of other undoable functions. This is a simple alias for
      >>> with stack.group('descr', flushUnusedRedos=True):
      >>>  func(*args, **kwargs)
    :param copyArgs: Whether to make a copy of the arguments used for the undo
      function. This is useful for functions where the input argument is modified
      during the function call. WARNING: UNTESTED
    """
    def decorator(generatorFn: Callable[[...], Generator]):
      nonlocal descr
      if descr is None:
        descr = generatorFn.__name__
      @wraps(generatorFn)
      def inner_group(*args, **kwargs):
        with self.group(descr, flushUnusedRedos=True):
          ret = generatorFn(*args, **kwargs)
        self._processCallbacks()
        return ret

      @wraps(generatorFn)
      def inner_action(*args, **kwargs):
        shouldAppend = True
        if copyArgs:
          args = tuple(copy.copy(arg) for arg in args)
          kwargs = {k: copy.copy(v) for k, v in kwargs.items()}
        action = FRAction(generatorFn, args, kwargs, descr)
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
        self._processCallbacks()
        return ret
      if asGroup:
        return inner_group
      else:
        return inner_action
    return decorator

  def _processCallbacks(self):
    if self._curReceiver is self.actions:
      for callback in self.stackChangedCallbacks:
        callback()

  @property
  def undoDescr(self):
    if self.canUndo:
      return self.actions[-1].descr
    else:
      return None

  @property
  def redoDescr(self):
    if self.canRedo:
      return self.actions[0].descr
    else:
      return None

  @property
  def canUndo(self):
    """ Return *True* if undos are available """
    return len(self.actions) > 0 and self.actions[-1].treatAsUndo

  @property
  def canRedo(self):
    """ Return *True* if redos are available """
    return len(self.actions) > 0 and not self.actions[0].treatAsUndo

  def resizeStack(self, newMaxLen: int):
    if newMaxLen == self.actions.maxlen:
      return
    newDeque: Deque[FRAction] = deque(maxlen=newMaxLen)
    newDeque.extend(self.actions)
    receiverNeedsReset = True if self._curReceiver is self.actions else False
    self.actions = newDeque
    if receiverNeedsReset:
      self._curReceiver = self.actions

  def flushUnusedRedos(self):
    while self.canRedo:
      if self.actions[0] is self._savepoint:
        self._savepoint = EMPTY
      self.actions.popleft()
    self._processCallbacks()

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
    self._processCallbacks()

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
      ret = self.actions[-1].forward(graceful=True)
    self._processCallbacks()
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
    self._processCallbacks()
    return ret

  def clear(self):
    """ Clear the undo list. """
    self._savepoint = EMPTY
    self.actions.clear()
    self._processCallbacks()

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