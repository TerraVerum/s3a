from typing import List, Any, Callable

from cdef.actionstack import FRActionStack
import numpy as np
import pytest

from cdef.structures import FRActionStackError

COUNT = 10

class StackForTesting(FRActionStack):
  def __init__(self):
    super().__init__()

    @self.undoable()
    def op(num=None):
      if num is None:
        num = len(self.lst)
      self.lst.append(num)
      yield
      self.lst.pop()

    @self.undoable('recurse')
    def recursiveOp_swapEveryTwo(lst=None):
      if lst is None:
        lst = self.lst
      sz = len(lst)
      if sz < 2: pass
      elif sz < 4:
        tmp = lst[0]
        lst[0] = lst[1]
        lst[1] = tmp
      else:
        for rng in slice(sz//2), slice(sz//2, sz):
          lst[rng] = recursiveOp_swapEveryTwo(lst[rng])
      yield lst
      recursiveOp_swapEveryTwo(lst)

    def grpOp():
      with self.group('grouping'):
        for _ in range(COUNT):
          self.op()

    self.recursive = recursiveOp_swapEveryTwo
    self.grpOp = grpOp
    self.op = op

    self.lst = []

def test_group():
  stack = StackForTesting()
  stack.grpOp()
  stack.op()
  assert stack.lst == list(range(COUNT+1))
  stack.undo()
  assert stack.lst == list(range(COUNT))
  stack.undo()
  assert stack.lst == []

def test_nested_doable():
  stack = StackForTesting()

  @stack.undoable('outer op')
  def outer():
    stack.lst.append(str(len(stack.lst)))
    stack.op()
    yield
    stack.lst.pop();stack.lst.pop()

  outer()
  assert stack.lst == ['0', 1]
  stack.undo()
  assert stack.lst == []

def test_recursive():
  stack = StackForTesting()

  nextPow2 = int(np.power(2, np.ceil(np.log2(COUNT))))
  stack.lst = list(range(nextPow2))
  origLst = stack.lst.copy()

  stack.recursive()
  swapped = stack.lst.copy()

  stack.undo()
  assert stack.lst == origLst

  stack.redo()
  assert stack.lst == swapped
  assert len(stack.actions) == 1

def test_bad_undo():
  stack = StackForTesting()

  for _ii in range(4):
    stack.op()
  for _ii in range(4):
    stack.undo()
  with pytest.raises(FRActionStackError):
    stack.undo()

def test_bad_redo():
  stack = StackForTesting()
  with pytest.raises(FRActionStackError):
    stack.redo()
  stack.op()
  with pytest.raises(FRActionStackError):
    stack.undo()
    stack.redo()
    stack.redo()

def test_invalidate_redos():
  stack = StackForTesting()

  for ii in range(COUNT):
    stack.op()

  assert stack.lst == list(range(COUNT))
  numEntriesToRemomve = COUNT//3
  for ii in range(numEntriesToRemomve):
    stack.undo()

  numRemainingEntries = COUNT-numEntriesToRemomve
  assert np.sum([a.treatAsUndo for a in stack.actions]) == numRemainingEntries

  stack.op(1)
  # New action should flush old ones
  with pytest.raises(FRActionStackError):
    stack.redo()
  stack.undo()
  assert len(stack.actions) == numRemainingEntries+1
  cmplst = list(range(numRemainingEntries))
  assert stack.lst == cmplst
  stack.redo()
  assert stack.lst == cmplst + [1]

def test_ignore_acts():
  stack = StackForTesting()
  with stack.ignoreActions():
    for _ in range(COUNT):
      stack.op()
  assert len(stack.actions) == 0
  with pytest.raises(FRActionStackError):
    stack.undo()

def test_max_len():
  stack = StackForTesting()
  cnt = stack.actions.maxlen + 20
  for ii in range(cnt):
    stack.op()

  with pytest.raises(FRActionStackError):
    for ii in range(cnt):
      stack.undo()

def test_grp_composite():
  stack = StackForTesting()
  with stack.group('outer grp', flushUnusedRedos=True):
    with stack.group('inner grp'):
      stack.grpOp()
      stack.recursive()
    stack.op()
    stack.grpOp()
    stack.op()
  assert len(stack.actions) == 1

  postOpLst = stack.lst.copy()
  stack.undo()
  assert len(stack.lst) == 0
  stack.redo()
  assert postOpLst == stack.lst

def test_savepoint():
  stack = StackForTesting()

  for _ in range(COUNT):
    stack.op()
  stack.setSavepoint()

  stack.op()
  assert stack.changedSinceLastSave

  stack.revertToSavepoint()
  assert not stack.changedSinceLastSave

  stack.op()
  stack.op()

  stack.revertToSavepoint()
  assert stack.lst == list(range(COUNT))