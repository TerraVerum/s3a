from typing import List, Any, Callable

from cdef.actionstack import FRActionStack
import numpy as np
import pytest

from cdef.structures import FRActionStackError

COUNT = 10

class StackForTesting(FRActionStack):
  def __init__(self):
    super().__init__()
    def op(num=None):
      if num is None:
        num = len(lst)
      lst.append(num)
      yield
      lst.pop()
    lst = []
    op = self.undoable()(op)

    self.op = op
    self.lst = lst

def test_group():
  stack = StackForTesting()
  with stack.group('multi op'):
    for ii in range(COUNT):
      stack.op()
      assert stack.lst[-1] == ii
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
  @stack.undoable('recursive op')
  def swapHalves(lst):
    sz = len(lst)
    if sz == 2:
      tmp = lst[0]
      lst[0] = lst[1]
      lst[1] = tmp
      yield lst
      return
    lst[:sz//2] = swapHalves(lst[:sz//2])
    lst[sz//2::] = swapHalves(lst[sz//2:])
    yield lst

    swapHalves(lst)

  origLst = stack.lst.copy()
  swapHalves(stack.lst)
  swapped = stack.lst.copy()
  stack.undo()
  assert stack.lst == origLst
  stack.redo()
  assert stack.lst == swapped

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
