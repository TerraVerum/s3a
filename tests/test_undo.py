from typing import List, Any

from cdef import undo
import numpy as np
import pytest

from cdef.structures import FRUndoStackError

COUNT = 10

def op(lst):
  lst.append(len(lst))
  yield
  del lst[-1]

def test_group():
  stack = undo.FRActionStack()
  mylst = []
  @stack.undoable('op')
  def op(lst):
    lst.append(len(lst))
    yield
    lst.pop()
  with stack.group('multi op'):
    for ii in range(COUNT):
      op(mylst)
      assert mylst[-1] == ii
  op(mylst)
  assert mylst == list(range(COUNT+1))
  stack.undo()
  assert mylst == list(range(COUNT))
  stack.undo()
  assert mylst == []

def test_nested_doable():
  stack = undo.FRActionStack()

  mylst = []
  @stack.undoable('outer op')
  def outer(lst):
    lst.append(str(len(lst)))
    inner(lst)
    yield
    lst.pop();lst.pop()

  @stack.undoable('inner  op')
  def inner(lst):
    lst.append(len(lst))
    yield
    lst.pop()

  outer(mylst)
  assert mylst == ['0', 1]
  stack.undo()
  assert mylst == []

def test_recursive():
  stack = undo.FRActionStack()

  nextPow2 = int(np.power(2, np.ceil(np.log2(COUNT))))
  mylst = list(range(nextPow2))
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

  origLst = mylst.copy()
  swapHalves(mylst)
  swapped = mylst.copy()
  stack.undo()
  assert mylst == origLst
  stack.redo()
  assert mylst == swapped

def test_bad_undo():
  stack = undo.FRActionStack()

  @stack.undoable('trivial')
  def op(lst):
    lst.append([lst]*2)
    yield
    del lst[-1]

  mylst = [1,2,3]
  for _ii in range(4):
    op(mylst)
  for _ii in range(4):
    stack.undo()
  with pytest.raises(FRUndoStackError):
    stack.undo()

def test_bad_redo():
  stack = undo.FRActionStack()
  with pytest.raises(FRUndoStackError):
    stack.redo()

def test_invalidate_redos():
  stack = undo.FRActionStack()

  @stack.undoable()
  def op(lst, el):
    lst.append(el)
    yield
    del lst[-1]

  mylst = []
  for ii in range(COUNT):
    op(mylst, ii)

  assert mylst == list(range(COUNT))
  numEntriesToRemomve = COUNT//3
  for ii in range(numEntriesToRemomve):
    stack.undo()

  numRemainingEntries = COUNT-numEntriesToRemomve
  assert np.sum([a.treatAsUndo for a in stack.actions]) == numRemainingEntries

  op(mylst, 1)
  with pytest.raises(FRUndoStackError):
    stack.redo()
  stack.undo()
  assert len(stack.actions) == numRemainingEntries+1
  cmplst = list(range(numRemainingEntries))
  assert mylst == cmplst
  stack.redo()
  assert mylst == cmplst + [1]

def test_ignore_acts():
  stack = undo.FRActionStack()
  curop = stack.undoable('test ignore')(op)
  mylst = []
  with stack.ignoreActions():
    for _ in range(COUNT):
      curop(mylst)
  assert len(stack.actions) == 0
  with pytest.raises(FRUndoStackError):
    stack.undo()
