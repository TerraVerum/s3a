import pytest
from s3a.structures import FRParam, FRParamGroup, FRS3AException, FRS3AWarning

pgroup = [FRParam('test'), FRParam('this')]
def test_fromString_valid():
  assert FRParamGroup.fromString(pgroup, 'test') == pgroup[0]

def test_fromString_invalid():
  with pytest.raises(FRS3AException):
    FRParamGroup.fromString(pgroup, 'noparam')
  with pytest.warns(FRS3AWarning):
    newParam = FRParamGroup.fromString(pgroup, 'noparam', default=pgroup[0])
  assert newParam == pgroup[0]