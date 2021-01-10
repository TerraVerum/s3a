import pytest
from s3a.structures import FRParam, FRParamGroup, S3AException, S3AWarning

pgroup = [FRParam('test'), FRParam('this')]
def test_fromString_valid():
  assert FRParamGroup.fieldFromParam(pgroup, 'test') == pgroup[0]

def test_fromString_invalid():
  with pytest.raises(S3AException):
    FRParamGroup.fieldFromParam(pgroup, 'noparam')
  with pytest.warns(S3AWarning):
    newParam = FRParamGroup.fieldFromParam(pgroup, 'noparam', default=pgroup[0])
  assert newParam == pgroup[0]