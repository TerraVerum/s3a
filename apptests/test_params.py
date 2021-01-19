import pytest
from s3a.structures import PrjParam, PrjParamGroup

pgroup = [PrjParam('test'), PrjParam('this')]
def test_fromString_valid():
  assert PrjParamGroup.fieldFromParam(pgroup, 'test') == pgroup[0]

def test_fromString_invalid():
  with pytest.raises(ValueError):
    PrjParamGroup.fieldFromParam(pgroup, 'noparam')
  with pytest.warns(UserWarning):
    newParam = PrjParamGroup.fieldFromParam(pgroup, 'noparam', default=pgroup[0])
  assert newParam == pgroup[0]