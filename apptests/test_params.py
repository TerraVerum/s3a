import pytest

from s3a.structures import OptionsDict, OptionsDictGroup

pgroup = [OptionsDict("test"), OptionsDict("this")]


def test_fromString_valid():
    assert OptionsDictGroup.fieldFromParameter(pgroup, "test") == pgroup[0]


def test_fromString_invalid():
    with pytest.raises(ValueError):
        OptionsDictGroup.fieldFromParameter(pgroup, "noparam")
    with pytest.warns(UserWarning):
        newParam = OptionsDictGroup.fieldFromParameter(
            pgroup, "noparam", default=pgroup[0]
        )
    assert newParam == pgroup[0]
