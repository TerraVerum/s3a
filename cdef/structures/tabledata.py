from __future__ import annotations
import copy
from datetime import datetime
from typing import Union, Collection, List
from ruamel.yaml import YAML

from os import PathLike
from pathlib import Path

from pandas import DataFrame as df

from . import FRParam
from ..frgraphics import parameditors
from ..projectvars import DATE_FORMAT, REQD_TBL_FIELDS, COMP_CLASS_NA, FR_CONSTS

yaml = YAML()

class FRTableData:

  def __init__(self, annAuthor: str=None, annFile: str=None):
    self.filter = parameditors.FRTableFilterEditor()

    self.annAuthor = annAuthor
    self.annFile = annFile

    self.allFields: List[FRParam] = list(REQD_TBL_FIELDS)
    self.compClasses = [COMP_CLASS_NA]

  def makeCompDf(self, numRows=1) -> df:
    """
    Creates a dataframe for the requested number of components.
    This is the recommended method for component instantiation prior to table insertion.
    """
    df_list = []
    dropRow = False
    if numRows <= 0:
      # Create one row and drop it, which ensures data types are correct in the empty
      # dataframe
      numRows = 1
      dropRow = True
    for _ in range(numRows):
      # Make sure to construct a separate component instance for
      # each row no objects have the same reference
      df_list.append([field.value for field in copy.copy(self.allFields)])
    outDf = df(df_list, columns=self.allFields).set_index(REQD_TBL_FIELDS.INST_ID, drop=False)
    # Set the metadata for this application run
    outDf[REQD_TBL_FIELDS.ANN_AUTHOR] = self.annAuthor
    outDf[REQD_TBL_FIELDS.ANN_TIMESTAMP] = datetime.now().strftime(DATE_FORMAT)
    outDf[REQD_TBL_FIELDS.ANN_FILENAME] = FR_CONSTS.ANN_CUR_FILE_INDICATOR.value
    if dropRow:
      outDf = outDf.drop(index=REQD_TBL_FIELDS.INST_ID.value)
    return outDf

  def loadCfg(self, cfgFname: Union[str, Path, PathLike]):
    with open(cfgFname, 'r') as ifile:
      cfg: dict = yaml.load(ifile)
    self.resetLists()
    for compCls in cfg.get('classes', []):
      newParam = FRParam(compCls, group=self.compClasses)
      self.compClasses.append(newParam)
    for field, values in cfg.get('opt-tbl-fields', {}).items():
      if isinstance(values, dict):
        param = FRParam(field, **values)
      else:
        param = FRParam(field, values)
      param.group = self.allFields
      self.allFields.append(param)
    self.filter.updateParamList(self.allFields)

  def resetLists(self):
    for lst in self.allFields, self.compClasses:
      lst.clear()
    self.allFields.extend(list(REQD_TBL_FIELDS))