from pathlib import Path
from io import StringIO

import numpy as np
import pytest

from appsetup import EXPORT_DIR, defaultApp_tester
from s3a.generalutils import augmentException
from s3a.projectvars import REQD_TBL_FIELDS
from s3a.structures import FRS3AException, FRComplexVertices, FRVertices, FRS3AWarning
from s3a.tablemodel import FRComponentIO

app, dfTester = defaultApp_tester()

def test_normal_export(sampleComps):
  io = app.compExporter
  io.exportOnlyVis = False
  io.prepareDf(sampleComps)
  for ftype in io.handledExportTypes:
    curPath = EXPORT_DIR / f'normalExport - All IDs.{ftype}'
    doAndAssertExport(curPath, io, 'Normal export with all IDs not successful.')



def test_filter_export(sampleComps):
  io = app.compExporter

  curPath = EXPORT_DIR / 'normalExport - Filtered IDs export all.csv'
  filterIds = np.array([0,3,2])
  io.exportOnlyVis = False
  io.prepareDf(sampleComps, filterIds)
  np.testing.assert_array_equal(io.compDf.index, sampleComps.index,
                                'Export DF should not use only filtered IDs'
                                ' when not exporting only visible, but'
                                ' ID lists don\'t match.')
  # With export only visible false, should still export whole frame
  doAndAssertExport(curPath, io, 'Normal export with filter ids passed not successful.')

  curPath = EXPORT_DIR / 'normalExport - Filtered IDs export filtered.csv'
  io.exportOnlyVis = True
  io.prepareDf(sampleComps, filterIds)
  np.testing.assert_array_equal(io.compDf.index, filterIds,
                                'Export DF should use only filtered IDs when exporting only '
                                'visible, but ID lists don\'t match.')
  # With export only visible false, should still export whole frame
  doAndAssertExport(curPath, io, 'Export with filtered ids not successful.')

def test_bad_import():
  io = app.compExporter
  ofile = open(EXPORT_DIR/'junkfile.csv', 'w')
  ofile.write('Vertices\nabsolute junk')
  ofile.close()
  with pytest.raises(Exception):
    io.buildFromCsv(EXPORT_DIR/'junkfile.csv')


def doAndAssertExport(fpath: Path, io: FRComponentIO, failMsg: str):
  try:
    io.exportByFileType(fpath)
  except Exception as ex:
    augmentException(ex, f'{failMsg}\n')
    raise
  assert fpath.exists(), 'File doesn\'t exist despite export'