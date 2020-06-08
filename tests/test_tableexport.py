from pathlib import Path

import numpy as np

from appsetup import EXPORT_DIR, defaultApp_tester
from cdef.generalutils import augmentException
from cdef.tablemodel import FRComponentIO

app, dfTester = defaultApp_tester()

def test_normal_export(sampleComps):
  io = app.compExporter
  io.exportOnlyVis = False
  curPath = EXPORT_DIR / 'normalExport - All IDs.csv'
  io.prepareDf(sampleComps)
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


def doAndAssertExport(fpath: Path, io: FRComponentIO, failMsg: str):
  try:
    io.exportCsv(str(fpath))
  except Exception as ex:
    augmentException(ex, f'{failMsg}\n')
    raise
  assert fpath.exists(), 'Csv file doesn\'t exist despite export'