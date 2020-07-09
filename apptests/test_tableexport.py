from pathlib import Path

import numpy as np
import pytest

from conftest import app
from s3a.generalutils import augmentException
from s3a.models.tablemodel import FRComponentIO

def test_normal_export(sampleComps, tmpdir):
  io = app.compExporter
  io.exportOnlyVis = False
  io.prepareDf(sampleComps)
  for ftype in io.handledExportTypes:
    curPath = tmpdir / f'normalExport - All IDs.{ftype}'
    doAndAssertExport(curPath, io, 'Normal export with all IDs not successful.')



def test_filter_export(sampleComps, tmpdir):
  io = app.compExporter

  curPath = tmpdir / 'normalExport - Filtered IDs export all.csv'
  filterIds = np.array([0,3,2])
  io.exportOnlyVis = False
  io.prepareDf(sampleComps, filterIds)
  np.testing.assert_array_equal(io.compDf.index, sampleComps.index,
                                'Export DF should not use only filtered IDs'
                                ' when not exporting only visible, but'
                                ' ID lists don\'t match.')
  # With export only visible false, should still export whole frame
  doAndAssertExport(curPath, io, 'Normal export with filter ids passed not successful.')

  curPath = tmpdir / 'normalExport - Filtered IDs export filtered.csv'
  io.exportOnlyVis = True
  io.prepareDf(sampleComps, filterIds)
  np.testing.assert_array_equal(io.compDf.index, filterIds,
                                'Export DF should use only filtered IDs when exporting only '
                                'visible, but ID lists don\'t match.')
  # With export only visible false, should still export whole frame
  doAndAssertExport(curPath, io, 'Export with filtered ids not successful.')

def test_bad_import(tmpdir):
  io = app.compExporter
  for ext in io.handledExportTypes:
    ofile = open(tmpdir/f'junkfile.{ext}', 'w')
    ofile.write('Vertices\nabsolute junk')
    ofile.close()
    with pytest.raises(Exception):
      io.buildFromCsv(tmpdir/f'junkfile.{ext}')


def doAndAssertExport(fpath: Path, io: FRComponentIO, failMsg: str):
  try:
    io.exportByFileType(fpath)
  except Exception as ex:
    augmentException(ex, f'{failMsg}\n')
    raise
  assert fpath.exists(), 'File doesn\'t exist despite export'