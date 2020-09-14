from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import app
from s3a.generalutils import augmentException
from s3a import FRComponentIO
from s3a.structures import FRIOError


def test_normal_export(sampleComps, tmpdir):
  io = app.compIo
  io.exportOnlyVis = False
  for ftype in io.handledIoTypes:
    curPath = tmpdir / f'normalExport - All IDs.{ftype}'
    doAndAssertExport(curPath, io, sampleComps.copy(), 'Normal export with all IDs not successful.')

@pytest.mark.withcomps
def test_filter_export(tmpdir, monkeypatch):
  io = app.compIo

  curPath = tmpdir / 'normalExport - Filtered IDs export all.csv'
  filterIds = np.array([0,3,2])
  sampleComps = app.compMgr.compDf
  with monkeypatch.context() as m:
    m.setattr(io, 'exportOnlyVis', False)
    m.setattr(app.compDisplay, 'displayedIds', filterIds)
    exportDf = app.exportableDf
  np.testing.assert_array_equal(exportDf.index, sampleComps.index,
                                'Export DF should not use only filtered IDs'
                                ' when not exporting only visible, but'
                                ' ID lists don\'t match.')
  # With export only visible false, should still export whole frame
  doAndAssertExport(curPath, io, exportDf, 'Normal export with filter ids passed not successful.')

  curPath = tmpdir / 'normalExport - Filtered IDs export filtered.csv'
  with monkeypatch.context() as m:
    m.setattr(io, 'exportOnlyVis', True)
    m.setattr(app.compDisplay, 'displayedIds', filterIds)
    exportDf = app.exportableDf
  np.testing.assert_array_equal(exportDf.index, filterIds,
                                'Export DF should use only filtered IDs when exporting only '
                                'visible, but ID lists don\'t match.')
  # With export only visible false, should still export whole frame
  doAndAssertExport(curPath, io, exportDf, 'Export with filtered ids not successful.')

def test_bad_import(tmpdir):
  io = app.compIo
  for ext in io.handledIoTypes:
    ofile = open(tmpdir/f'junkfile.{ext}', 'w')
    ofile.write('Vertices\nabsolute junk')
    ofile.close()
    with pytest.raises(Exception):
      io.buildFromCsv(tmpdir/f'junkfile.{ext}')


def doAndAssertExport(fpath: Path, io: FRComponentIO, compDf: pd.DataFrame, failMsg: str):
  fpath = Path(fpath)
  try:
    io.exportByFileType(compDf, fpath)
  except Exception as ex:
    augmentException(ex, f'{failMsg}\n')
    raise
  assert fpath.exists(), 'File doesn\'t exist despite export'
  inDf = io.buildByFileType(fpath, app.mainImg.image.shape[:2])
  assert len(inDf) > 0

def test_impossible_io(tmpdir, sampleComps):
  io = app.compIo
  with pytest.raises(FRIOError):
    io.exportByFileType(sampleComps, './nopossible.exporttype$')
  with pytest.raises(FRIOError):
    io.buildByFileType('./nopossible.importtype$')