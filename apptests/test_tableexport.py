from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s3a.generalutils import augmentException
from s3a import ComponentIO
from testingconsts import SAMPLE_IMG_FNAME


@pytest.mark.withcomps
def test_normal_export(sampleComps, tmp_path, app):
  io = app.compIo
  app.exportOnlyVis = False
  for ftype in io.handledIoTypes:
    curPath = tmp_path / f'normalExport - All IDs.{ftype}'
    doAndAssertExport(app, curPath, io, app.exportableDf, 'Normal export with all IDs not successful.')

@pytest.mark.withcomps
def test_filter_export(tmp_path, monkeypatch, app):
  io = app.compIo

  curPath = tmp_path / 'normalExport - Filtered IDs export all.csv'
  filterIds = np.array([0,3,2])
  sampleComps = app.compMgr.compDf
  with monkeypatch.context() as m:
    m.setattr(app, 'exportOnlyVis', False)
    m.setattr(app.compDisplay, 'displayedIds', filterIds)
    exportDf = app.exportableDf
  np.testing.assert_array_equal(exportDf.index, sampleComps.index,
                                'Export DF should not use only filtered IDs'
                                ' when not exporting only visible, but'
                                ' ID lists don\'t match.')
  # With export only visible false, should still export whole frame
  doAndAssertExport(app, curPath, io, exportDf, 'Normal export with filter ids passed not successful.')

  curPath = tmp_path / 'normalExport - Filtered IDs export filtered.csv'
  with monkeypatch.context() as m:
    m.setattr(app, 'exportOnlyVis', True)
    m.setattr(app.compDisplay, 'displayedIds', filterIds)
    exportDf = app.exportableDf
  np.testing.assert_array_equal(exportDf.index, filterIds,
                                'Export DF should use only filtered IDs when exporting only '
                                'visible, but ID lists don\'t match.')
  # With export only visible false, should still export whole frame
  doAndAssertExport(app, curPath, io, exportDf, 'Export with filtered ids not successful.')

def test_bad_import(tmp_path, app):
  io = app.compIo
  for ext in io.handledIoTypes:
    ofile = open(tmp_path/f'junkfile.{ext}', 'w')
    ofile.write('Vertices\nabsolute junk')
    ofile.close()
    with pytest.raises(Exception):
      io.buildFromCsv(tmp_path/f'junkfile.{ext}')


def doAndAssertExport(app, fpath: Path, io: ComponentIO, compDf: pd.DataFrame, failMsg: str):
  fpath = Path(fpath)
  try:
    io.exportByFileType(compDf, fpath, imShape=app.mainImg.image.shape[:2],
                        imgDir=SAMPLE_IMG_FNAME.parent)
  except ValueError as ve:
    if 'Full I/O' not in str(ve):
      raise
  except Exception as ex:
    augmentException(ex, f'{failMsg}\n')
    raise
  else:
    assert fpath.exists(), 'File doesn\'t exist despite export'
    try:
      inDf = io.buildByFileType(fpath, app.mainImg.image.shape[:2])
      assert len(inDf) > 0
    except ValueError:
      pass

def test_impossible_io(tmp_path, sampleComps, app):
  io = app.compIo
  with pytest.raises(IOError):
    io.exportByFileType(sampleComps, './nopossible.exporttype$')
  with pytest.raises(IOError):
    io.buildByFileType('./nopossible.importtype$')