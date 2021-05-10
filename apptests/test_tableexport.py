from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s3a.generalutils import augmentException
from s3a import ComponentIO, REQD_TBL_FIELDS
from testingconsts import SAMPLE_IMG_FNAME


@pytest.mark.withcomps
def test_normal_export(sampleComps, tmp_path, app):
  io = app.compIo
  app.exportOnlyVis = False
  for ftype in io.exportTypes:
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
  for ext in io.importTypes:
    ofile = open(tmp_path/f'junkfile.{ext}', 'w')
    ofile.write('Vertices\nabsolute junk')
    ofile.close()
    with pytest.raises(Exception):
      io.importCsv(tmp_path/f'junkfile.{ext}')

@pytest.mark.withcomps
def test_bad_integrity(tmp_path, app, monkeypatch, qtbot):
  oldBuild = app.compIo.importByFileType
  def badBuilder(*args, **kwargs):
    df = oldBuild(*args, **kwargs)
    df[REQD_TBL_FIELDS.VERTICES] = 5
    return df

  with monkeypatch.context() as m:
    m.setattr(app.compIo, 'importByFileType', badBuilder)
    with pytest.warns(UserWarning):
      app.exportCurAnnotation(tmp_path/'test.csv', verifyIntegrity=True)

def test_serial_export(tmp_path, sampleComps, app):
  with pytest.raises(ValueError):
    app.compIo.exportSerialized(sampleComps, tmp_path/'test.nopandastypehere')

  # No export without a file path
  app.compIo.exportSerialized(sampleComps)
  assert len(list(tmp_path.glob('*.*'))) == 0

  sampleComps = sampleComps.copy()
  # Put in data that pandas can't handle
  class BadRep:
    def __str__(self):
      raise AttributeError
  sampleComps[REQD_TBL_FIELDS.INST_ID] = BadRep()
  with pytest.raises(Exception):
    app.compIo.exportSerialized(sampleComps, tmp_path/'test.csv')


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
      inDf = io.importByFileType(fpath, app.mainImg.image.shape[:2])
      assert len(inDf) > 0
    except (ValueError, IOError):
      pass

def test_impossible_io(tmp_path, sampleComps, app):
  io = app.compIo
  with pytest.raises(IOError):
    io.exportByFileType(sampleComps, './nopossible.exporttype$')
  with pytest.raises(IOError):
    io.importByFileType('./nopossible.importtype$')

@pytest.mark.withcomps
def test_opts_insertion(app, sampleComps, tmp_path):
  io = app.compIo
  fn = io._ioWrapper(io.importSerialized)
  io.importOpts['reindex'] = True
  sampleComps.index = np.linspace(0, 10000, len(sampleComps), dtype=int)
  io.exportSerialized(sampleComps, tmp_path/'test.csv')
  imported = fn(tmp_path/'test.csv')
  assert not (imported.index == sampleComps.index).all()
