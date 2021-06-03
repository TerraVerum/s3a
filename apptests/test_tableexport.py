from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from apptests.helperclasses import CompDfTester
from apptests.testingconsts import SAMPLE_SMALL_IMG_FNAME, SAMPLE_IMG_FNAME, SAMPLE_SMALL_IMG, TEST_FILE_DIR
from s3a.generalutils import augmentException
from s3a import ComponentIO, REQD_TBL_FIELDS, ComplexXYVertices, XYVertices, PRJ_CONSTS as CNST
from s3a.parameditors.table import TableData

@pytest.fixture
def _simpleTbl(tmp_path):
  td = TableData()
  td.loadCfg(tmp_path/'test.cfg', {'fields': {'List': list('abcdefg')}})
  return td

@pytest.mark.withcomps
def test_normal_export(sampleComps, tmp_path, app):
  io = app.compIo
  app.props[CNST.EXP_ONLY_VISIBLE] = False
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
    m.setitem(app.props, CNST.EXP_ONLY_VISIBLE, False)
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
    m.setitem(app.props, CNST.EXP_ONLY_VISIBLE, True)
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
                        srcDir=SAMPLE_IMG_FNAME.parent)
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

@pytest.mark.withcomps
def test_compimgs_export(tmp_path, _simpleTbl):
  io = ComponentIO()
  io.tableData = _simpleTbl
  tester = CompDfTester(100, tableData=_simpleTbl)
  tester.compDf[REQD_TBL_FIELDS.INST_ID] = tester.index
  tester.fillRandomVerts(SAMPLE_SMALL_IMG.shape[:2])
  tester.compDf.loc[:50, REQD_TBL_FIELDS.SRC_IMG_FILENAME] = SAMPLE_SMALL_IMG_FNAME
  tester.compDf.loc[50:, REQD_TBL_FIELDS.SRC_IMG_FILENAME] = SAMPLE_IMG_FNAME
  tester.compDf.loc[50:, REQD_TBL_FIELDS.INST_ID] = tester.compDf.index[:50]
  tester.compDf.index = np.concatenate([tester.compDf.index[:50], tester.compDf.index[:50]])

  # Do df export just to test the output file capability and various options
  io.exportCompImgsDf(tester.compDf, tmp_path/'test.pkl', lblField='List')
  assert (tmp_path/'test.pkl').exists()
  df = io.exportCompImgsDf(tester.compDf, asIndiv=True)
  for mask in df.labelMask:
    assert len(np.unique(mask)) <= 2

  outPath = tmp_path/'out'
  io.exportCompImgsZip(tester.compDf, outPath)
  for path_ in outPath, outPath/'data', outPath/'labels':
    assert path_.exists()
  assert [f.name for f in (outPath/'data').iterdir()] == [f.name for f in (outPath/'labels').iterdir()]

  for file in (outPath/'data').iterdir():
    with Image.open(file) as dataImg:
      with Image.open(outPath/'labels'/file.name) as lblImg:
        assert dataImg.size == lblImg.size

  io.exportCompImgsZip(tester.compDf, outPath, (300,300))
  for file in (outPath / 'data').iterdir():
    with Image.open(file) as dataImg:
      with Image.open(outPath / 'labels' / file.name) as lblImg:
        assert dataImg.size == lblImg.size == (300,300)

  io.exportCompImgsZip(tester.compDf, tmp_path/'out_zip', archive=True)
  assert (tmp_path/'out_zip.zip').exists()

def test_convert(app, tmp_path):
  sampleComps = app.exportableDf
  io = ComponentIO()
  pklFile = tmp_path/'pklexport.pkl'
  csvFile = tmp_path/'csvexport.csv'
  io.exportPkl(sampleComps, pklFile)
  io.convert(pklFile, csvFile)
  assert (tmp_path/'csvexport.csv').exists()
  reads = [io.importByFileType(f) for f in [pklFile, csvFile]]
  assert np.array_equal(*reads)

def test_lblpng_export(_simpleTbl):
  io = ComponentIO()
  io.tableData = _simpleTbl
  tester = CompDfTester(15, tableData=_simpleTbl)
  sampleComps = tester.compDf

  with pytest.raises(ValueError):
    io.exportLblPng(sampleComps, lblField='badlbl')
  with pytest.raises(ValueError):
    io.exportLblPng(sampleComps, bgColor=-1)

  export, mapping = io.exportLblPng(sampleComps, rescaleOutput=True, returnLblMapping=True)
  assert np.all(np.isin(mapping.index, export))
  assert np.max(mapping.index) > np.max(sampleComps.index)

  field = _simpleTbl.fieldFromName('List')
  export, mapping = io.exportLblPng(sampleComps, returnLblMapping=True, lblField=field)
  assert (mapping.to_numpy() == field.opts['limits']).all()

  sampleComps[field] = 'a'
  # Make sure full mapping is made even when not all values exist
  export, mapping = io.exportLblPng(sampleComps, returnLblMapping=True, lblField=field)
  assert (mapping.to_numpy() == field.opts['limits']).all()

def test_geojson_import():
  # Limited support for now so need tmp files to test this functionality
  io = ComponentIO()
  with pytest.raises(ValueError):
    io.importByFileType(TEST_FILE_DIR/'sample.geojson')
  df = io.importByFileType(TEST_FILE_DIR/'sample.geojson', parseErrorOk=True)
  assert len(df) == 1
  cmpVerts = XYVertices([[100, 0], [101, 0], [101, 1], [100,1], [100,0]])
  loadedVerts = df.at[df.index[0], REQD_TBL_FIELDS.VERTICES].stack()
  assert (loadedVerts == cmpVerts).all()

def test_superannotate_import():
  # Limited support for now so need tmp files to test this functionality
  io = ComponentIO()
  file = TEST_FILE_DIR/'sample.superannotate.json'
  with pytest.raises(ValueError):
    io.importByFileType(file)
  df = io.importByFileType(file, parseErrorOk=True).reset_index(drop=True)
  assert len(df) == 3
  loadedVerts = df.at[df.index[1], REQD_TBL_FIELDS.VERTICES].stack()
  assert len(loadedVerts) == 17