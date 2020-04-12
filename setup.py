from setuptools import setup, find_packages

from pathlib import Path

def _mltGlob(curDir, *globs):
  curDir = Path(curDir)
  outFiles = []
  for curGlob in globs:
    outFiles.extend(curDir.glob(curGlob))
  outFiles = [str(f) for f in outFiles]
  return outFiles

setup(
  name='cdef',
  version='0.1.0',
  package_dir={'cdef':'cdef/cdef'},
  # include_package_data=True,
  url='https://gitlab.com/ficsresearch/cdef',
  license='',
  author='Nathan Jessurun',
  author_email='njessurun@ufl.edu',
)
