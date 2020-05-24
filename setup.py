from setuptools import setup, find_packages

from pathlib import Path

def _mltGlob(curDir, *globs):
  curDir = Path(curDir)
  outFiles = []
  for curGlob in globs:
    outFiles.extend(curDir.glob(curGlob))
  outFiles = [str(f) for f in outFiles]
  return outFiles

here = Path(__file__).parent
lines = (here/'requirements.txt').open('r').readlines()
required = []
for line in lines:
  if not line.startswith('#'):
    required.append(line.strip('\n'))


setup(
  name='cdef',
  version='0.1.0',
  package_dir={'cdef':'cdef'},
  packages=find_packages(),
  data_files=[('cdef/cdef/icons', _mltGlob('cdef/icons', '*.svg', '*.png')),
  ('cdef/cdef/', _mltGlob('cdef/', '*.yml'))],
  install_requires=required,
  include_package_data=True,
  url='https://gitlab.com/ficsresearch/cdef',
  license='',
  author='Nathan Jessurun',
  author_email='njessurun@ufl.edu',
)
