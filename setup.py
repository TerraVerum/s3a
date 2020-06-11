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
  name='s3a',
  version='0.1.0',
  package_dir={'s3a':'s3a'},
  packages=find_packages(),
  data_files=[('s3a/s3a/icons', _mltGlob('s3a/icons', '*.svg', '*.png')),
  ('s3a/s3a', _mltGlob('s3a/', '*.yml'))],
  install_requires=required,
  include_package_data=True,
  url='https://gitlab.com/ficsresearch/s3a',
  license='',
  author='Nathan Jessurun',
  author_email='njessurun@ufl.edu',
)
