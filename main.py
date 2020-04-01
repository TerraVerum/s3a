#!/usr/bin/env python
from pathlib import Path

from cdef.__main__ import main
from cdef.projectvars import BASE_DIR
sampleImPath = Path(BASE_DIR).joinpath('..', 'Images', 'circuitBoard.png').absolute()
main(sampleImPath)