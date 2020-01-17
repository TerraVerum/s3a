import pyqtgraph as pg
from cProfile import run
from MainWindow import MainWindow
from constants import BASE_DIR

from os import path

def profileFunc(func, numTimes, *funcArgs, **funcKwargs):
  for _ in range(numTimes):
    func(*funcArgs, **funcKwargs)

if __name__ == '__main__':
  startImgFpath = path.join(BASE_DIR, './Images/fast.tif')
  app = pg.mkQApp()
  win = MainWindow(startImgFpath)
  #p = run('profileFunc(win.estBoundsBtnClicked, 1)')
  win.show()
  app.exec()

