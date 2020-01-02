import pyqtgraph as pg
from cProfile import run
from MainWindow import MainWindow
## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  app = pg.mkQApp()
  win = MainWindow()
  stmt = '(lambda times: [win.estBoundsBtnClicked() for ii in range(times)])(1)'
  #p = run(stmt)
  win.show()
  app.exec()