
from neuro_seg_plot import NeuroSegPlot as nsp
import numpy as np
from mayavi import mlab

path = np.array([[1, 1, 1], [20, 10, 10], [20, 20, 10], [20, 30, 20]], dtype=np.uint32)
path = np.swapaxes(path, 0, 1)

nsp.start_figure()
nsp.add_path(path)
nsp.show()
