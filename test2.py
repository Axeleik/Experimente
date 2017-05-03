from mayavi.mlab import contour3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = np.zeros((100,100,100), dtype=np.int32)
a[50,50,:] = 1
a[48:52,48:52,48:52] = 1
a[46:54,46:54,70:78] = 1

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

line1 = ax.plot(a,'ok')

plt.show()