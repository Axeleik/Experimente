
import h5py
import scipy
from scipy import interpolate
import numpy as np
import curvature
import matplotlib.pyplot as plt

f = h5py.File("/home/axeleik/Downloads/cremi.paths.crop.split_z.h5", mode='r')
data = np.array(f["z_predict0/truepaths/z/0/beta_0.5/247/0"])[:,:]

data = np.array([(elem1, elem2,elem3*10) for elem1, elem2,elem3 in data])

data = data.transpose()


#now we get all the knots and info about the interpolated spline
tck, u= interpolate.splprep(data,s=50000)

#here we generate the new interpolated dataset,
#increase the resolution by increasing the spacing, 500 in this example
new = interpolate.splev(np.linspace(0,1,500), tck)
print "data: ", data


print "diff: ", np.diff(new,axis=1).transpose()

len1 = curvature.length(np.diff(new,axis=1).transpose(), 1)

#len1=len1.transpose()
print "len1: ", len1
plt.figure()
plt.subplot(2, 1, 1)

a1 = plt.scatter(*zip(*len1), color="red")
plt.show()
