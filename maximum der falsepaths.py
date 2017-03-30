from Curvature_final import maximum_ausgeben
import h5py
import scipy
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
import pickle
import time

def printname(name):

    if name.count('/') == 6 :

        glob.append(name)






start_time = time.time()
# your code

glob=[]


f = h5py.File("/home/axeleik/Downloads/cremi.paths.crop.split_z.h5", mode='r')

f.visit(printname)

print "glob: ", glob
print "len(glob): ", len(glob)
data = np.array(f[u'z_predict0/falsepaths/z/0/beta_0.5/247/0'])

array=np.array([["placeholder",0]])
a=0

for i in glob:
    data=np.array(f[i])
    array = np.append(array, [[i,maximum_ausgeben(data)]], axis=0)
    print a
    a=a+1

print array
#np.savetxt("/home/axeleik/Documents/curves.txt", array,fmt="%s")
with open("/home/axeleik/Documents/curves.pkl", mode='w') as f:
    pickle.dump(array, f)
elapsed_time = time.time() - start_time
print "duration: ", elapsed_time


