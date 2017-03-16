import h5py

f = h5py.File("/home/axeleik/Downloads/cremi.paths.crop.split_z.h5", mode='r')


def printname(name):
    print name

#f.visit(printname)
z_predict0/truepaths/z/0/beta_0.5/11/0

f = h5py.File("/home/axeleik/Downloads/testdata.h5", mode='r')
image = f["/volumes/raw"][0,:,:]
