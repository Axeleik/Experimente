
import vigra.filters as vig
import numpy as np

# a = np.zeros((200, 200, 100))
#
# print a.shape
#
# b = vigra.filters.gaussianSmoothing(a, 3)
#
# print a.dtype
#
# a = a.astype(np.uint32)
#
# print a.dtype
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import h5py

f = h5py.File("/home/axeleik/Downloads/testdata.h5", mode='r')
print (f.items())
print(f["/volumes"].items())
print('\n', f["/volumes/raw"])

#image = f.create_dataset(data="/volumes/raw",shape=(0,0,0),)

image = f["/volumes/raw"][0,:,:]
plt.imshow(image)

print "\n",image.dtype
#image2 = image[:,:,0]
print (image)
#img=mpimg.imread('/home/axeleik//Documents/image3.png')
#print img.shape
#print "\n Bild:","\n",img
plt.figure()

gauss= vig.gaussianSmoothing(image,3)

i=1
while i<30:
    gauss = vig.gaussianSmoothing(gauss, (3,2))
    i=i+1
print (gauss)
plt.imshow(gauss)
plt.show()
f.close()

"""with h5py.File() as f:

    image = f[...]

# TODO: Distance transform auf segmentierung
# TODO: Filter auf raw data (z.b. Gauss glaettung)

from matplotlib import pyplot as plt

plt.imshow(image[]"""

