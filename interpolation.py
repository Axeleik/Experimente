

import h5py
import scipy
from scipy import interpolate
import numpy as np
import curvature

def curve_slice(array):
    #slice small_array into curves
    i0=0
    i1=0
    i2 = 0
    ausgabe=[np.array(([1,2,3]),0),(([1,2,3]),0)]
    for item in array:

        if array[i0][1]+1 == array[i0+1][1]:
            ausgabe[i1][i2] = array[i0]
            i2=i2+1
        else:
            i2=1
            i1=i1+1
            ausgabe[i1].append(np.array(array[i0]))
        i0=i0+1
    print "\n\n Ausgabe:\n\n",i0," ",ausgabe
    return ausgabe


f = h5py.File("/home/axeleik/Downloads/cremi.paths.crop.split_z.h5", mode='r')
data = np.array(f["z_predict0/truepaths/z/0/beta_0.5/247/0"])[:,:]

data = np.array([(elem1, elem2,elem3*10) for elem1, elem2,elem3 in data])

#This is your data, but we're 'zooming' into just 5 data points
#because it'll provide a better visually illustration
#also we need to transpose to get the data in the right format
data = data.transpose()


#now we get all the knots and info about the interpolated spline
tck, u= interpolate.splprep(data,s=50000)

#here we generate the new interpolated dataset,
#increase the resolution by increasing the spacing, 500 in this example
new = interpolate.splev(np.linspace(0,1,10000), tck)


#now lets plot it!
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


x=int(input("Skala: "))

curves=curvature.plot_tf(np.array(new).transpose(),"t",x,10,20,1,2)
print "curves: " , curves
#curvature.plot_tf(np.diff(np.array(new)),"t",1,1,1,3,4)
y=float(input("Kleiner als: "))

#alle vektoren <y in ein neues array
small = np.array([(elem1, elem2) for (elem1, elem2) in curves if elem2<y])

# print "small: ", small
print "small: " , small
#nummer der pixel raus sortieren
orte=[elem1 for (elem1, elem2) in small]
print "orte: ", orte
print "len(orte): ",len(orte)
print "len(small): ",len(small)


new_new= np.array(new).transpose()
#print "\n\nnew_new",new_new.transpose()
print "new_new.shape:",new_new.shape

small_array = np.array([elem1 for elem1 in np.array(new_new)[orte]])


where= np.where(np.diff(np.array(orte))>1)
where= where[0]
print "where: ",where


diff =np.diff(np.array(orte))
print "diff: ", diff
print "diff[where]: ", diff[where]

ein=[]
i0 = 0
i1 = 0

for item in where:

    ein.append((np.array(small_array[i0:(where[i1]+1)])))
    i0=where[i1]+1
    i1=i1+1


ein.append((np.array(small_array[i0:])))

#print "ein: ", ein[0]
#print "ein2: ", ein[0].transpose()
#print "len(ein[0]): ", len(ein[0])+len(ein[1])+len(ein[2])+len(ein[3])

#print "small_array: ", small_array[1:10]

#orte.append(small_array[1:10])
#print "orte append: ", orte



#small_array= curve_slice(small_array)





#print "len(small_array): ", len(small_array)

new_new=small_array.transpose()
#print "new_new: ", new_new
#print "new_new:",new_new
#print "len(new_new[1]): ", len(new_new[1])
#ax.plot(new_new[0], new_new[1], new_new[2], label='small', lw =2, c='yellow')

fig = plt.figure(1)
ax = Axes3D(fig)
ax.plot(data[0], data[1], data[2], label='original_true', lw =2, c='Dodgerblue') #gezackt
ax.plot(new[0], new[1], new[2], label='fit_true', lw =3, c='red') #plot
ax.plot(new[0][6500:7500], new[1][6500:7500], new[2][6500:7500], label='fit_true', lw =4, c='green') #plot
ax.plot(new[0][200:1000], new[1][200:1000], new[2][200:1000], label='fit_true', lw =4, c='green') #plot

#print "len(ein): ",len(ein)
ax.plot(ein[0].transpose()[0], ein[0].transpose()[1], ein[0].transpose()[2], label='small', lw =2, c='yellow')
i2=0
while i2<len(ein):
    ax.plot(ein[i2].transpose()[0], ein[i2].transpose()[1], ein[i2].transpose()[2], lw=2, c='yellow')
    i2=i2+1

ax.legend()


plt.show()

