import numpy as np
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt

def curvtest(data):
    dx_dt = np.gradient(data[0])
    dy_dt = np.gradient(data[1])
    dz_dt = np.gradient(data[2])

    velocity = np.array([[dx_dt[i], dy_dt[i], dz_dt[i]] for i in range(dx_dt.size)]) #

    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt+ dz_dt * dz_dt) #
    print ds_dt.transpose()
    print np.array([1 / ds_dt] * 2).transpose().shape
    print velocity

    tangent = np.array([1 / ds_dt] * 3).transpose() * velocity

    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]
    tangent_z = tangent[:, 2]

    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)
    deriv_tangent_z = np.gradient(tangent_z)

    dT_dt = np.array([[deriv_tangent_x[i], deriv_tangent_y[i], deriv_tangent_z[i]] for i in range(deriv_tangent_x.size)])

    length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y + deriv_tangent_z * deriv_tangent_z)
    normal = np.array([1 / length_dT_dt] * 3).transpose() * dT_dt
    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    d2z_dt2 = np.gradient(dz_dt)

    curvature = np.sqrt((d2z_dt2 * dy_dt - dz_dt * d2y_dt2)**2 + (d2x_dt2 * dz_dt - dx_dt * d2z_dt2)**2 + (d2y_dt2 * dx_dt - dy_dt * d2x_dt2)**2) / (dx_dt * dx_dt + dy_dt * dy_dt + dz_dt * dz_dt) ** 1.5
    t_component = np.array([d2s_dt2] * 3).transpose()
    n_component = np.array([curvature * ds_dt * ds_dt] * 3).transpose()

    acceleration = t_component * tangent + n_component * normal
    size = n_component.shape[0]
    new = np.zeros((size, 2))

    for i in xrange(0, size):
        new[i, 0] = 1 + i
        new[i,1] = np.linalg.norm(n_component[i])

    plt.figure()
    a2 = plt.scatter(*zip(*new), color="red")
    plt.suptitle('Curvature')
    plt.show()

f = h5py.File("/home/axeleik/Documents/data/cremi.splB.paths.h5", mode='r')


data = np.array(f["z_predict1/truepaths/z/1/beta_0.5/85/3"])
data = np.array([(elem1, elem2, elem3 * 10) for elem1, elem2, elem3 in data])
data = data.transpose()

tck, u = interpolate.splprep(data, s=10000)

new = interpolate.splev(np.linspace(0, 1, 250), tck)


curvtest(new)




import pickle

a = [1, 2]


with open('filename.pkl', mode='w') as f:
    pickle.dump(a, f)


with open('filename.pkl', mode='r') as f:
    b = pickle.load(f)

