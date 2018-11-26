import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

def PlotHistogram3D(x, y, z):
    data_array = np.array(z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_data, y_data = np.meshgrid( np.arange(data_array.shape[1]),np.arange(data_array.shape[0]) )
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()
    #ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1,1,z_data)
    ax.scatter(x_data, y_data, z_data)
    plt.show()

def PlotHistogram2D(hist, xedges, yedges):
    fig = plt.figure(figsize=(7, 3)) 
    ax = fig.add_subplot(111, title='Offset Histogram: HeatMap') 
    plt.imshow(hist, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()

def ScatterPlot3D(x, y, z, domain):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-domain[0], domain[0]])
    ax.set_ylim([-domain[1], domain[1]])
    plt.show()