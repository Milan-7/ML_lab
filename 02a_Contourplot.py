#Visualize the n-dimensional data using contour plots.
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-3,3,100)
y=np.linspace(-3,3,100)
X,Y=np.meshgrid(x,y)
Z=np.exp(-(X**2 + Y**2))
plt.contour(X,Y,Z)
plt.show()
