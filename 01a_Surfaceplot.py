#Visualize the n-dimensional data using 3D surface plots.
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

x=np.linspace(-3,3,100)
y=np.linspace(-3,3,100)
X,Y=np.meshgrid(x,y)
Z=np.exp(-(X**2 + Y**2))
fig=go.Figure(data=[go.Surface(z=Z)])
fig.show()
