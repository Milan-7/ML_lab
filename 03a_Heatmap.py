#Visualize the n-dimensional data using heat-map.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1)
n = 100
data = np.random.randn(n, 4)

sns.heatmap(data)
plt.show()
