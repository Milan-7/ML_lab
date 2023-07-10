#Visualize the n-dimensional data using Box-plot.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# create some sample data
np.random.seed(1)
n = 100
data = np.random.randn(n, 4)

sns.boxplot(data=data)
plt.show()
