import csv
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data2.csv')
data_array = data.to_numpy()

x = data_array[:, 3]
y = data_array[:, 4]

plt.plot(x, y, 'ro')

plt.show()
