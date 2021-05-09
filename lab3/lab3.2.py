import pandas as pd
import numpy as np
from sklearn import linear_model, pipeline, preprocessing
import matplotlib.pyplot as plt

# Wczytanie danych
data = pd.read_csv("data6.tsv", delim_whitespace=True, header=None)

x = data.iloc[:, 0].to_numpy().reshape(-1, 1)
y = data.iloc[:, 1].to_numpy()

# 1. pierwszego stopnia (funkcja liniowa):
regression = linear_model.LinearRegression().fit(x,y)
plt.scatter(x, y, color="green")
plt.plot(x, regression.predict(x), color="red")
plt.show()

# 2. drugiego stopnia (funkcja kwadratowa)
regression_2 = pipeline.make_pipeline(preprocessing.PolynomialFeatures(2), linear_model.LinearRegression()).fit(x, y)
prediction = regression_2.predict(np.linspace(x.min(), x.max()).reshape(-1, 1))
plt.scatter(x, y, color="green")
plt.plot(np.linspace(x.min(), x.max()).reshape(-1, 1), prediction, color="red")
plt.show()

# 3. piątego stopnia (wielomian 5. stopnia)
regression_5 = pipeline.make_pipeline(preprocessing.PolynomialFeatures(5), linear_model.LinearRegression()).fit(x, y)
prediction = regression_5.predict(np.linspace(x.min(), x.max()).reshape(-1, 1))
plt.scatter(x, y, color="green")
plt.plot(np.linspace(x.min(), x.max()).reshape(-1, 1), prediction, color="red")
plt.show()
