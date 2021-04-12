import numpy as np
from sklearn import linear_model
import csv

reader = csv.reader(open('fires_thefts.csv'), delimiter=',')

x = list()
y = list()
for xi, yi in reader:
    x.append(float(xi))
    y.append(float(yi))

# Tworzenie modelu regresji liniowej
regression = linear_model.LinearRegression()

# Trenowanie modelu
regression.fit(np.array(x).reshape(-1, 1), y)

# Przewidywana liczba włamań dla 50, 100, 200 pożarów
predicted = regression.predict(np.array([50]).reshape(-1,1))
print('50 pożarów: ' + str(predicted[0]))
predicted = regression.predict(np.array([100]).reshape(-1,1))
print('100 pożarów: ' + str(predicted[0]))
predicted = regression.predict(np.array([200]).reshape(-1,1))
print('200 pożarów: ' + str(predicted[0]))
