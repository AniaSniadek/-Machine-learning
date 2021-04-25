import pandas as pd
from sklearn import linear_model, preprocessing, model_selection

# Wczytanie danych
data = pd.read_csv("mushrooms.tsv", sep="\t").to_numpy()

x = data[:, 1:10]
y = data[:, 0].reshape(-1, 1)

x_OneHotEncoder = preprocessing.OneHotEncoder()
x = x_OneHotEncoder.fit(x).transform(x).toarray()
y_OneHotEncoder = preprocessing.OneHotEncoder()
y = y_OneHotEncoder.fit(y).transform(y).toarray()[:, 1]

# Podział danych na zbiory uczący i testowy
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

# Uczenie modelu
model = linear_model.LogisticRegression().fit(x_train, y_train)

# Predykcja wyników dla danych testowych
Y_expected = y_test.astype(int)
Y_predicted = model.predict(x_test).astype(int)

# Obliczmy TP, TN, FP i FN
tp = 0
tn = 0
fp = 0
fn = 0

for i in range(len(Y_expected)):
    if Y_expected[i] == 1 and Y_predicted[i] == 1:
        tp += 1
    elif Y_expected[i] == 0 and Y_predicted[i] == 0:
        tn += 1
    elif Y_expected[i] == 0 and Y_predicted[i] == 1:
        fp += 1
    elif Y_expected[i] == 1 and Y_predicted[i] == 0:
        fn += 1
        
accuracy = (tp + tn) / (tp + tn + fp + fn)
print('Accuracy:', accuracy)
precision = tp / (tp + fp)
print('Precision:', precision)
recall = tp / (tp + fn)
print('Recall:', recall)
fscore = (2 * precision * recall) / (precision + recall)
print('F-score:', fscore)
