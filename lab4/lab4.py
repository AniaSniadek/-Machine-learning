# Konieczne importy
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import model_selection

# Przygotowanie danych
data = pd.read_csv("vote.csv")

x = data.iloc[:, 0:-1].to_numpy()
y = data.iloc[:, -1].to_numpy().reshape(-1, 1)

# Podział danych na zbiory uczący i testowy
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

# Stworzenie modelu
model = keras.Sequential(
    [
        layers.Dense(7, activation="sigmoid"),
        layers.Dropout(rate=0.2),
        layers.Dense(5, activation="sigmoid"),
        layers.Dropout(rate=0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Uczenie modelu
batch_size = 32
epochs = 200

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Ewaluacja modelu
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
