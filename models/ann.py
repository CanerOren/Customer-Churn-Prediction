import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import (
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam, SGD

from imblearn.over_sampling import SMOTE

path= "models/Customer-Churn-Records.csv"
churn_df = pd.read_csv(path)
churn_df = churn_df.drop(["RowNumber", "Surname", "CustomerId", "Complain"], axis=1)

mean_balance = churn_df['Balance'].replace(0, churn_df['Balance'].mean())
churn_df['Balance']=mean_balance
X = churn_df.drop("Exited", axis=1)
y = churn_df["Exited"]

X.to_csv('X_nonprocessed.csv',index=False)

hot = pd.get_dummies(X[["Geography", "Gender", "Card Type"]])
X = pd.concat([X, hot], axis=1)
X = X.drop(["Geography", "Gender", "Card Type"], axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

smote = SMOTE(random_state=42)

X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential()
model.add(Dense(16, activation="relu", input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))
optimzer = SGD(learning_rate=0.1, momentum=0.2)
model.compile(optimizer=optimzer, loss="binary_crossentropy", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)
predicted_probabilities = predictions.flatten()

adjusted_probabilities = []
for label, prob in zip(predicted_labels, predicted_probabilities):
    if label == 0:
        adjusted_probabilities.append(1 - prob)  # Probability of class 0
    else:
        adjusted_probabilities.append(prob)  # Probability of class 1

print(
    f"Predicted Label: {predicted_labels[0]}, Predicted Probability: {adjusted_probabilities[0]}"
)

f1 = f1_score(y_test, predicted_labels)

recall = recall_score(y_test, predicted_labels)

confusion_matrix = confusion_matrix(y_test, predicted_labels)

TP = confusion_matrix[1, 1]
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]

print("True Positive:", TP)
print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)

print(f"F1 Score: {f1}")
print(f"Recall: {recall}")

print(classification_report(y_test, predicted_labels))

model.save("ann.h5")


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epochs")
plt.ylabel("ACC")
plt.legend(["Training", "Validation"])
plt.title("Accuracy")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("ACC")
plt.legend(["Training", "Validation"])
plt.title("Loss Function")
plt.show()
