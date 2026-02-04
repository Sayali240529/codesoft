

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv("creditcard.csv")

print("Dataset Shape:", data.shape)
print(data.head())


X = data.drop("Class", axis=1)
y = data["Class"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


sample_transaction = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample_transaction)

if prediction[0] == 1:
    print("\nTransaction is FRAUD")
else:
    print("\nTransaction is LEGIT")