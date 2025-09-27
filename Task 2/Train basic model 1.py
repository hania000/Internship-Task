# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import matplotlib.pyplot as plt

# Load your preprocessed dataset
# Assuming 'data' is your DataFrame and 'target' is the target variable
data = pd.read_csv('House Prices Prediction Dataset.csv')
X = data.drop('price', axis=1)
y = data['price']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Linear Regression (for regression problems)
if y.dtype.kind in 'bifc':  # Check if target is continuous
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    print(f"Linear Regression RMSE: {mean_squared_error(y_test, y_pred_lr, squared=False)}")

## Logistic Regression (for classification problems)
if y.dtype.kind in 'O':  # Check if target is categorical
    model_log = LogisticRegression(max_iter=1000)
    model_log.fit(X_train, y_train)
    y_pred_log = model_log.predict(X_test)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log)}")
    print(f"Logistic Regression F1 Score: {f1_score(y_test, y_pred_log, average='weighted')}")

## Decision Tree
if y.dtype.kind in 'bifc':  # Regression
    model_dt = DecisionTreeRegressor()
elif y.dtype.kind in 'O':  # Classification
    model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
if y.dtype.kind in 'bifc':
    print(f"Decision Tree RMSE: {mean_squared_error(y_test, y_pred_dt, squared=False)}")
else:
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt)}")

# Visualization example (training vs test for a metric like accuracy)
# This part needs adjustment based on your specific task and data type
plt.bar(['Train', 'Test'], [accuracy_score(y_train, model_log.predict(X_train)), accuracy_score(y_test, y_pred_log)])
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Training vs Test Accuracy')
plt.show()