# =======================
# Week 1: Data Preprocessing (Load CSV)
# =======================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load dataset from CSV
# Make sure '"House Prices Prediction Dataset.csv' is in your working directory
df = pd.read_csv("House Prices Prediction Dataset.csv")

# 2. Inspect dataset
print("Dataset Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print(df.head())

# 3. Encode binary categorical variables
binary_cols = ["price","area","bedroom"]
print(df.columns)
for col in binary_cols:
    df[col] = df[col].map({'yes':1, 'no':0})

# 4. One-hot encode furnishingstatus
df = pd.get_dummies(df, columns=['furnishingstatus'], prefix='furn')

# 5. Feature-target split
X = df.drop('price', axis=1)
y = df['price']

# 6. Scale numeric columns
num_cols = ['area','bedrooms','bathrooms','stories','parking']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# 8. Save preprocessed dataset
df.to_csv("preprocessed_housing.csv", index=False)
print("Preprocessed dataset saved as preprocessed_housing.csv")