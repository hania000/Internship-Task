import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (example using Student Performance Dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"  # Note: Actual download might need manual extraction
df = pd.read_csv('student-mat.csv', sep=';')  # Assuming binary classification for simplicity (e.g., pass/fail based on G3)
df['pass'] = df['G3'] > 10  # Binary target: pass if G3 > 10
df['pass'] = df['pass'].astype(int)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (example using Student Performance Dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"  # Note: Actual download might need manual extraction
df = pd.read_csv('student-mat.csv', sep=';')  # Assuming binary classification for simplicity (e.g., pass/fail based on G3)
df['pass'] = df['G3'] > 10  # Binary target: pass if G3 > 10
df['pass'] = df['pass'].astype(int)
X = df.drop(['G3', 'pass'], axis=1)
y = df['pass']

# Convert categorical variables to numeric
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_pred = best_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()