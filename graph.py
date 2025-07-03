import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
df = pd.read_csv("./data/diabetes_prediction_dataset.csv")

# Data Preprocessing
enc = OrdinalEncoder()
df["smoking_history"] = enc.fit_transform(df[["smoking_history"]])
df["gender"] = enc.fit_transform(df[["gender"]])

# Define Independent and Dependent Variables
x = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Train models and store accuracy
accuracies = {}
for model_name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracies[model_name] = metrics.accuracy_score(y_test, y_pred)

# Convert to DataFrame
accuracy_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])

# Plot Model Accuracies
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=accuracy_df, palette='coolwarm')
plt.title('Model Accuracies for Diabetes Prediction')
plt.xlabel('Accuracy Score')
plt.ylabel('Model')
plt.xlim(0, 1)
plt.show()

# Train Random Forest to get feature importances
rf_model = RandomForestClassifier().fit(x_train, y_train)
importances = rf_model.feature_importances_
feature_names = x.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importances in Diabetes Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
