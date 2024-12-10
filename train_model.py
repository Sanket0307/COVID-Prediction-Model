import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imblearn.over_sampling import SMOTE
import numpy as np

# Load your dataset
data = pd.read_csv('Cleaned-Data.csv')  # Adjust the path as needed

# Check the columns available in the dataset
print("Available columns:", data.columns)

# Prepare the target variable based on your actual classification column
# Replace 'Severity_None' with the appropriate column that indicates COVID status if needed
data['CLASIFFICATION_FINAL'] = data['Severity_None'].apply(lambda x: 0 if x == 1 else 1)  # 0: Non-COVID, 1: COVID

# One-hot encoding for categorical variables
data = pd.get_dummies(data, columns=['Gender_Female', 'Gender_Male', 'Gender_Transgender'], drop_first=True)

# Define feature columns
feature_columns = [
    'Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
    'None_Sympton', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea',
    'None_Experiencing', 'Severity_Mild', 'Severity_Moderate', 'Severity_None',
    'Severity_Severe', 'Contact_Dont-Know', 'Contact_No', 'Contact_Yes'
] + [col for col in data.columns if col.startswith('Age_')]

X = data[feature_columns]  # Features
y = data['CLASIFFICATION_FINAL']  # Target variable

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

# Grid search for optimal parameters
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(best_model, 'covid_symptom_prediction_model.pkl')
print("Model trained and saved as 'covid_symptom_prediction_model.pkl'")

# Feature importance analysis
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. {feature_columns[indices[f]]} ({importances[indices[f]]})")
