import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("data.csv")

X = df[['Socioeconomic Score', 'Study Hours', 'Sleep Hours', 'Attendance (%)']]
y = (df['Grades'] < 50).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(eval_metric='logloss', importance_type='gain')
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print(f"\nXGBoost Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
plot_importance(xgb, importance_type='gain', ax=plt.gca())
plt.title("XGBoost Feature Importance")
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not At Risk', 'At Risk'], yticklabels=['Not At Risk', 'At Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

def generate_risk_report(student_id):
    student = df.iloc[student_id]
    print(f"\nRisk Report for Student {student_id}")
    print(f"Socioeconomic Score: {student['Socioeconomic Score']}, Study Hours: {student['Study Hours']}, Sleep Hours: {student['Sleep Hours']}, Attendance: {student['Attendance (%)']}%")
    print(f"Grades: {student['Grades']}")
    risk_status = 'At Risk' if y.iloc[student_id] == 1 else 'Not At Risk'
    print(f"Predicted Status: {risk_status}")
    
try:
    student_id = int(input("Enter the student number (index) for the risk report: "))
    generate_risk_report(student_id)
except ValueError:
    print("Invalid input! Please enter a valid integer for the student number.")
except IndexError:
    print("Student number out of range! Please enter a valid student number.")
