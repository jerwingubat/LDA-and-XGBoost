import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from gensim.models.ldamodel import LdaModel
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gensim.models import CoherenceModel
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv("Fully_Cleaned_Student_Performance_Data.csv")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return words

df['Cleaned_Feedback'] = df['School_Environment_Feedback'].apply(preprocess_text)

bigram = Phrases(df['Cleaned_Feedback'], min_count=5, threshold=10)
trigram = Phrases(bigram[df['Cleaned_Feedback']], threshold=10)
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

df['Cleaned_Feedback'] = df['Cleaned_Feedback'].apply(lambda x: trigram_mod[bigram_mod[x]])

dictionary = Dictionary(df['Cleaned_Feedback'])
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(text) for text in df['Cleaned_Feedback']]

coherence_scores = []
topic_range = range(3, 8)

for num_topics in topic_range:
    model = LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=10,
        iterations=300,
        random_state=42
    )
    coherence_model = CoherenceModel(
        model=model,
        texts=df['Cleaned_Feedback'],
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_scores.append(coherence_model.get_coherence())

optimal_topics = topic_range[np.argmax(coherence_scores)]
print(f"Optimal number of topics: {optimal_topics}")

lda_model = LdaModel(
    corpus,
    num_topics=optimal_topics,
    id2word=dictionary,
    passes=15,
    iterations=400,
    random_state=42
)

topic_features = []
for i, text in enumerate(df['Cleaned_Feedback']):
    bow = dictionary.doc2bow(text)
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
    topic_dist_sorted = sorted(topic_dist, key=lambda x: x[0])
    topic_features.append([prob for _, prob in topic_dist_sorted])

for i in range(optimal_topics):
    df[f'Topic_{i+1}'] = [vec[i] for vec in topic_features]

features = [
    'Age', 'GWA', 'Attendance_Rate', 'Library_Usage_Hours',
    'Counseling_Sessions', 'Gender', 'Course', 'Year_Level',
    'Scholarship_Status'
] + [f'Topic_{i+1}' for i in range(optimal_topics)]

df = pd.get_dummies(df, columns=['Gender', 'Course', 'Year_Level', 'Scholarship_Status'])

le = LabelEncoder()
df['Academic_Standing_Encoded'] = le.fit_transform(df['Academic_Standing'])

X = df[features]
y = df['Academic_Standing_Encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print(f"\nXGBoost Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

plt.figure(figsize=(12, 8))
plot_importance(xgb, max_num_features=15, importance_type='weight')
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred, labels=le.transform(le.classes_))
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

def generate_risk_report(student_id):
    try:
        student = df.iloc[student_id]
        print(f"\nRisk Report for Student ID: {student['Student_ID']}")
        print(f"Academic Standing: {student['Academic_Standing']}")
        print(f"Age: {student['Age']}, Course: {student['Course']}, Year: {student['Year_Level']}")
        print(f"GWA: {student['GWA']:.2f}, Attendance: {student['Attendance_Rate']:.2f}%")
        print(f"Library Hours: {student['Library_Usage_Hours']:.2f}, Counseling Sessions: {student['Counseling_Sessions']}")

        print("\nKey Feedback Topics:")
        for i in range(optimal_topics):
            print(f"Topic {i+1}: {student[f'Topic_{i+1}']:.4f}")
        print(f"\nFeedback Excerpt: {student['School_Environment_Feedback'][:100]}...")
        
    except IndexError:
        print(f"Error: Student ID {student_id} is out of range")
try:
    student_id = int(input("\nEnter student index (0-999) for detailed report: "))
    generate_risk_report(student_id)
except ValueError:
    print("Invalid input! Please enter a number between 0-999")