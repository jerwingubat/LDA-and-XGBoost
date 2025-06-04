import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gensim.models import CoherenceModel
import multiprocessing

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("StudentsPerformance.csv")

    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    if 'test preparation course' in df.columns:
        df['Cleaned_Feedback'] = df['test preparation course'].apply(preprocess_text)
    else:
        df['Cleaned_Feedback'] = ""


    texts = [text.split() for text in df['Cleaned_Feedback']]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

    def get_topic_distribution(text):
        bow = dictionary.doc2bow(text.split())
        topics = lda_model.get_document_topics(bow)
        return topics[0][1] if topics else 0

    df['Topic_Feature'] = df['Cleaned_Feedback'].apply(get_topic_distribution).astype(float)


    perplexity = lda_model.log_perplexity(corpus)
    print(f"LDA Model Perplexity: {perplexity}")

    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    print(f"LDA Model Coherence Score: {coherence_score}")

    X = df[['math score', 'reading score', 'writing score', 'Topic_Feature']]
    y = (df[['math score', 'reading score', 'writing score']].mean(axis=1) < 50).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBClassifier(eval_metric='logloss', importance_type='gain')
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Model Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    plot_importance(xgb, importance_type='gain', ax=plt.gca())
    plt.title("XGBoost Feature Importance")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not At Risk', 'At Risk'], yticklabels=['Not At Risk', 'At Risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    def generate_risk_report(student_id):
        student = df.iloc[student_id]
        print(f"\nRisk Report for Student {student_id}")
        print(f"Math Score: {student['math score']}, Reading Score: {student['reading score']}, Writing Score: {student['writing score']}")
        print(f"Topic Score: {student['Topic_Feature']}")
        risk_status = 'At Risk' if y.iloc[student_id] == 1 else 'Not At Risk'
        print(f"Predicted Status: {risk_status}")

    # User input for the student ID
    try:
        student_id = int(input("Enter the student number (index) for the risk report: "))
        generate_risk_report(student_id)
    except ValueError:
        print("Invalid input! Please enter a valid integer for the student number.")
    except IndexError:
        print("Student number out of range! Please enter a valid student number.")
