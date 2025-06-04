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


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

if __name__ == "__main__":
    
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    df = pd.read_csv("StudentsPerformance.csv")

    df['Cleaned_Feedback'] = df.get('test preparation course', "").apply(preprocess_text)

    bigram = Phrases(df['Cleaned_Feedback'], min_count=5, threshold=10)
    trigram = Phrases(bigram[df['Cleaned_Feedback']], threshold=10)

    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    df['Cleaned_Feedback'] = df['Cleaned_Feedback'].apply(lambda x: trigram_mod[bigram_mod[x]])

    dictionary = Dictionary(df['Cleaned_Feedback'])
    corpus = [dictionary.doc2bow(text) for text in df['Cleaned_Feedback']]

    coherence_scores = []
    topic_range = range(2, 6)

    for num_topics in topic_range:
        model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, iterations=400, alpha='auto', eta='auto')
        coherence_model = CoherenceModel(model=model, texts=df['Cleaned_Feedback'], dictionary=dictionary, coherence='c_v')
        coherence_scores.append(coherence_model.get_coherence())

    optimal_topics = topic_range[np.argmax(coherence_scores)]
    lda_model = LdaModel(corpus, num_topics=optimal_topics, id2word=dictionary, passes=15, iterations=400, alpha='auto', eta='auto')

    def get_topic_distribution(text):
        bow = dictionary.doc2bow(text)
        topic_probs = lda_model.get_document_topics(bow, minimum_probability=0)
        weighted_sum = sum(prob * topic for topic, prob in topic_probs)
        return weighted_sum / len(topic_probs) if topic_probs else 0

    df['Topic_Feature'] = df['Cleaned_Feedback'].apply(get_topic_distribution).astype(float)

    print(f"LDA Model Perplexity: {lda_model.log_perplexity(corpus)}")
    coherence_model_lda = CoherenceModel(model=lda_model, texts=df['Cleaned_Feedback'], dictionary=dictionary, coherence='c_v')
    print(f"LDA Model Coherence Score: {coherence_model_lda.get_coherence()}")

    X = df[['math score', 'reading score', 'writing score', 'Topic_Feature']]
    y = (df[['math score', 'reading score', 'writing score']].mean(axis=1) < 50).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBClassifier(eval_metric='logloss', importance_type='gain')
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)

    print(f"XGBoost Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
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
        print(f"Math Score: {student['math score']}, Reading Score: {student['reading score']}, Writing Score: {student['writing score']}")
        print(f"Topic Score: {student['Topic_Feature']}")
        risk_status = 'At Risk' if y.iloc[student_id] == 1 else 'Not At Risk'
        print(f"Predicted Status: {risk_status}")
        
    try:
        student_id = int(input("Enter the student number (index) for the risk report: "))
        generate_risk_report(student_id)
    except ValueError:
        print("Invalid input! Please enter a valid integer for the student number.")
    except IndexError:
        print("Student number out of range! Please enter a valid student number.")

    generate_risk_report()
